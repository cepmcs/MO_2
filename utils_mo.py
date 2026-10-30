"""
Utilidades para optimización multi-objetivo de moléculas en espacio latente VAE.

Objetivos: QED (↑), SA (↓) y docking (↓) → pymoo minimiza [-QED, SA, dock].
Fsp3 va como constraint (Fsp3 ≥ FSP3_MIN).

Dos opciones, apagadas por defecto (así el experimento es el de exp4): pChEMBL como cuarto
objetivo (↑, se minimiza -pChEMBL) y logP como constraint (LOGP_MIN ≤ logP ≤ LOGP_MAX).
"""

import re, os, sys, glob, functools
import torch
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import QED as QED_module, rdMolDescriptors
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.mutation.pm import PM

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'SA_Score'))
import sascorer
from vae_model_lstm import MolecularVAE_LSTM
from docking import DockingCache, DOCK_BEST, DOCK_WORST
from pchembl import PChEMBLPredictor, PCHEMBL_MIN, PCHEMBL_MAX
RDLogger.DisableLog('rdApp.*')

# ─── Configuración ───────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(ROOT_DIR, "SMILES_LSTM_2_256_300_lr1e4_b64.pth")
MOSES_CSV   = os.path.join(ROOT_DIR, "data", "moses.csv")
MOSES_TRAIN_CACHE = os.path.join(ROOT_DIR, "data", "moses_train_smiles.pkl.gz")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "exp4")
# Las dos VAE entre las que elige experimento.py (--vae): la original y la ajustada con ligandos de MRP1.
VAE_PATHS = {
    'original': MODEL_PATH,
    'mrp1': os.path.join(ROOT_DIR, "SMILES_LSTM_2_256_300_lr1e4_b64_mrp1.pth"),
}


def results_dir(exp_name):
    """Carpeta de resultados de un experimento: results/<exp_name>."""
    return os.path.join(ROOT_DIR, "results", exp_name)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 100

# El experimento: lo que corre run_experiments.py y los defaults de experimento.py.
POP_SIZE = 100
N_GEN    = 1000     # 100.000 evaluaciones
CX_PROB  = 1.0
MUTATION = 'pm'
MUT_PROB = 0.1      # por gen

# Constraint: factible si Fsp3 ≥ FSP3_MIN.
FSP3_MIN = 0.3
# Constraint opcional de logP (el alogp de RDKit): factible si LOGP_MIN ≤ logP ≤ LOGP_MAX.
LOGP_MIN = 1.0
LOGP_MAX = 4.0

# Bounds de [-QED, SA, dock] para normalizar el HV a [0,1]^3.
F_MIN     = np.array([-1.0, 1.0, DOCK_BEST])
F_RANGE   = np.array([ 1.0, 9.0, DOCK_WORST - DOCK_BEST])
HV_REF    = np.array([1.1, 1.1, 1.1])   # 10% más allá del peor; HV ∈ [0, 1.331]
INVALID_F = [1.0, 12.0, DOCK_WORST]     # penalización de las inválidas
INVALID_G = 1.0                    # una inválida nunca es factible

# Con pChEMBL como cuarto objetivo (se minimiza -pChEMBL): las mismas cotas con una dimensión más.
F_MIN_4     = np.append(F_MIN, -PCHEMBL_MAX)
F_RANGE_4   = np.append(F_RANGE, PCHEMBL_MAX - PCHEMBL_MIN)
HV_REF_4    = np.append(HV_REF, 1.1)    # HV ∈ [0, 1.4641]
INVALID_F_4 = INVALID_F + [-PCHEMBL_MIN]
ESCALAS = {3: (F_MIN, F_RANGE, HV_REF), 4: (F_MIN_4, F_RANGE_4, HV_REF_4)}   # por nº de objetivos

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|<|\*|\$|%[0-9]{2}|[0-9])"
)


# ─── Operadores genéticos ────────────────────────────────────────────────────
# Solo se parametrizan las probabilidades; el resto queda en el default de pymoo.
# La mutación va por-gen (prob_var) con prob=1.0.

CROSSOVERS = {
    'sbx': lambda cx_prob: SBX(prob=cx_prob),
    'pcx': lambda cx_prob: PCX(prob=cx_prob),
}
MUTATIONS = {
    'pm': lambda mut_prob: PM(prob=1.0, prob_var=mut_prob),
}


def get_operators(crossover, mutation, cx_prob, mut_prob):
    """(crossover, mutation) de pymoo: cx_prob por apareamiento, mut_prob por-gen."""
    return CROSSOVERS[crossover](cx_prob), MUTATIONS[mutation](mut_prob)


def _slug(x):
    """Número → string compacto para nombres de carpeta (1.0 → '1')."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def ga_run_dir(alg_name, crossover, mutation, cx_prob, mut_prob,
               pop_size, n_gen, run_id, exp_name='exp4'):
    """Directorio de una run GA: results/<exp_name>/<ALG>/<cruce_mut>/<config>/run_k."""
    combo = f"{crossover}_{mutation}"
    cfg   = f"cx{_slug(cx_prob)}_mut{_slug(mut_prob)}_pop{pop_size}_gen{n_gen}"
    return os.path.join(results_dir(exp_name), alg_name, combo, cfg, f"run_{run_id + 1:02d}")


def get_ref_dirs(n_points, n_obj=3):
    """n_points direcciones sobre el símplex de n_obj objetivos (método energy, seed fija),
    para NSGA-III.  Determinista y ~2 s: no se cachea."""
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("energy", n_obj, n_points, seed=1)



# ─── VAE: cargar, encodear, decodificar ──────────────────────────────────────

def set_device(name):
    """Fuerza el dispositivo ('cpu' / 'cuda'). Llamar antes de load_model()."""
    global DEVICE
    DEVICE = torch.device(name)


def load_model(path=MODEL_PATH):
    """Carga modelo VAE desde checkpoint y retorna (model, stoi, itos, latent_dim)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    h    = ckpt['hyperparams']
    model = MolecularVAE_LSTM(
        vocab_size=h['vocab_size'], embed_size=h['embed'],
        hidden_size=h['hidden'], latent_size=h['latent'],
        num_layers=h.get('num_layers', 1)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['vocab_stoi'], ckpt['vocab_itos'], h['latent']


def _smiles_to_tensor(smi, stoi):
    """Tokeniza SMILES → tensor de índices [SOS, ..., EOS, PAD]. None si falla."""
    tokens = SMILES_REGEX.findall(smi)
    if not tokens:
        return None
    ids = [stoi['[SOS]']]
    for t in tokens:
        idx = stoi.get(t)
        if idx is None:
            return None
        ids.append(idx)
    ids.append(stoi['[EOS]'])
    pad = stoi.get('[PAD]', 0)
    ids = ids[:MAX_LEN] + [pad] * max(0, MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_smiles(model, smiles_list, stoi):
    """SMILES → vectores latentes μ, en lote. Descarta los no-tokenizables."""
    tensors = [t for smi in smiles_list
               if (t := _smiles_to_tensor(smi, stoi)) is not None]
    if not tensors:
        return np.array([])
    with torch.no_grad():
        batch = torch.stack(tensors).to(DEVICE)             # [B, MAX_LEN]
        _, (h, _) = model.encoder_rnn(model.embedding(batch))
        mus = model.fc_mu(h[-1])                            # [B, latent]
    return mus.cpu().numpy()


def decode_z_batch(model, z_np, stoi, itos):
    """Lote de latentes z → SMILES canónicos (argmax greedy). None si es inválido."""
    z = torch.as_tensor(np.asarray(z_np, dtype=np.float32), device=DEVICE)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    n = z.shape[0]

    h = model.decoder_input_h(z).unsqueeze(0).repeat(model.num_layers, 1, 1).contiguous()
    c = model.decoder_input_c(z).unsqueeze(0).repeat(model.num_layers, 1, 1).contiguous()
    cur = torch.full((n, 1), stoi['[SOS]'], dtype=torch.long, device=DEVICE)

    eos_idx  = stoi['[EOS]']
    special  = {'[PAD]', '[SOS]', '[EOS]', '[UNK]'}
    finished = [False] * n
    n_done   = 0
    tokens   = [[] for _ in range(n)]

    with torch.no_grad():
        for _ in range(MAX_LEN):
            emb = model.embedding(cur)
            out, (h, c) = model.decoder_rnn(emb, (h, c))
            idx = model.fc_out(out.squeeze(1)).argmax(dim=-1)   # [n]
            idx_list = idx.tolist()
            for i in range(n):
                if finished[i]:
                    continue
                ti = idx_list[i]
                if ti == eos_idx:
                    finished[i] = True
                    n_done += 1
                else:
                    tok = itos[ti]
                    if tok not in special:
                        tokens[i].append(tok)
            if n_done == n:
                break
            cur = idx.unsqueeze(1)

    results = []
    for i in range(n):
        mol = Chem.MolFromSmiles("".join(tokens[i]))
        results.append(Chem.MolToSmiles(mol) if mol else None)
    return results


def _build_moses_train_smiles():
    """Parsea el CSV de MOSES y devuelve la Series de SMILES del split 'train'."""
    df = pd.read_csv(MOSES_CSV, usecols=['SMILES', 'SPLIT'])
    return df[df['SPLIT'] == 'train']['SMILES'].dropna().reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_moses_train_smiles():
    """SMILES de train de MOSES, cacheados en MOSES_TRAIN_CACHE. Se reconstruye si
    el cache falta o moses.csv es más nuevo."""
    cache = MOSES_TRAIN_CACHE
    if (os.path.exists(cache)
            and os.path.getmtime(cache) >= os.path.getmtime(MOSES_CSV)):
        return pd.read_pickle(cache, compression='gzip')

    train_smi = _build_moses_train_smiles()
    tmp = f"{cache}.{os.getpid()}.tmp"
    try:
        train_smi.to_pickle(tmp, compression='gzip')
        os.replace(tmp, cache)                      # atómico en el mismo filesystem
    except OSError:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return train_smi


def load_seed_mus(model, stoi, n_samples, run_id):
    """Población inicial: n_samples moléculas de MOSES encodeadas.  Sobremuestrea
    20% para compensar los descartes por tokenización."""
    train_smi = _load_moses_train_smiles()
    pool = train_smi.sample(int(n_samples * 1.2), random_state=run_id).tolist()
    mus = encode_smiles(model, pool, stoi)
    return mus[:n_samples]


# ─── Propiedades moleculares ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=100_000)
def calc_properties(smi):
    """Propiedades de un SMILES, o None si es inválido.  Cacheado: la población
    converge y muchos latentes decodifican al mismo SMILES."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    # QED.properties trae de una alogp/hbd/mw/hba, que salen gratis.
    qp = QED_module.properties(mol)
    return {
        'smiles': smi,
        'qed':   QED_module.qed(mol, qedProperties=qp),   # objetivo (↑)
        'sa':    sascorer.calculateScore(mol),            # objetivo (↓)
        'fsp3':  rdMolDescriptors.CalcFractionCSP3(mol),  # constraint (↑)
        'alogp': qp.ALOGP,   # de acá para abajo, solo reporte
        'hbd':   qp.HBD,
        'mw':    qp.MW,
        'hba':   qp.HBA,
    }


# ─── Problema pymoo ──────────────────────────────────────────────────────────

class MolecularLatentProblem(Problem):
    """Optimización multiobjetivo con constraints en el espacio latente del VAE.

    F = [-QED, SA, dock]          → minimizar.  Con pchembl=True se suma -pChEMBL.
    G = [FSP3_MIN - Fsp3]         → factible si ≤ 0.  Con logp=True se suman
        [LOGP_MIN - logP, logP - LOGP_MAX].  Solo las factibles se dockean."""

    def __init__(self, model, stoi, itos, latent_dim, pchembl=False, logp=False):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.eval_log = []
        self.dock = DockingCache()
        self.pchembl = PChEMBLPredictor() if pchembl else None
        self.logp = logp
        super().__init__(n_var=latent_dim, n_obj=3 + bool(pchembl),
                         n_ieq_constr=1 + 2 * bool(logp), xl=-5.0, xu=5.0)

    def _factible(self, props):
        """Cumple Fsp3 y, si el constraint de logP está activo, también el rango de logP."""
        return props['fsp3'] >= FSP3_MIN and (
            not self.logp or LOGP_MIN <= props['alogp'] <= LOGP_MAX)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = decode_z_batch(self.model, x, self.stoi, self.itos)
        feasibles = [s for s in smiles if (p := calc_properties(s)) and self._factible(p)]
        dock = dict(zip(feasibles, self.dock(feasibles)))
        validos = [s for s in smiles if calc_properties(s)]
        pchembl = dict(zip(validos, self.pchembl(validos))) if self.pchembl else {}
        F = np.empty((len(smiles), self.n_obj), dtype=float)
        G = np.empty((len(smiles), self.n_ieq_constr), dtype=float)
        for i, smi in enumerate(smiles):
            props = calc_properties(smi)
            if props is None:
                F[i] = INVALID_F_4 if self.pchembl else INVALID_F
                G[i] = INVALID_G
                self.eval_log.append({
                    'smiles': None, 'qed': None, 'sa': None,
                    'fsp3': None, 'dock': None, 'valid': False, 'feasible': False,
                    **({'pchembl': None} if self.pchembl else {}),
                    **({'logp': None} if self.logp else {}),
                })
            else:
                F[i, :3] = (-props['qed'], props['sa'], dock.get(smi, DOCK_WORST))
                G[i, 0] = FSP3_MIN - props['fsp3']
                if self.pchembl:
                    F[i, 3] = -pchembl[smi]
                if self.logp:
                    G[i, 1:] = (LOGP_MIN - props['alogp'], props['alogp'] - LOGP_MAX)
                self.eval_log.append({
                    'smiles': props['smiles'], 'qed': props['qed'],
                    'sa': props['sa'], 'fsp3': props['fsp3'], 'dock': dock.get(smi),
                    'valid': True, 'feasible': bool(self._factible(props)),
                    **({'pchembl': pchembl[smi]} if self.pchembl else {}),
                    **({'logp': props['alogp']} if self.logp else {}),
                })
        out["F"] = F
        out["G"] = G


class LatentSampling(Sampling):
    """Sampling inicial desde vectores μ de moléculas MOSES."""
    def __init__(self, mus):
        super().__init__()
        self.mus = mus

    def _do(self, problem, n_samples, **kwargs):
        idx = np.random.choice(len(self.mus), size=n_samples,
                               replace=len(self.mus) < n_samples)
        return self.mus[idx]


# ─── Callback: tracking por generación ───────────────────────────────────────

def load_train_smiles():
    """Set de SMILES de train de MOSES, para novelty.  Libera la Series cacheada."""
    train_smi = _load_moses_train_smiles()
    smiles_set = set(train_smi)
    _load_moses_train_smiles.cache_clear()
    return smiles_set


class GenerationTracker(Callback):
    """Registra HV, validez, unicidad y novedad por generación."""

    def __init__(self, problem, train_smiles):
        super().__init__()
        self.problem = problem
        self.train_smiles = train_smiles
        self.history = []
        self._last_idx = 0

    def notify(self, algorithm):
        gen = algorithm.n_gen

        new = self.problem.eval_log[self._last_idx:]
        self._last_idx = len(self.problem.eval_log)
        for e in new:
            e['gen'] = gen

        # Solo los factibles, igual que el HV final: si no, la curva mediría algo
        # distinto de su propio punto de llegada.
        feas = algorithm.pop.get("FEAS")
        pop_feasible = algorithm.pop[feas.flatten()] if feas is not None else algorithm.pop
        if len(pop_feasible) == 0:
            hv = 0.0
        else:
            f_min, f_range, hv_ref = ESCALAS[self.problem.n_obj]
            F = (pop_feasible.get("F") - f_min) / f_range
            try:
                hv = float(HV(ref_point=hv_ref)(F))
            except Exception:
                hv = 0.0

        valid  = [e['smiles'] for e in new if e['valid']]
        n_valid = len(valid)
        unique  = set(valid)
        n_novel = sum(1 for s in valid if s not in self.train_smiles)
        n_feas  = sum(1 for e in new if e['valid'] and e.get('feasible'))

        self.history.append({
            'gen': gen,
            'hv': round(hv, 6),
            'n_feasible': n_feas,
            'feasibility': round(n_feas / n_valid, 4) if n_valid else 0.0,
            'n_eval': len(new),
            'n_valid': n_valid,
            'validity': round(n_valid / len(new), 4) if new else 0.0,
            'uniqueness': round(len(unique) / n_valid, 4) if n_valid else 0.0,
            'novelty': round(n_novel / n_valid, 4) if n_valid else 0.0,
        })


# ─── Métricas: Pareto, HV, Spacing ──────────────────────────────────────────

def _non_dominated_front(F):
    """Índices del frente no-dominado (minimización).

    Filtro estilo Kung, memoria O(frente).  El de pymoo arma una matriz O(n²) y
    con las ~100k moléculas de una run se va a OOM."""
    n = len(F)
    if n == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort(F.T[::-1])                      # asc por obj0, luego obj1, ...
    kept_idx = []
    cap = 256
    buf = np.empty((cap, F.shape[1]))                  # objetivos de los no-dominados
    m = 0
    for i in order:
        f = F[i]
        if m and np.any(np.all(buf[:m] <= f, axis=1) & np.any(buf[:m] < f, axis=1)):
            continue                                   # dominado por alguno ya guardado
        if m == cap:
            cap *= 2
            nb = np.empty((cap, F.shape[1])); nb[:m] = buf; buf = nb
        buf[m] = f; m += 1
        kept_idx.append(int(i))
    return np.array(kept_idx, dtype=int)


def _objetivos(results):
    """Matriz [-QED, SA, dock] de los resultados, con -pChEMBL al final si lo traen."""
    return np.array([[-r['qed'], r['sa'], r['dock']] + ([-r['pchembl']] if 'pchembl' in r else [])
                     for r in results], dtype=float)


def non_dominated(results):
    """Filtra las soluciones no-dominadas."""
    if not results:
        return []
    F = _objetivos(results)
    return [results[i] for i in _non_dominated_front(F)]


def compute_hv(pareto):
    """Hypervolume del frente de Pareto sobre objetivos normalizados a [0,1]^n
    (n = 3, o 4 si el frente trae pChEMBL)."""
    if not pareto:
        return 0.0
    F = _objetivos(pareto)
    f_min, f_range, hv_ref = ESCALAS[F.shape[1]]
    F = (F - f_min) / f_range
    try:
        return float(HV(ref_point=hv_ref)(F))
    except Exception:
        return 0.0


def compute_spacing(pareto):
    """Spacing de Schott normalizado (CV de distancias al vecino más cercano)."""
    if len(pareto) < 2:
        return 0.0
    F = _objetivos(pareto)
    # Normalizar los ejes: tienen escalas distintas.
    ranges = F.max(axis=0) - F.min(axis=0)
    ranges[ranges == 0] = 1.0
    F_norm = F / ranges
    dmin = []
    for i in range(len(F_norm)):
        d = np.sqrt(np.sum((F_norm - F_norm[i]) ** 2, axis=1))
        d[i] = np.inf
        dmin.append(np.min(d))
    dmin = np.array(dmin)
    mu = np.mean(dmin)
    return round(float(np.std(dmin) / mu), 6) if mu > 0 else 0.0


def build_pareto(eval_log):
    """Frente de Pareto del log completo → (pareto, validity, feasibility).
    Solo compiten las factibles."""
    n_valid = 0
    n_feasible = 0
    seen = {}
    for e in eval_log:
        if not e['valid']:
            continue
        n_valid += 1
        if not e.get('feasible', True):
            continue
        n_feasible += 1
        smi = e['smiles']
        if smi not in seen:
            seen[smi] = {
                'smiles': smi, 'qed': e['qed'],
                'sa': e['sa'], 'fsp3': e['fsp3'], 'dock': e['dock'],
                **{k: e[k] for k in ('pchembl', 'logp') if k in e},
            }
    validity = round(n_valid / len(eval_log), 4) if eval_log else 0.0
    feasibility = round(n_feasible / n_valid, 4) if n_valid else 0.0
    pareto = non_dominated(list(seen.values()))

    return pareto, validity, feasibility


# ─── I/O ─────────────────────────────────────────────────────────────────────

def save_metrics(path, row):
    """Métricas de una run en su propio CSV; consolidate_all los junta después."""
    pd.DataFrame([row]).to_csv(path, index=False)


def save_molecules(pareto, run_dir, extras=()):
    """Frente de Pareto → molecules.csv, escritura atómica.  Se escribe aunque esté
    vacío: su existencia es la señal de run completa.  extras: columnas de más
    ('pchembl', 'logp') cuando el experimento las usa."""
    cols = ['smiles', 'qed', 'sa', 'fsp3', 'dock', *extras]
    out = os.path.join(run_dir, "molecules.csv")
    tmp = out + ".tmp"
    if not pareto:
        pd.DataFrame(columns=cols).to_csv(tmp, index=False)
    else:
        df = pd.DataFrame(pareto).sort_values('sa', ascending=True)
        df[[c for c in cols if c in df.columns]].to_csv(tmp, index=False)
    os.replace(tmp, out)


def save_tracking(tracker, run_dir):
    """Convergencia por generación y log completo de evaluaciones."""
    pd.DataFrame(tracker.history).to_csv(
        os.path.join(run_dir, "convergence.csv"), index=False)
    pd.DataFrame(tracker.problem.eval_log).to_csv(
        os.path.join(run_dir, "all_molecules.csv.gz"),
        index=False, compression='gzip', float_format='%.4f')


def consolidate_all(exp_name='exp4'):
    """Junta los run_*/metrics.csv en <results>/all_metrics.csv, una fila por run
    con sus hiperparámetros como columnas."""
    base = results_dir(exp_name)
    files = sorted(glob.glob(os.path.join(base, "**", "run_*", "metrics.csv"),
                             recursive=True))
    if not files:
        print(f"ERROR: no hay run_*/metrics.csv bajo {base}")
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = os.path.join(base, "all_metrics.csv")
    df.to_csv(out, index=False)
    print(f"Consolidado: {len(df)} runs de {len(files)} archivos → {out}")
    return df


# ─── Post-procesamiento ─────────────────────────────────────────────────────

def postprocess_run(alg_name, pop_size, n_gen, run_id, problem, tracker, elapsed, run_dir, hp=None):
    """Calcula métricas y guarda los CSVs de una run.  hp: los hiperparámetros
    de la corrida, que van como columnas de metrics.csv."""
    pareto, validity, feasibility = build_pareto(problem.eval_log)
    hv      = compute_hv(pareto)
    spacing = compute_spacing(pareto)

    # Novelty: válidas que no están en el training set.
    valid_smiles = [e['smiles'] for e in problem.eval_log if e['valid']]
    n_novel = sum(1 for s in valid_smiles if s not in tracker.train_smiles)
    novelty = round(n_novel / len(valid_smiles), 4) if valid_smiles else 0.0

    metrics = {
        'algorithm': alg_name, 'pop_size': pop_size, 'n_gen': n_gen,
        **(hp or {}),                       # los hiperparámetros de la corrida
        'run': run_id + 1, 'n_pareto': len(pareto),
        'hypervolume': round(hv, 6), 'spacing': spacing,
        'validity': validity, 'feasibility': feasibility, 'novelty': novelty,
        'best_qed': round(max((r['qed'] for r in pareto), default=float('nan')), 4),
        'best_sa': round(min((r['sa'] for r in pareto), default=float('nan')), 2),
        'best_dock': round(min((r['dock'] for r in pareto), default=float('nan')), 2),
        'n_docked': problem.dock.n_docked,
        'n_dock_fail': problem.dock.fallos,
        'mean_fsp3': round(float(np.mean([r['fsp3'] for r in pareto])), 4) if pareto else float('nan'),
        'time_sec': round(elapsed, 1),
    }
    # columnas de más cuando el experimento usa pChEMBL (objetivo) o logP (constraint)
    extras = []
    if problem.pchembl:
        extras.append('pchembl')
        metrics['best_pchembl'] = round(max(r['pchembl'] for r in pareto), 4) if pareto else float('nan')
    if problem.logp:
        extras.append('logp')
        metrics['mean_logp'] = round(float(np.mean([r['logp'] for r in pareto])), 4) if pareto else float('nan')

    # molecules.csv al final: es la señal de run completa.
    save_metrics(os.path.join(run_dir, "metrics.csv"), metrics)
    save_tracking(tracker, run_dir)
    save_molecules(pareto, run_dir, extras)

    return metrics, pareto, hv, spacing, validity

