"""
Utilidades para optimización multi-objetivo de moléculas en espacio latente VAE.
Objetivos: QED (↑), SA (↓), Lipinski (↑)  →  pymoo minimiza [-QED, SA, -Lipinski].
"""

import re, os, sys, glob, functools
import torch
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import QED as QED_module, Descriptors, Lipinski
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.gauss import GaussianMutation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'SA_Score'))
import sascorer
from vae_model_lstm import MolecularVAE_LSTM
RDLogger.DisableLog('rdApp.*')

# ─── Configuración ───────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(ROOT_DIR, "SMILES_LSTM_2_256_300_lr1e4_b64.pth")
MOSES_CSV   = os.path.join(ROOT_DIR, "data", "moses.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 100

# Bounds teóricos por objetivo [-QED, SA, -Lip] para normalizar el HV a [0,1]^3:
#   -QED ∈ [-1, 0],  SA ∈ [1, 10],  -Lip ∈ [-1, 0]
F_MIN   = np.array([-1.0, 1.0, -1.0])   # mejor caso por objetivo
F_RANGE = np.array([ 1.0, 9.0, 1.0])    # peor - mejor
# Ref point HV en el espacio normalizado: 10% más allá del peor (1.0) en cada eje.
# Los 3 objetivos pesan igual y el HV queda en [0, 1.1^3 ≈ 1.331].
HV_REF    = np.array([1.1, 1.1, 1.1])
# Penalización para inválidas: fuera del hipercubo de referencia (escala cruda).
INVALID_F = [1.0, 12.0, 1.0]

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|<|\*|\$|%[0-9]{2}|[0-9])"
)


# ─── Operadores genéticos ────────────────────────────────────────────────────

CROSSOVERS = {
    'sbx': lambda: SBX(prob=0.9, eta=20),
    'pcx': lambda: PCX(eta=0.1, zeta=0.1),
}
MUTATIONS = {
    'pm':    lambda: PM(eta=20),
    'gauss': lambda: GaussianMutation(sigma=0.1),
}


def get_operators(crossover, mutation):
    """Instancia los operadores de pymoo a partir de sus nombres."""
    return CROSSOVERS[crossover](), MUTATIONS[mutation]()


BASELINE_COMBO = ('sbx', 'pm')   # combo canónico: comparación entre algoritmos + baseline de la ablación


def get_results_dir(crossover, mutation):
    """Cada combinación de operadores escribe en results/<crossover>_<mutation>/.
    sbx_pm es el combo canónico (comparación entre algoritmos y baseline de la
    ablación de operadores); no recibe trato especial en disco."""
    return os.path.join(RESULTS_DIR, f"{crossover}_{mutation}")



# ─── VAE: cargar, encodear, decodificar ──────────────────────────────────────

def load_model():
    """Carga modelo VAE desde checkpoint y retorna (model, stoi, itos, latent_dim)."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
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
    """Codifica lista de SMILES → vectores latentes μ. Descarta no-tokenizables."""
    mus = []
    with torch.no_grad():
        for smi in smiles_list:
            t = _smiles_to_tensor(smi, stoi)
            if t is None:
                continue
            _, (h, _) = model.encoder_rnn(model.embedding(t.unsqueeze(0).to(DEVICE)))
            mu = model.fc_mu(h[-1])
            mus.append(mu.squeeze(0).cpu().numpy())
    return np.array(mus)


def decode_z_batch(model, z_np, stoi, itos):
    """Decodifica un lote de vectores latentes z → lista de SMILES canónicos
    (argmax greedy, batcheado). Cada entrada es None si el SMILES es inválido.

    Equivale a aplicar la decodificación greedy individual a cada fila de z,
    pero ejecuta los pasos del LSTM sobre todo el lote a la vez (batch=n en
    lugar de n decodes con batch=1), lo que amortiza el overhead por paso."""
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


@functools.lru_cache(maxsize=1)
def _load_moses_df():
    """Carga y cachea MOSES CSV (~1.9M filas)."""
    return pd.read_csv(MOSES_CSV)


def load_seed_mus(model, stoi, n_samples, run_id):
    """Encodea n_samples moléculas de MOSES train como población inicial.
    Sobremuestrea 20% para compensar descartes por tokenización."""
    df = _load_moses_df()
    train_smi = df[df['SPLIT'] == 'train']['SMILES'].dropna()
    pool = train_smi.sample(int(n_samples * 1.2), random_state=run_id).tolist()
    mus = encode_smiles(model, pool, stoi)
    return mus[:n_samples]


# ─── Propiedades moleculares ─────────────────────────────────────────────────

def lipinski_score(mol):
    """Score de Lipinski (Rule of Five) en {0, 0.25, 0.5, 0.75, 1.0}.
    Suma 0.25 por cada una de las 4 condiciones que se cumple."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)

    score = (0.25 * (mw   <= 500)
           + 0.25 * (logp <= 5)
           + 0.25 * (hbd  <= 5)
           + 0.25 * (hba  <= 10))

    return round(score, 4)


@functools.lru_cache(maxsize=None)
def calc_properties(smi):
    """Calcula propiedades de un SMILES. Retorna dict o None si inválido.

    Cacheado por string SMILES: a medida que la población converge, muchos
    vectores latentes distintos decodifican al mismo SMILES, así que el cálculo
    RDKit (QED/SA/Lipinski) se reutiliza en lugar de recomputarse cada vez.
    Los callers solo leen el dict (build_pareto copia con dict.update sobre otro
    objeto), por lo que compartir la instancia cacheada es seguro."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    return {
        'smiles': smi,
        'qed': QED_module.qed(mol),
        'sa': sascorer.calculateScore(mol),
        'lipinski': lipinski_score(mol),
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
    }


# ─── Problema pymoo ──────────────────────────────────────────────────────────

class MolecularLatentProblem(Problem):
    """Optimización tri-objetivo en espacio latente VAE.
    F = [-QED, SA, -Lipinski] → pymoo minimiza los 3.

    Problema vectorizado: pymoo entrega la población como matriz [n, latent_dim]
    y se decodifica todo el lote en una sola pasada batcheada del LSTM."""

    def __init__(self, model, stoi, itos, latent_dim):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.eval_log = []
        super().__init__(n_var=latent_dim, n_obj=3, xl=-5.0, xu=5.0)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = decode_z_batch(self.model, x, self.stoi, self.itos)
        F = np.empty((len(smiles), self.n_obj), dtype=float)
        for i, smi in enumerate(smiles):
            props = calc_properties(smi)
            if props is None:
                F[i] = INVALID_F
                self.eval_log.append({
                    'smiles': None, 'qed': None, 'sa': None,
                    'lipinski': None, 'valid': False,
                })
            else:
                F[i] = (-props['qed'], props['sa'], -props['lipinski'])
                self.eval_log.append({
                    'smiles': props['smiles'], 'qed': props['qed'],
                    'sa': props['sa'], 'lipinski': props['lipinski'],
                    'valid': True,
                })
        out["F"] = F


class NormalizedMolecularLatentProblem(MolecularLatentProblem):
    """MolecularLatentProblem con normalización estática de objetivos.

    Normaliza F a [0,1]^3 usando bounds teóricos fijos:
      -QED ∈ [-1, 0],  SA ∈ [1, 10],  -Lipinski ∈ [-1, 0]

    eval_log mantiene valores crudos → post-procesamiento comparable.
    Diseñado para MOEA/D y MOPSO donde la escala cruda de SA
    domina la descomposición Tchebycheff / velocidad de partículas.
    """

    _F_MIN   = F_MIN     # bounds teóricos compartidos con el cálculo del HV
    _F_RANGE = F_RANGE

    def _evaluate(self, x, out, *args, **kwargs):
        super()._evaluate(x, out, *args, **kwargs)
        out["F"] = (out["F"] - self._F_MIN) / self._F_RANGE


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
    """Retorna set de SMILES de entrenamiento de MOSES (para novelty)."""
    df = _load_moses_df()
    return set(df[df['SPLIT'] == 'train']['SMILES'].dropna().tolist())


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

        F = algorithm.pop.get("F")
        # El HV se mide sobre objetivos normalizados a [0,1]^3 (ver HV_REF).
        # NormalizedMolecularLatentProblem ya entrega F normalizado; el problema
        # crudo entrega [-QED, SA, -Lip] y hay que normalizarlo aquí.
        if not hasattr(self.problem, '_F_MIN'):
            F = (F - F_MIN) / F_RANGE
        try:
            hv = float(HV(ref_point=HV_REF)(F))
        except Exception:
            hv = 0.0

        valid  = [e['smiles'] for e in new if e['valid']]
        n_valid = len(valid)
        unique  = set(valid)
        n_novel = sum(1 for s in valid if s not in self.train_smiles)

        self.history.append({
            'gen': gen,
            'hv': round(hv, 6),
            'n_eval': len(new),
            'n_valid': n_valid,
            'validity': round(n_valid / len(new), 4) if new else 0.0,
            'uniqueness': round(len(unique) / n_valid, 4) if n_valid else 0.0,
            'novelty': round(n_novel / n_valid, 4) if n_valid else 0.0,
        })


# ─── Métricas: Pareto, HV, Spacing ──────────────────────────────────────────

def non_dominated(results):
    """Filtra soluciones no-dominadas usando NDS de pymoo."""
    if not results:
        return []
    F = np.array([[-r['qed'], r['sa'], -r['lipinski']] for r in results])
    front_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return [results[i] for i in front_idx]


def compute_hv(pareto):
    """Hypervolume del frente de Pareto sobre objetivos normalizados a [0,1]^3."""
    if not pareto:
        return 0.0
    F = np.array([[-r['qed'], r['sa'], -r['lipinski']] for r in pareto])
    F = (F - F_MIN) / F_RANGE
    try:
        return float(HV(ref_point=HV_REF)(F))
    except Exception:
        return 0.0


def compute_spacing(pareto):
    """Spacing de Schott normalizado (CV de distancias al vecino más cercano)."""
    if len(pareto) < 2:
        return 0.0
    F = np.array([[-r['qed'], r['sa'], -r['lipinski']] for r in pareto])
    # Normalizar ejes para compensar escalas distintas (SA~[1,10] vs QED/Lip~[0,1])
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
    """Construye frente de Pareto desde el log completo. Retorna (pareto, validity)."""
    n_valid = 0
    seen = {}
    for e in eval_log:
        if not e['valid']:
            continue
        n_valid += 1
        smi = e['smiles']
        if smi not in seen:
            seen[smi] = {
                'smiles': smi, 'qed': e['qed'],
                'sa': e['sa'], 'lipinski': e['lipinski'],
            }
    validity = round(n_valid / len(eval_log), 4) if eval_log else 0.0
    pareto = non_dominated(list(seen.values()))

    # Agregar propiedades extendidas (mw, logp, hbd, hba) para el CSV
    for m in pareto:
        props = calc_properties(m['smiles'])
        if props:
            m.update(props)

    return pareto, validity


# ─── I/O ─────────────────────────────────────────────────────────────────────

def save_metrics(path, row):
    """Escribe la fila de métricas de UNA run en su propio CSV (un archivo por
    run, no compartido): al correr en paralelo cada proceso escribe el suyo, sin
    escritura concurrente al mismo CSV. Overwrite (no append) para que un
    reintento pise su fila en vez de duplicarla; generate_summary los consolida
    en alg_dir/metrics.csv al final del job."""
    pd.DataFrame([row]).to_csv(path, index=False)


def save_molecules(pareto, run_dir):
    """Guarda moléculas del frente de Pareto en CSV.

    Aunque el frente esté vacío escribe el CSV (solo header): la run terminó de
    verdad y el .sh marca el .done por la existencia de molecules.csv. Si no lo
    escribiéramos, una run con frente vacío nunca generaría el archivo y se
    relanzaría en cada job para siempre."""
    cols = ['smiles', 'qed', 'sa', 'lipinski', 'mw', 'logp', 'hbd', 'hba']
    out = os.path.join(run_dir, "molecules.csv")
    if not pareto:
        pd.DataFrame(columns=cols).to_csv(out, index=False)
        return
    df = pd.DataFrame(pareto).sort_values('qed', ascending=False)
    df[[c for c in cols if c in df.columns]].to_csv(out, index=False)


def save_tracking(tracker, run_dir):
    """Guarda convergencia por generación y log completo de evaluaciones."""
    pd.DataFrame(tracker.history).to_csv(
        os.path.join(run_dir, "convergence.csv"), index=False)
    pd.DataFrame(tracker.problem.eval_log).to_csv(
        os.path.join(run_dir, "all_molecules.csv.gz"),
        index=False, compression='gzip')


def generate_summary(alg_name, pop_size, results_dir=None):
    """Genera y muestra resumen estadístico (media ± std) de todas las runs."""
    base = results_dir if results_dir is not None else RESULTS_DIR
    alg_dir = os.path.join(base, alg_name, f"pop{pop_size}")

    # Consolida los metrics.csv por-run en uno por config (para el repo de
    # graficación), regenerado de cero: refleja exacto las runs presentes.
    run_files = sorted(glob.glob(os.path.join(alg_dir, "run_*", "metrics.csv")))
    if not run_files:
        print(f"ERROR: no hay run_*/metrics.csv en {alg_dir}")
        return
    df = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
    if 'run' in df.columns:
        df = df.sort_values('run').reset_index(drop=True)
    df.to_csv(os.path.join(alg_dir, "metrics.csv"), index=False)

    n = len(df)
    print(f"\n{'='*55}")
    print(f"  {alg_name} pop={pop_size} ({n} runs)")
    print(f"{'='*55}")
    for col, fmt in [('hypervolume', '.4f'), ('spacing', '.4f'),
                     ('best_qed', '.4f'), ('best_sa', '.2f'),
                     ('best_lipinski', '.2f')]:
        if col in df.columns:
            print(f"  {col:15s} {df[col].mean():{fmt}} ± {df[col].std():{fmt}}")
    if 'validity' in df.columns:
        print(f"  {'validity':15s} {df['validity'].mean():.2%} ± {df['validity'].std():.2%}")
    if 'novelty' in df.columns:
        print(f"  {'novelty':15s} {df['novelty'].mean():.2%} ± {df['novelty'].std():.2%}")
    if 'n_pareto' in df.columns:
        print(f"  {'n_pareto':15s} {df['n_pareto'].mean():.1f} ± {df['n_pareto'].std():.1f}")


# ─── Post-procesamiento ─────────────────────────────────────────────────────

def postprocess_run(alg_name, pop_size, n_gen, run_id, problem, tracker, elapsed, run_dir, results_dir=None):
    """Calcula métricas, guarda resultados y genera gráficas para una run."""
    pareto, validity = build_pareto(problem.eval_log)
    hv      = compute_hv(pareto)
    spacing = compute_spacing(pareto)

    # Novelty: fracción de moléculas válidas no presentes en el training set
    valid_smiles = [e['smiles'] for e in problem.eval_log if e['valid']]
    n_novel = sum(1 for s in valid_smiles if s not in tracker.train_smiles)
    novelty = round(n_novel / len(valid_smiles), 4) if valid_smiles else 0.0

    metrics = {
        'algorithm': alg_name, 'pop_size': pop_size, 'n_gen': n_gen,
        'run': run_id + 1, 'n_pareto': len(pareto),
        'hypervolume': round(hv, 6), 'spacing': spacing,
        'validity': validity, 'novelty': novelty,
        'best_qed': round(max((r['qed'] for r in pareto), default=float('nan')), 4),
        'best_sa': round(min((r['sa'] for r in pareto), default=float('nan')), 2),
        'best_lipinski': round(max((r['lipinski'] for r in pareto), default=0), 4),
        'time_sec': round(elapsed, 1),
    }

    # metrics.csv por-run: sin escritura concurrente (ver save_metrics).
    save_metrics(os.path.join(run_dir, "metrics.csv"), metrics)
    save_molecules(pareto, run_dir)
    save_tracking(tracker, run_dir)
    # Las gráficas no se generan aquí; se regeneran desde los CSV en el
    # repo de graficación.

    return metrics, pareto, hv, spacing, validity

