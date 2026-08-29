"""
Utilidades para optimización multi-objetivo de moléculas en espacio latente VAE.
Objetivos: QED (↑), SA (↓)  →  pymoo minimiza [-QED, SA].
Fsp3 (↑) NO es objetivo: entra como CONSTRAINT de desigualdad (Fsp3 ≥ FSP3_MIN),
así el frente de Pareto queda en 2D y la saturación solo filtra qué es admisible.
QED ∈ [0,1] (druglikeness agregada, rdkit.Chem.QED.qed); Fsp3 ∈ [0,1] (fracción de
carbonos sp3, CalcFractionCSP3); SA ∈ [1,10] (accesibilidad sintética).
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
from pymoo.operators.mutation.gauss import GaussianMutation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'SA_Score'))
import sascorer
from vae_model_lstm import MolecularVAE_LSTM
RDLogger.DisableLog('rdApp.*')

# ─── Configuración ───────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(ROOT_DIR, "SMILES_LSTM_2_256_300_lr1e4_b64.pth")
MOSES_CSV   = os.path.join(ROOT_DIR, "data", "moses.csv")
MOSES_TRAIN_CACHE = os.path.join(ROOT_DIR, "data", "moses_train_smiles.pkl.gz")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 100

# Umbral del constraint de saturación: factible si Fsp3 ≥ FSP3_MIN  →  G = FSP3_MIN - Fsp3.
# Como Fsp3 dejó de ser objetivo, nada empuja por encima del umbral: las soluciones se
# estacionan EN el borde, así que este valor es el Fsp3 que se obtiene, no un piso.
# Referencias medidas sobre los datos del proyecto, para interpretar el valor elegido:
#   etapa lipinski (Fsp3 sin optimizar)  media 0.073, solo 0.6% del frente llega a 0.3
#   MOSES train (lo que aprendió el VAE) mediana 0.333, media 0.350; 56% cumple ≥ 0.3
#   etapa con Fsp3 como objetivo          media 0.61-0.70 por frente
# Costo en QED medio según banda (frentes de la etapa a 3 objetivos): 0.906 en
# [0.25,0.35) vs 0.887 en [0.35,0.45).  El máximo de QED (0.948) no cambia con la banda.
FSP3_MIN = 0.3

# Bounds teóricos por objetivo [-QED, SA] para normalizar el HV a [0,1]^2:
#   -QED ∈ [-1, 0],  SA ∈ [1, 10]
F_MIN   = np.array([-1.0, 1.0])   # mejor caso por objetivo
F_RANGE = np.array([ 1.0, 9.0])   # peor - mejor
# Ref point HV en el espacio normalizado: 10% más allá del peor (1.0) en cada eje.
# Los 2 objetivos pesan igual y el HV queda en [0, 1.1^2 = 1.21].
# OJO: no comparable con el HV de los experimentos a 3 objetivos (máx 1.331).
HV_REF    = np.array([1.1, 1.1])
# Penalización para inválidas: fuera del hipercubo de referencia (escala cruda).
INVALID_F = [1.0, 12.0]
# Violación asignada a las inválidas: un SMILES que no parsea no puede ser factible,
# si no competiría de igual a igual contra moléculas reales que sí cumplen el umbral.
INVALID_G = 1.0

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|<|\*|\$|%[0-9]{2}|[0-9])"
)


# ─── Operadores genéticos ────────────────────────────────────────────────────
# Sensibilidad de hiperparámetros: SOLO se barren las PROBABILIDADES (cruce y
# mutación). Todo lo demás queda en el default de pymoo:
#   SBX eta=15 · PCX eta=zeta=0.1 · PM eta=20 · Gauss sigma=0.1
# La mutación se barre por-gen (prob_var); prob=1.0 (siempre se intenta mutar),
# así el nº de genes mutados lo controla solo prob_var y PM/Gauss son comparables.

CROSSOVERS = {
    'sbx': lambda cx_prob: SBX(prob=cx_prob),
    'pcx': lambda cx_prob: PCX(prob=cx_prob),
}
MUTATIONS = {
    'pm':    lambda mut_prob: PM(prob=1.0, prob_var=mut_prob),
    'gauss': lambda mut_prob: GaussianMutation(prob=1.0, prob_var=mut_prob, sigma=0.1),
}


def get_operators(crossover, mutation, cx_prob, mut_prob):
    """Instancia (crossover, mutation) de pymoo con las probabilidades barridas.
    cx_prob = prob de cruce (por apareamiento); mut_prob = prob de mutación por-gen."""
    return CROSSOVERS[crossover](cx_prob), MUTATIONS[mutation](mut_prob)


def _slug(x):
    """Número → string compacto y estable para nombres de carpeta (0.9→'0.9', 1.0→'1')."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def ga_run_dir(alg_name, crossover, mutation, cx_prob, mut_prob,
               pop_size, n_gen, run_id, results_dir=None):
    """Directorio de una run GA: results/<ALG>/<cruce_mut>/<config>/run_k.
    Se agrupa por combo de operadores; el slug de config codifica las probabilidades
    y el reparto pob×gen → cada celda del grid tiene su carpeta y no se pisan.
    Única fuente de verdad del path (la usa el script y el orquestador)."""
    base = results_dir if results_dir is not None else RESULTS_DIR
    combo = f"{crossover}_{mutation}"
    cfg   = f"cx{_slug(cx_prob)}_mut{_slug(mut_prob)}_pop{pop_size}_gen{n_gen}"
    return os.path.join(base, alg_name, combo, cfg, f"run_{run_id + 1:02d}")


def cmopso_run_dir(pop_size, n_gen, elite_size, mut_prob, vel_rate, run_id,
                   results_dir=None):
    """Directorio de una run CMOPSO (pob, gen, elite_size, mutación por-gen, velocidad).

    CMOPSO no tiene w/c1/c2: su ecuación de velocidad usa coeficientes aleatorios por
    dimensión y no hay pbest, así que esas tres perillas del grid MOPSO desaparecen."""
    base = results_dir if results_dir is not None else RESULTS_DIR
    cfg = (f"pop{pop_size}_gen{n_gen}_e{_slug(elite_size)}"
           f"_mut{_slug(mut_prob)}_vel{_slug(vel_rate)}")
    return os.path.join(base, "CMOPSO", cfg, f"run_{run_id + 1:02d}")


def get_ref_dirs(n_points):
    """Direcciones de referencia (Das-Dennis uniforme, 2 objetivos) para NSGA-III /
    MOEA-D: exactamente n_points, equiespaciadas sobre el símplex.

    Con 2 objetivos el símplex es un segmento y Das-Dennis con n_points-1 particiones
    da la solución EXACTA: n_points puntos a distancia 1/(n_points-1).  La etapa a 3
    objetivos usaba Riesz s-energy porque ahí Das-Dennis no puede dar un n arbitrario,
    pero en 2D energy es un optimizador numérico que aproxima algo que ya tiene forma
    cerrada, y lo hace peor: medido con seed=1 devuelve una dirección DUPLICADA en
    p=200 y p=400, y huecos de hasta 3-6x el espaciado ideal.  El duplicado importa
    sobre todo en MOEA/D, donde cada dirección es un subproblema y pop_size = len(
    ref_dirs): dos direcciones idénticas son un slot de población desperdiciado.

    Es determinista y cuesta ~1 ms, así que no se cachea en disco (el cache anterior
    llevaba el nº de objetivos en el nombre y era una fuente de errores silenciosos)."""
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("uniform", 2, n_partitions=n_points - 1)



# ─── VAE: cargar, encodear, decodificar ──────────────────────────────────────

def set_device(name):
    """Fuerza el dispositivo de cómputo ('cpu' / 'cuda'). Llamar antes de load_model().
    El default (a nivel módulo) ya autoselecciona GPU si hay CUDA disponible."""
    global DEVICE
    DEVICE = torch.device(name)


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
    """Codifica lista de SMILES → vectores latentes μ (en lote). Descarta no-tokenizables."""
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
    """Decodifica un lote de vectores latentes z → lista de SMILES canónicos
    (argmax greedy, batcheado). Cada entrada es None si el SMILES es inválido."""
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
    """SMILES de train de MOSES (~1.6M) como Series, cacheada en MOSES_TRAIN_CACHE
    (pickle gzip). Se reconstruye si el cache no existe o si moses.csv es más nuevo.
    Escritura atómica (tmp + os.replace) para no dejar un cache a medias si se interrumpe."""
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
    """Encodea n_samples moléculas de MOSES train como población inicial.
    Sobremuestrea 20% para compensar descartes por tokenización."""
    train_smi = _load_moses_train_smiles()
    pool = train_smi.sample(int(n_samples * 1.2), random_state=run_id).tolist()
    mus = encode_smiles(model, pool, stoi)
    return mus[:n_samples]


# ─── Propiedades moleculares ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=100_000)
def calc_properties(smi):
    """Calcula propiedades de un SMILES. Retorna dict o None si inválido.

    Objetivos: QED (↑) y SA (↓).  Fsp3 (↑) NO es objetivo: entra como constraint
    (Fsp3 ≥ FSP3_MIN).  QED es la druglikeness agregada de RDKit (∈[0,1]); Fsp3 es
    la fracción de carbonos sp3 (∈[0,1], CalcFractionCSP3, devuelve 0 si la molécula
    no tiene carbonos); SA es la accesibilidad sintética (∈[1,10]).

    Cacheado por SMILES (LRU acotado): la población converge y muchos latentes
    decodifican al mismo SMILES, reusando el cálculo RDKit."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    # QED.properties calcula de una los descriptores que usa el QED; alogp/hbd/mw/hba
    # salen gratis de ahí.  Ya NO se escriben a molecules.csv (nadie los leía); quedan
    # en el dict por si hacen falta a mano, y se recalculan desde el SMILES.
    qp = QED_module.properties(mol)
    return {
        'smiles': smi,
        'qed':   QED_module.qed(mol, qedProperties=qp),   # objetivo (↑), ∈[0,1]
        'sa':    sascorer.calculateScore(mol),            # objetivo (↓), ∈[1,10]
        'fsp3':  rdMolDescriptors.CalcFractionCSP3(mol),  # constraint (↑), ∈[0,1]
        'alogp': qp.ALOGP,   # ALOGP crudo (Crippen MolLogP) — reporte
        'hbd':   qp.HBD,     # HBD crudo (CalcNumHBD) — reporte
        'mw':    qp.MW,
        'hba':   qp.HBA,
    }


# ─── Problema pymoo ──────────────────────────────────────────────────────────

class MolecularLatentProblem(Problem):
    """Optimización bi-objetivo con constraint de saturación en espacio latente VAE.

    F = [-QED, SA]            → pymoo minimiza los 2.
    G = [FSP3_MIN - Fsp3]     → factible si ≤ 0, o sea Fsp3 ≥ FSP3_MIN.

    El constraint no entra al frente ni al HV: los algoritmos lo manejan por
    dominancia de factibilidad (factible gana a infactible; entre infactibles,
    menor violación), nativa en NSGA-II/III, AGE-MOEA y CMOPSO.  El único que
    necesita subclase es MOEA/D (ver algoritmos_mo.MOEADConstr)."""

    def __init__(self, model, stoi, itos, latent_dim):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.eval_log = []
        super().__init__(n_var=latent_dim, n_obj=2, n_ieq_constr=1, xl=-5.0, xu=5.0)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = decode_z_batch(self.model, x, self.stoi, self.itos)
        F = np.empty((len(smiles), self.n_obj), dtype=float)
        G = np.empty((len(smiles), 1), dtype=float)
        for i, smi in enumerate(smiles):
            props = calc_properties(smi)
            if props is None:
                F[i] = INVALID_F
                G[i] = INVALID_G
                self.eval_log.append({
                    'smiles': None, 'qed': None, 'sa': None,
                    'fsp3': None, 'valid': False, 'feasible': False,
                })
            else:
                F[i] = (-props['qed'], props['sa'])
                G[i] = FSP3_MIN - props['fsp3']
                self.eval_log.append({
                    'smiles': props['smiles'], 'qed': props['qed'],
                    'sa': props['sa'], 'fsp3': props['fsp3'],
                    'valid': True, 'feasible': bool(props['fsp3'] >= FSP3_MIN),
                })
        out["F"] = F
        out["G"] = G


class NormalizedMolecularLatentProblem(MolecularLatentProblem):
    """MolecularLatentProblem con normalización estática de objetivos.

    Normaliza F a [0,1]^2 usando bounds teóricos fijos:
      -QED ∈ [-1, 0],  SA ∈ [1, 10]

    G queda SIN normalizar a propósito: con dominancia de factibilidad los
    objetivos y la violación nunca se mezclan en la misma escala, y dejar G
    crudo mantiene la violación interpretable como "cuánto Fsp3 falta".

    eval_log mantiene valores crudos → post-procesamiento comparable.
    Diseñado para MOEA/D y CMOPSO donde la escala cruda de SA
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
    """Set de SMILES de train de MOSES (para novelty). Tras construir el set libera
    la Series cacheada (cache_clear)."""
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

        # HV solo sobre los individuos FACTIBLES de la población: el HV final se
        # calcula sobre el frente factible, y mezclar infactibles acá haría que la
        # curva de convergencia mida algo distinto de su propio punto de llegada.
        feas = algorithm.pop.get("FEAS")
        pop_feasible = algorithm.pop[feas.flatten()] if feas is not None else algorithm.pop
        if len(pop_feasible) == 0:
            hv = 0.0
        else:
            F = pop_feasible.get("F")
            # HV sobre objetivos normalizados a [0,1]^2. El problema Normalized ya
            # entrega F normalizado; el crudo ([-QED, SA]) se normaliza aquí.
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
    """Índices del frente no-dominado (minimización), exacto.

    Sustituye a NonDominatedSorting de pymoo, que construye una matriz de
    dominación O(n²) en RAM: con las ~100k moléculas únicas que acumula una run
    exploradora (p. ej. PCX) eso son ~14 GB → OOM. Este filtro estilo Kung usa
    memoria O(tamaño del frente): ordena lexicográficamente (todo dominador de un
    punto aparece antes) y mantiene solo los puntos no-dominados vistos."""
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


def non_dominated(results):
    """Filtra soluciones no-dominadas (memoria O(frente); ver _non_dominated_front)."""
    if not results:
        return []
    F = np.array([[-r['qed'], r['sa']] for r in results], dtype=float)
    return [results[i] for i in _non_dominated_front(F)]


def compute_hv(pareto):
    """Hypervolume del frente de Pareto sobre objetivos normalizados a [0,1]^2."""
    if not pareto:
        return 0.0
    F = np.array([[-r['qed'], r['sa']] for r in pareto])
    F = (F - F_MIN) / F_RANGE
    try:
        return float(HV(ref_point=HV_REF)(F))
    except Exception:
        return 0.0


def compute_spacing(pareto):
    """Spacing de Schott normalizado (CV de distancias al vecino más cercano)."""
    if len(pareto) < 2:
        return 0.0
    F = np.array([[-r['qed'], r['sa']] for r in pareto])
    # Normalizar ejes para compensar escalas distintas (SA~[1,10] vs QED~[0,1])
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
    """Construye frente de Pareto desde el log completo.
    Retorna (pareto, validity, feasibility).

    Solo compiten las moléculas FACTIBLES (Fsp3 ≥ FSP3_MIN): el frente reportado
    debe cumplir el constraint, si no molecules.csv publicaría soluciones que
    violan el umbral. 'feasibility' es la fracción de válidas que lo cumplen."""
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
                'sa': e['sa'], 'fsp3': e['fsp3'],
            }
    validity = round(n_valid / len(eval_log), 4) if eval_log else 0.0
    feasibility = round(n_feasible / n_valid, 4) if n_valid else 0.0
    pareto = non_dominated(list(seen.values()))

    return pareto, validity, feasibility


# ─── I/O ─────────────────────────────────────────────────────────────────────

def save_metrics(path, row):
    """Escribe la fila de métricas de UNA run en su propio CSV (overwrite).
    generate_summary los consolida en alg_dir/metrics.csv al final."""
    pd.DataFrame([row]).to_csv(path, index=False)


def save_molecules(pareto, run_dir):
    """Guarda el frente de Pareto en molecules.csv (escritura atómica: tmp + os.replace).
    Escribe aunque el frente esté vacío: el orquestador (run_experiments.py) usa la
    existencia de este archivo como señal de run completa."""
    cols = ['smiles', 'qed', 'sa', 'fsp3']
    out = os.path.join(run_dir, "molecules.csv")
    tmp = out + ".tmp"
    if not pareto:
        pd.DataFrame(columns=cols).to_csv(tmp, index=False)
    else:
        df = pd.DataFrame(pareto).sort_values('sa', ascending=True)
        df[[c for c in cols if c in df.columns]].to_csv(tmp, index=False)
    os.replace(tmp, out)


def save_tracking(tracker, run_dir):
    """Guarda convergencia por generación y log completo de evaluaciones.

    El log va con qed/sa/fsp3 a 4 decimales: es la misma precisión que guarda
    metrics.csv y la que muestran las figuras, y achica el .gz a la mitad.
    float_format toca solo las columnas float; smiles, valid/feasible y gen
    quedan intactos, y las inválidas siguen saliendo vacías."""
    pd.DataFrame(tracker.history).to_csv(
        os.path.join(run_dir, "convergence.csv"), index=False)
    pd.DataFrame(tracker.problem.eval_log).to_csv(
        os.path.join(run_dir, "all_molecules.csv.gz"),
        index=False, compression='gzip', float_format='%.4f')


def consolidate_all(results_dir=None):
    """Junta TODOS los run_*/metrics.csv del grid en <results>/all_metrics.csv.

    Cada fila = una run, con sus hiperparámetros (operadores, probabilidades, pob,
    gen / w, c1, c2) como columnas. Es el único artefacto que necesita el análisis
    de sensibilidad: un solo CSV largo, listo para agrupar por perilla."""
    base = results_dir if results_dir is not None else RESULTS_DIR
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
    """Calcula métricas, guarda resultados y CSVs de una run.
    hp: dict con los hiperparámetros barridos (se agregan como columnas a metrics.csv)."""
    pareto, validity, feasibility = build_pareto(problem.eval_log)
    hv      = compute_hv(pareto)
    spacing = compute_spacing(pareto)

    # Novelty: fracción de moléculas válidas no presentes en el training set
    valid_smiles = [e['smiles'] for e in problem.eval_log if e['valid']]
    n_novel = sum(1 for s in valid_smiles if s not in tracker.train_smiles)
    novelty = round(n_novel / len(valid_smiles), 4) if valid_smiles else 0.0

    metrics = {
        'algorithm': alg_name, 'pop_size': pop_size, 'n_gen': n_gen,
        **(hp or {}),                       # hiperparámetros barridos como columnas
        'run': run_id + 1, 'n_pareto': len(pareto),
        'hypervolume': round(hv, 6), 'spacing': spacing,
        'validity': validity, 'feasibility': feasibility, 'novelty': novelty,
        'best_qed': round(max((r['qed'] for r in pareto), default=float('nan')), 4),
        'best_sa': round(min((r['sa'] for r in pareto), default=float('nan')), 2),
        # Fsp3 ya no es objetivo: el frente es factible por construcción, así que
        # 'mean_fsp3' (dónde se paró respecto del umbral) informa más que el máximo.
        # El análisis la recalcula desde molecules.csv; acá va para el log de la run.
        'mean_fsp3': round(float(np.mean([r['fsp3'] for r in pareto])), 4) if pareto else float('nan'),
        'time_sec': round(elapsed, 1),
    }

    # molecules.csv se escribe AL FINAL: run_experiments.py lo usa como señal de run completa.
    save_metrics(os.path.join(run_dir, "metrics.csv"), metrics)
    save_tracking(tracker, run_dir)
    save_molecules(pareto, run_dir)

    return metrics, pareto, hv, spacing, validity

