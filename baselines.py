"""
Baselines "no tan modernos" — cribado de MOSES, muestreo aleatorio, escalador
y GA de suma ponderada.
Sirven de piso de comparación frente a los MOEAs (NSGA2/NSGA3/MOEAD/AGEMOEA/
CMOPSO): ninguno hace búsqueda multi-objetivo real de Pareto.
Objetivos: QED (↑), SA (↓).  Constraint: Fsp3 ≥ FSP3_MIN  (igual que el resto
del proyecto).

OJO: el cuerpo de los métodos quedó en la versión de 3 objetivos y NO corre con
el utils_mo actual (2 objetivos).  Falla al normalizar F y al asignar INVALID_F,
y sus frentes tampoco filtran por el constraint.  Hay que migrarlo antes de la
etapa 4.

Este archivo tiene TODO lo de baselines: los cuatro métodos y el orquestador que
los corre en paralelo.  El orquestador se relanza a sí mismo como subproceso (una
run por proceso), igual que hace run_experiments.py con los MOEAs: así cada run
libera el VAE y el set de MOSES al terminar, los límites de hilos se fijan por
run, y una que explote no se lleva puesto al resto.

Guardan en results_baselines/ (NO en results/), para no mezclarse con el grid
de sensibilidad de hiperparámetros de los MOEAs.

Mismo presupuesto que los MOEAs (100.000 evaluaciones = 400 × 250) y las mismas
20 semillas, de modo que la comparación posterior sea pareada por semilla.

Reanudable: una corrida cuenta como completa si existe su molecules.csv.

Uso:
    python baselines.py todas
    python baselines.py todas --methods random screening
    python baselines.py todas --n-runs 5 --parallel 2
    python baselines.py una --method random --pop_size 400 --n_gen 250 --run_id 0
    python baselines.py una --method weighted_ga --weights 0.5,0.3,0.2 --pop_size 400 --run_id 0
    python baselines.py resumen
"""

import os, sys, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    MolecularLatentProblem, LatentSampling, get_operators,
    decode_z_batch, calc_properties, postprocess_run, consolidate_all,
    F_MIN, F_RANGE, HV_REF, INVALID_F,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS_DIR = os.path.join(ROOT_DIR, "results_baselines")
PYTHON = sys.executable      # el python del entorno actual, sin rutas hardcodeadas
Z_LOW, Z_HIGH = -5.0, 5.0    # mismos bounds que MolecularLatentProblem (utils_mo)

POP_SIZE, N_GEN = 400, 250   # 100.000 evaluaciones, igual que los MOEAs
# Los mismos cuatro que analiza analisis.py (BASELINE_KEYS).
METHODS = ['screening', 'random', 'hill_climber', 'weighted_ga']
DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


# ─── Rutas de las corridas ───────────────────────────────────────────────────

def _fmt(x):
    return f"{x:g}"


def tag_de_pesos(method, weights):
    """weighted_ga guarda bajo un subdirectorio con los pesos; el resto no lleva tag.
    Lo usan el worker (al escribir) y el orquestador (al chequear si ya está hecha),
    así que vive en un solo lugar y no pueden divergir."""
    if method != 'weighted_ga':
        return None
    return f"w{_fmt(weights[0])}_{_fmt(weights[1])}_{_fmt(weights[2])}"


def baseline_run_dir(method, pop_size, n_gen, run_id, tag=None):
    """results_baselines/<METHOD>/[tag/]pop{P}_gen{G}/run_k — separado de results/."""
    parts = [BASELINE_RESULTS_DIR, method.upper()]
    if tag:
        parts.append(tag)
    parts += [f"pop{pop_size}_gen{n_gen}", f"run_{run_id + 1:02d}"]
    return os.path.join(*parts)


def is_done(method, run_id):
    """Una corrida está completa si existe su molecules.csv (se escribe al final)."""
    d = baseline_run_dir(method, POP_SIZE, N_GEN, run_id,
                         tag=tag_de_pesos(method, DEFAULT_WEIGHTS))
    return os.path.exists(os.path.join(d, "molecules.csv"))


# ─── Tracker genérico (no depende de una Population de pymoo) ────────────────

class BaselineTracker:
    """Convergencia por lote de evaluaciones ('gen'), calculada sobre los objetivos
    crudos del propio lote. A diferencia de GenerationTracker (utils_mo), no asume
    que exista algorithm.pop: el muestreo aleatorio no es poblacional, y weighted_ga solo
    trae un objetivo escalar — así que el HV se recalcula aquí desde eval_log."""

    def __init__(self, problem, train_smiles):
        self.problem = problem
        self.train_smiles = train_smiles
        self.history = []
        self._last_idx = 0

    def update(self, gen):
        new = self.problem.eval_log[self._last_idx:]
        self._last_idx = len(self.problem.eval_log)
        for e in new:
            e['gen'] = gen

        valid = [e for e in new if e['valid']]
        if valid:
            F = np.array([[-e['qed'], e['sa'], -e['fsp3']] for e in valid])
            Fn = (F - F_MIN) / F_RANGE
            try:
                hv = float(HV(ref_point=HV_REF)(Fn))
            except Exception:
                hv = 0.0
        else:
            hv = 0.0

        smis = [e['smiles'] for e in valid]
        n_valid = len(smis)
        n_novel = sum(1 for s in smis if s not in self.train_smiles)
        self.history.append({
            'gen': gen, 'hv': round(hv, 6), 'n_eval': len(new), 'n_valid': n_valid,
            'validity': round(n_valid / len(new), 4) if new else 0.0,
            'uniqueness': round(len(set(smis)) / n_valid, 4) if n_valid else 0.0,
            'novelty': round(n_novel / n_valid, 4) if n_valid else 0.0,
        })


class TrackerCallback(Callback):
    """Adaptador para que pymoo (weighted_ga) dispare BaselineTracker.update()."""

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def notify(self, algorithm):
        self.tracker.update(algorithm.n_gen)


# ─── Problema para el GA de suma ponderada (single-objective) ────────────────

class WeightedSumLatentProblem(Problem):
    """Mismo espacio/objetivos que MolecularLatentProblem, pero out['F'] es UN
    escalar (suma ponderada de [-QED, SA, -Fsp3] normalizados a [0,1]) para que
    el GA single-objective de pymoo lo use como fitness. eval_log guarda los 3
    valores crudos igual que en los experimentos multi-objetivo, así se reusa
    postprocess_run (Pareto/HV/spacing) sin cambios."""

    def __init__(self, model, stoi, itos, latent_dim, weights):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.eval_log = []
        super().__init__(n_var=latent_dim, n_obj=1, xl=Z_LOW, xu=Z_HIGH)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = decode_z_batch(self.model, x, self.stoi, self.itos)
        F3 = np.empty((len(smiles), 3), dtype=float)
        for i, smi in enumerate(smiles):
            props = calc_properties(smi)
            if props is None:
                F3[i] = INVALID_F
                self.eval_log.append({'smiles': None, 'qed': None, 'sa': None,
                                      'fsp3': None, 'valid': False})
            else:
                F3[i] = (-props['qed'], props['sa'], -props['fsp3'])
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'fsp3': props['fsp3'],
                                      'valid': True})
        F3_norm = (F3 - F_MIN) / F_RANGE
        out["F"] = (F3_norm * self.weights).sum(axis=1, keepdims=True)


# ─── Muestreo aleatorio: barrido manual, sin optimizador ─────────────────────

def run_random(problem, tracker, pop_size, n_gen, run_id):
    """Muestreo del prior del VAE, N(0, I): la distribución sobre la que se
    entrenó el decodificador.  Muestrear uniforme en [-5,5]^256 caería a norma
    ~46 en vez de ~16 y hundiría la validez al 59%, con lo que la diferencia
    frente a los MOEAs mediría el fallo del decoder fuera de la variedad de
    datos y no la ausencia de búsqueda."""
    rng = np.random.default_rng(run_id)
    for gen in range(1, n_gen + 1):
        X = rng.normal(0.0, 1.0, size=(pop_size, problem.n_var))
        problem._evaluate(X, {})
        tracker.update(gen)


class ScreeningProblem:
    """Cribado virtual: no hay espacio latente ni decodificación, solo se evalúan
    moléculas tomadas de MOSES.  Expone eval_log para reusar postprocess_run."""

    def __init__(self):
        self.eval_log = []
        self.n_var = 0

    def evaluate_smiles(self, smiles_list):
        for smi in smiles_list:
            props = calc_properties(smi)
            if props is None:
                self.eval_log.append({'smiles': None, 'qed': None, 'sa': None,
                                      'fsp3': None, 'valid': False})
            else:
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'fsp3': props['fsp3'],
                                      'valid': True})


def run_screening(problem, tracker, pop_size, n_gen, run_id, train_smiles_series):
    """Toma pop_size*n_gen moléculas de MOSES al azar y las evalúa.  Sin generar
    nada: mide qué se consigue cribando una biblioteca existente."""
    n_total = pop_size * n_gen
    pool = train_smiles_series.sample(n_total, replace=False,
                                      random_state=run_id).tolist()
    for gen in range(1, n_gen + 1):
        problem.evaluate_smiles(pool[(gen - 1) * pop_size: gen * pop_size])
        tracker.update(gen)


def run_hill_climber(problem, mus, tracker, pop_size, n_gen, run_id, sigma=0.5):
    """Escalador (1+λ): un único candidato que en cada paso genera pop_size
    mutaciones gaussianas y se mueve solo si la mejor supera a la actual.  Sin
    población ni cruce: aísla cuánto aporta la búsqueda poblacional."""
    rng = np.random.default_rng(run_id)
    x = mus[0].copy()
    best = np.inf
    for gen in range(1, n_gen + 1):
        X = np.clip(x + rng.normal(0, sigma, size=(pop_size, problem.n_var)),
                    Z_LOW, Z_HIGH)
        out = {}
        problem._evaluate(X, out)
        f = np.asarray(out["F"]).ravel()
        i = int(np.argmin(f))
        if f[i] < best:                 # solo acepta mejoras
            best, x = f[i], X[i].copy()
        tracker.update(gen)


def run_weighted_ga(problem, mus, pop_size, n_gen, run_id, tracker):
    cx_prob, mut_prob = 0.9, 1.0 / problem.n_var
    crossover, mutation = get_operators('sbx', 'pm', cx_prob, mut_prob)
    algorithm = GA(
        pop_size=pop_size,
        sampling=LatentSampling(mus),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    minimize(problem, algorithm, ('n_gen', n_gen), seed=run_id, verbose=False,
             callback=TrackerCallback(tracker))
    return cx_prob, mut_prob


# ─── Correr UNA baseline ─────────────────────────────────────────────────────

def correr_una(args):
    """Ejecuta una sola corrida. Es lo que 'todas' lanza en cada subproceso."""
    if args.device != 'auto':
        set_device(args.device)

    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    train_smiles = load_train_smiles()
    if args.method == 'screening':
        model = stoi = itos = None          # el cribado no decodifica nada
        latent_dim = 0
    else:
        model, stoi, itos, latent_dim = load_model()

    weights = (tuple(float(w) for w in args.weights.split(','))
               if args.weights else DEFAULT_WEIGHTS)
    run_dir = baseline_run_dir(args.method, args.pop_size, args.n_gen, args.run_id,
                               tag=tag_de_pesos(args.method, weights))
    os.makedirs(run_dir, exist_ok=True)
    label = f"{args.method.upper()}/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id + 1:02d}"
    print(f"[{label}] Iniciando...", flush=True)

    hp = {}
    t0 = time.time()
    if args.method == 'screening':
        from utils_mo import _load_moses_train_smiles
        problem = ScreeningProblem()
        tracker = BaselineTracker(problem, train_smiles)
        run_screening(problem, tracker, args.pop_size, args.n_gen, args.run_id,
                      _load_moses_train_smiles())
    elif args.method == 'hill_climber':
        mus = load_seed_mus(model, stoi, 1, args.run_id)
        problem = WeightedSumLatentProblem(model, stoi, itos, latent_dim, weights)
        tracker = BaselineTracker(problem, train_smiles)
        run_hill_climber(problem, mus, tracker, args.pop_size, args.n_gen,
                         args.run_id, sigma=args.sigma)
        hp = {'sigma': args.sigma, 'w_qed': round(weights[0], 4),
              'w_sa': round(weights[1], 4), 'w_fsp3': round(weights[2], 4)}
    elif args.method == 'random':
        problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
        tracker = BaselineTracker(problem, train_smiles)
        run_random(problem, tracker, args.pop_size, args.n_gen, args.run_id)
    else:  # weighted_ga
        mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
        problem = WeightedSumLatentProblem(model, stoi, itos, latent_dim, weights)
        tracker = BaselineTracker(problem, train_smiles)
        cx_prob, mut_prob = run_weighted_ga(problem, mus, args.pop_size, args.n_gen,
                                            args.run_id, tracker)
        hp = {'crossover': 'sbx', 'mutation': 'pm', 'cx_prob': cx_prob,
              'mut_prob': round(mut_prob, 6),
              'w_qed': round(weights[0], 4), 'w_sa': round(weights[1], 4),
              'w_fsp3': round(weights[2], 4)}
    elapsed = time.time() - t0

    metrics, pareto, hv, spacing, validity = postprocess_run(
        args.method.upper(), args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Fsp3={metrics['mean_fsp3']}  t={metrics['time_sec']}s", flush=True)


# ─── Orquestador: correr TODAS en paralelo ───────────────────────────────────

def _lanzar(t, device, threads):
    """Corre una tarea en un subproceso de este mismo archivo (modo 'una').
    Devuelve (tarea, ok, segundos, stdout)."""
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""      # sin CUDA: evita inicializar la GPU

    cmd = [PYTHON, os.path.abspath(__file__), "una",
           "--method", t['method'], "--pop_size", str(POP_SIZE),
           "--n_gen", str(N_GEN), "--run_id", str(t['run']), "--device", device]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return t, is_done(t['method'], t['run']), time.time() - t0, proc.stdout


def _human(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def correr_todas(args):
    """Arma la lista de corridas, saltea las hechas y lanza el resto en paralelo."""
    sys.stdout.reconfigure(line_buffering=True)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    threads = max(1, (os.cpu_count() or 1) // args.parallel)

    tasks = [{'method': m, 'run': r}
             for m in args.methods for r in range(args.n_runs)]
    pending = [t for t in tasks if not is_done(t['method'], t['run'])]
    done0 = len(tasks) - len(pending)

    print("=" * 58)
    print("  BASELINES — cribado / aleatorio / escalador / GA ponderado")
    print(f"  Presupuesto  : {POP_SIZE} × {N_GEN} = {POP_SIZE * N_GEN:,} evaluaciones")
    print(f"  Métodos      : {', '.join(args.methods)}")
    print(f"  Dispositivo  : {device}   ({args.parallel} en paralelo)")
    print(f"  Corridas     : {len(tasks)}  (hechas: {done0}, pendientes: {len(pending)})")
    print("=" * 58)

    if not pending:
        print("Todas completas.")
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    t_start, completed, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(_lanzar, t, device, threads): t for t in pending}
        for fut in as_completed(futs):
            t, ok, dt, out = fut.result()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed else 0
            eta = _human((len(pending) - completed) / rate) if rate else "…"
            print(f"[{time.strftime('%T')}] {done0 + completed}/{len(tasks)} | "
                  f"{'ok' if ok else 'FALLÓ':5s} {t['method']:11s} run {t['run'] + 1:02d} "
                  f"({dt:.0f}s) | ETA {eta}", flush=True)
            if not ok:
                failed.append(t)
                print("  └─ " + "\n     ".join(out.strip().splitlines()[-6:]),
                      flush=True)

    print("=" * 58)
    print(f"  FIN: {len(tasks) - len(failed)}/{len(tasks)} en {_human(time.time() - t_start)}")
    if failed:
        for t in failed:
            print(f"    falló: {t['method']} run {t['run'] + 1}")
    print("=" * 58)

    consolidate_all(results_dir=BASELINE_RESULTS_DIR)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Baselines simples — piso de comparación frente a los MOEAs.")
    sub = ap.add_subparsers(dest='modo', required=True, metavar='todas|una|resumen')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p1 = sub.add_parser('todas', formatter_class=fmt,
                        help="Corre todas las baselines en paralelo. Reanudable.")
    p1.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p1.add_argument('--n-runs', type=int, default=20, help="Semillas por método.")
    p1.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help="Dispositivo para el VAE (auto → GPU si hay CUDA).")
    p1.add_argument('-p', '--parallel', type=int, default=3,
                    help="Corridas concurrentes.")
    p1.set_defaults(func=correr_todas)

    p2 = sub.add_parser('una', formatter_class=fmt,
                        help="Corre UNA corrida. Es lo que 'todas' lanza por subproceso.")
    p2.add_argument('--method', required=True, choices=METHODS)
    p2.add_argument('--pop_size', type=int, required=True)
    p2.add_argument('--n_gen', type=int, default=N_GEN)
    p2.add_argument('--run_id', type=int, required=True)
    p2.add_argument('--weights', type=str, default=None,
                    help="Solo weighted_ga: 'w_qed,w_sa,w_fsp3' (default: iguales).")
    p2.add_argument('--sigma', type=float, default=0.5,
                    help="Solo hill_climber: desvío del paso de mutación.")
    p2.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help="Dispositivo para el VAE (auto → GPU si hay CUDA).")
    p2.set_defaults(func=correr_una)

    p3 = sub.add_parser('resumen', formatter_class=fmt,
                        help="Consolida los metrics.csv en all_metrics.csv.")
    p3.set_defaults(func=lambda a: consolidate_all(results_dir=BASELINE_RESULTS_DIR))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
