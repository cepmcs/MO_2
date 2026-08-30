"""
Baselines: cribado de MOSES, muestreo aleatorio, escalador y GA de suma ponderada.
Ninguno hace búsqueda de Pareto: son el piso de comparación de los MOEAs.

Mismo presupuesto (100.000 evaluaciones = 400 × 250) y las mismas 20 semillas, así
la comparación queda pareada.  Guardan en resultados/baselines/, aparte de resultados/grid.
Reanudable: una corrida cuenta como completa si existe su molecules.csv.

    python baselines.py todas
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
    F_MIN, F_RANGE, HV_REF, INVALID_F, INVALID_G, FSP3_MIN,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS_DIR = os.path.join(ROOT_DIR, "resultados", "baselines")
PYTHON = sys.executable      # el python del entorno actual
Z_LOW, Z_HIGH = -5.0, 5.0    # mismos bounds que MolecularLatentProblem

POP_SIZE, N_GEN = 400, 250   # 100.000 evaluaciones, igual que los MOEAs
METHODS = ['screening', 'random', 'hill_climber', 'weighted_ga']
DEFAULT_WEIGHTS = (0.5, 0.5)   # un peso por objetivo ([-QED, SA])


# ─── Rutas de las corridas ───────────────────────────────────────────────────

def _fmt(x):
    return f"{x:g}"


def tag_de_pesos(method, weights):
    """weighted_ga guarda bajo un subdirectorio con los pesos; el resto no lleva tag."""
    if method != 'weighted_ga':
        return None
    return f"w{_fmt(weights[0])}_{_fmt(weights[1])}"


def baseline_run_dir(method, pop_size, n_gen, run_id, tag=None):
    """resultados/baselines/<METHOD>/[tag/]pop{P}_gen{G}/run_k."""
    parts = [BASELINE_RESULTS_DIR, method.upper()]
    if tag:
        parts.append(tag)
    parts += [f"pop{pop_size}_gen{n_gen}", f"run_{run_id + 1:02d}"]
    return os.path.join(*parts)


def is_done(method, run_id):
    """Una corrida está completa si existe su molecules.csv."""
    d = baseline_run_dir(method, POP_SIZE, N_GEN, run_id,
                         tag=tag_de_pesos(method, DEFAULT_WEIGHTS))
    return os.path.exists(os.path.join(d, "molecules.csv"))


# ─── Tracker genérico (no depende de una Population de pymoo) ────────────────

class BaselineTracker:
    """Convergencia por lote de evaluaciones.  A diferencia de GenerationTracker no
    asume que haya algorithm.pop: el HV se recalcula desde eval_log."""

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
        # Solo las factibles, igual que GenerationTracker.
        feasible = [e for e in valid if e['feasible']]
        if feasible:
            F = np.array([[-e['qed'], e['sa']] for e in feasible])
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
        # Mismas columnas y orden que el convergence.csv de los MOEAs.
        self.history.append({
            'gen': gen, 'hv': round(hv, 6), 'n_feasible': len(feasible),
            'feasibility': round(len(feasible) / n_valid, 4) if n_valid else 0.0,
            'n_eval': len(new), 'n_valid': n_valid,
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
    """MolecularLatentProblem con F escalar: la suma ponderada de [-QED, SA]
    normalizados.  El constraint va como G y no en la suma: ponderarlo lo haría
    negociable."""

    def __init__(self, model, stoi, itos, latent_dim, weights):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.eval_log = []
        super().__init__(n_var=latent_dim, n_obj=1, n_ieq_constr=1,
                         xl=Z_LOW, xu=Z_HIGH)

    def _evaluate(self, x, out, *args, **kwargs):
        smiles = decode_z_batch(self.model, x, self.stoi, self.itos)
        F = np.empty((len(smiles), 2), dtype=float)
        G = np.empty((len(smiles), 1), dtype=float)
        for i, smi in enumerate(smiles):
            props = calc_properties(smi)
            if props is None:
                F[i] = INVALID_F
                G[i] = INVALID_G
                self.eval_log.append({'smiles': None, 'qed': None, 'sa': None,
                                      'fsp3': None, 'valid': False,
                                      'feasible': False})
            else:
                F[i] = (-props['qed'], props['sa'])
                G[i] = FSP3_MIN - props['fsp3']
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'fsp3': props['fsp3'],
                                      'valid': True,
                                      'feasible': bool(props['fsp3'] >= FSP3_MIN)})
        F_norm = (F - F_MIN) / F_RANGE
        out["F"] = (F_norm * self.weights).sum(axis=1, keepdims=True)
        out["G"] = G


# ─── Muestreo aleatorio: barrido manual, sin optimizador ─────────────────────

def run_random(problem, tracker, pop_size, n_gen, run_id):
    """Muestreo del prior del VAE, N(0, I).  Uniforme en [-5,5]^256 caería fuera de
    la variedad de datos y mediría el fallo del decoder, no la falta de búsqueda."""
    rng = np.random.default_rng(run_id)
    for gen in range(1, n_gen + 1):
        X = rng.normal(0.0, 1.0, size=(pop_size, problem.n_var))
        problem._evaluate(X, {})
        tracker.update(gen)


class ScreeningProblem:
    """Cribado virtual: evalúa moléculas de MOSES, sin latente ni decodificación.
    Sin F ni G, solo registra.  Expone eval_log para reusar postprocess_run."""

    def __init__(self):
        self.eval_log = []
        self.n_var = 0

    def evaluate_smiles(self, smiles_list):
        for smi in smiles_list:
            props = calc_properties(smi)
            if props is None:
                self.eval_log.append({'smiles': None, 'qed': None, 'sa': None,
                                      'fsp3': None, 'valid': False,
                                      'feasible': False})
            else:
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'fsp3': props['fsp3'],
                                      'valid': True,
                                      'feasible': bool(props['fsp3'] >= FSP3_MIN)})


def run_screening(problem, tracker, pop_size, n_gen, run_id, train_smiles_series):
    """pop_size*n_gen moléculas de MOSES al azar.  Sin prefiltrar por Fsp3:
    prefiltrarlo le regalaría el constraint."""
    n_total = pop_size * n_gen
    pool = train_smiles_series.sample(n_total, replace=False,
                                      random_state=run_id).tolist()
    for gen in range(1, n_gen + 1):
        problem.evaluate_smiles(pool[(gen - 1) * pop_size: gen * pop_size])
        tracker.update(gen)


def run_hill_climber(problem, mus, tracker, pop_size, n_gen, run_id, sigma=0.5):
    """Escalador (1+λ): pop_size mutaciones gaussianas por paso, se mueve solo si
    la mejor supera a la actual.  No pasa por pymoo, así que la regla de Deb va
    escrita acá: ordena por (violación, fitness)."""
    rng = np.random.default_rng(run_id)
    x = mus[0].copy()
    best = None                          # (violación, fitness) del candidato actual
    for gen in range(1, n_gen + 1):
        X = np.clip(x + rng.normal(0, sigma, size=(pop_size, problem.n_var)),
                    Z_LOW, Z_HIGH)
        out = {}
        problem._evaluate(X, out)
        f = np.asarray(out["F"]).ravel()
        cv = np.maximum(np.asarray(out["G"]).ravel(), 0.0)   # 0 si es factible
        i = int(np.lexsort((f, cv))[0])   # lexsort: la última clave manda (cv)
        cand = (cv[i], f[i])
        if best is None or cand < best:
            best, x = cand, X[i].copy()
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

    # fsp3_min en todas: all_metrics.csv junta baselines y MOEAs.
    hp = {'fsp3_min': FSP3_MIN}
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
        hp |= {'sigma': args.sigma, 'w_qed': round(weights[0], 4),
               'w_sa': round(weights[1], 4)}
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
        hp |= {'crossover': 'sbx', 'mutation': 'pm', 'cx_prob': cx_prob,
               'mut_prob': round(mut_prob, 6),
               'w_qed': round(weights[0], 4), 'w_sa': round(weights[1], 4)}
    elapsed = time.time() - t0

    metrics, pareto, hv, spacing, validity = postprocess_run(
        args.method.upper(), args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  "
          f"Valid={validity:.0%}  Feas={metrics['feasibility']:.0%}  n={len(pareto)}  "
          f"QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
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
        env["CUDA_VISIBLE_DEVICES"] = ""      # evita inicializar la GPU

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
                    help="Solo weighted_ga y hill_climber: 'w_qed,w_sa' "
                         "(default: iguales).")
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
