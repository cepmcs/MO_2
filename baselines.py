"""
Baselines simples: cribado de MOSES, muestreo aleatorio, escalador y GA de suma
ponderada.  Ninguno de los cuatro hace búsqueda multiobjetivo de Pareto, así que
sirven de piso de comparación frente a los MOEAs (NSGA2/NSGA3/MOEAD/AGEMOEA/
MOPSO).  Objetivos: QED (↑), SA (↓), Lipinski (↑).

Guardan en results_baselines/, no en results/, para no mezclarse con el grid de
sensibilidad de hiperparámetros.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    MolecularLatentProblem, LatentSampling, get_operators,
    decode_z_batch, calc_properties, postprocess_run, consolidate_all,
    F_MIN, F_RANGE, HV_REF, INVALID_F,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_RESULTS_DIR = os.path.join(ROOT_DIR, "results_baselines")
Z_LOW, Z_HIGH = -5.0, 5.0   # mismos bounds que MolecularLatentProblem

METHODS = ['screening', 'random', 'hill_climber', 'weighted_ga']
# Mismo presupuesto que los MOEAs (100.000 evaluaciones) y las mismas semillas,
# para que la comparación de la etapa 4 sea pareada.
POP_SIZE, N_GEN = 400, 250
PESOS_IGUALES = (1 / 3, 1 / 3, 1 / 3)


def _fmt(x):
    return f"{x:g}"


def _tag(method, weights):
    """weighted_ga guarda bajo un subdirectorio con sus pesos."""
    return (f"w{_fmt(weights[0])}_{_fmt(weights[1])}_{_fmt(weights[2])}"
            if method == 'weighted_ga' else None)


def baseline_run_dir(method, pop_size, n_gen, run_id, tag=None):
    """results_baselines/<METODO>/[tag/]pop{P}_gen{G}/run_k."""
    parts = [BASELINE_RESULTS_DIR, method.upper()]
    if tag:
        parts.append(tag)
    parts += [f"pop{pop_size}_gen{n_gen}", f"run_{run_id + 1:02d}"]
    return os.path.join(*parts)


# ─── Tracker (no depende de una Population de pymoo) ─────────────────────────

class BaselineTracker:
    """Convergencia por lote de evaluaciones.  A diferencia de GenerationTracker
    no asume que exista algorithm.pop: el muestreo aleatorio no es poblacional y
    weighted_ga solo trae un objetivo escalar, así que el HV se recalcula acá
    desde eval_log."""

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
        hv = 0.0
        if valid:
            F = np.array([[-e['qed'], e['sa'], -e['lipinski']] for e in valid])
            try:
                hv = float(HV(ref_point=HV_REF)((F - F_MIN) / F_RANGE))
            except Exception:
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


# ─── Problema para el GA de suma ponderada ───────────────────────────────────

class WeightedSumLatentProblem(Problem):
    """Mismo espacio que MolecularLatentProblem, pero out['F'] es un escalar (la
    suma ponderada de los objetivos normalizados) para el GA single-objective.
    eval_log guarda los 3 valores crudos, así que postprocess_run calcula el
    Pareto y el HV sin cambios."""

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
                                      'lipinski': None, 'valid': False})
            else:
                F3[i] = (-props['qed'], props['sa'], -props['lipinski'])
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'lipinski': props['lipinski'],
                                      'valid': True})
        out["F"] = (((F3 - F_MIN) / F_RANGE) * self.weights).sum(axis=1, keepdims=True)


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
                                      'lipinski': None, 'valid': False})
            else:
                self.eval_log.append({'smiles': props['smiles'], 'qed': props['qed'],
                                      'sa': props['sa'], 'lipinski': props['lipinski'],
                                      'valid': True})


# ─── Los cuatro métodos ──────────────────────────────────────────────────────

def run_random(problem, tracker, pop_size, n_gen, run_id):
    """Muestreo del prior del VAE, N(0, I).  Muestrear uniforme en los bounds del
    espacio latente caería fuera de la variedad de datos y mediría el fallo del
    decoder en vez de la ausencia de búsqueda."""
    rng = np.random.default_rng(run_id)
    for gen in range(1, n_gen + 1):
        problem._evaluate(rng.normal(0.0, 1.0, size=(pop_size, problem.n_var)), {})
        tracker.update(gen)


def run_screening(problem, tracker, pop_size, n_gen, run_id, train_smiles_series):
    """Toma pop_size*n_gen moléculas de MOSES al azar y las evalúa.  Sin generar
    nada: mide qué se consigue cribando una biblioteca existente."""
    pool = train_smiles_series.sample(pop_size * n_gen, replace=False,
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


# ─── Una corrida ─────────────────────────────────────────────────────────────

def ejecutar(args):
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
               if args.weights else PESOS_IGUALES)
    run_dir = baseline_run_dir(args.method, args.pop_size, args.n_gen,
                               args.run_id, tag=_tag(args.method, weights))
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{args.method.upper()}/pop{args.pop_size}xgen{args.n_gen}"
             f"/run_{args.run_id + 1:02d}")
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
              'w_sa': round(weights[1], 4), 'w_lip': round(weights[2], 4)}
    elif args.method == 'random':
        problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
        tracker = BaselineTracker(problem, train_smiles)
        run_random(problem, tracker, args.pop_size, args.n_gen, args.run_id)
    else:   # weighted_ga
        mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
        problem = WeightedSumLatentProblem(model, stoi, itos, latent_dim, weights)
        tracker = BaselineTracker(problem, train_smiles)
        cx_prob, mut_prob = run_weighted_ga(problem, mus, args.pop_size,
                                            args.n_gen, args.run_id, tracker)
        hp = {'crossover': 'sbx', 'mutation': 'pm', 'cx_prob': cx_prob,
              'mut_prob': round(mut_prob, 6),
              'w_qed': round(weights[0], 4), 'w_sa': round(weights[1], 4),
              'w_lip': round(weights[2], 4)}
    elapsed = time.time() - t0

    metrics, pareto, hv, spacing, validity = postprocess_run(
        args.method.upper(), args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Lip={metrics['best_lipinski']}  t={metrics['time_sec']}s", flush=True)


# ─── Barrido completo (todas las semillas de cada método) ────────────────────

def _done(method, run_id):
    """Una corrida está completa si existe su molecules.csv (se escribe al final)."""
    run_dir = baseline_run_dir(method, POP_SIZE, N_GEN, run_id,
                               tag=_tag(method, PESOS_IGUALES))
    return os.path.exists(os.path.join(run_dir, "molecules.csv"))


def _lanzar(method, run_id, device, threads):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [sys.executable, os.path.abspath(__file__), "--method", method,
           "--pop_size", str(POP_SIZE), "--n_gen", str(N_GEN),
           "--run_id", str(run_id), "--device", device]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return method, run_id, _done(method, run_id), time.time() - t0, proc.stdout


def _humano(s):
    h, s = divmod(int(s), 3600)
    m, _ = divmod(s, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def sweep(args):
    """Corre todos los métodos × semillas en subprocesos.  Reanudable."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    threads = max(1, (os.cpu_count() or 1) // args.parallel)

    tareas = [(m, r) for m in args.methods for r in range(args.n_runs)]
    pendientes = [t for t in tareas if not _done(*t)]
    hechas = len(tareas) - len(pendientes)

    print("=" * 58)
    print("  BASELINES — cribado / aleatorio / escalador / GA ponderado")
    print(f"  Presupuesto  : {POP_SIZE} × {N_GEN} = {POP_SIZE * N_GEN:,} evaluaciones")
    print(f"  Métodos      : {', '.join(args.methods)}")
    print(f"  Dispositivo  : {device}   ({args.parallel} en paralelo)")
    print(f"  Corridas     : {len(tareas)}  (hechas: {hechas}, "
          f"pendientes: {len(pendientes)})")
    print("=" * 58)

    if not pendientes:
        print("Todas completas.")
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    t_start, completadas, fallidas = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(_lanzar, m, r, device, threads) for m, r in pendientes]
        for fut in as_completed(futs):
            method, run_id, ok, dt, salida = fut.result()
            completadas += 1
            tasa = completadas / (time.time() - t_start)
            eta = _humano((len(pendientes) - completadas) / tasa) if tasa else "…"
            print(f"[{time.strftime('%T')}] {hechas + completadas}/{len(tareas)} | "
                  f"{'ok' if ok else 'FALLÓ':5s} {method:11s} run {run_id + 1:02d} "
                  f"({dt:.0f}s) | ETA {eta}", flush=True)
            if not ok:
                fallidas.append((method, run_id))
                print("  └─ " + "\n     ".join(salida.strip().splitlines()[-6:]),
                      flush=True)

    print("=" * 58)
    print(f"  FIN: {len(tareas) - len(fallidas)}/{len(tareas)} en "
          f"{_humano(time.time() - t_start)}")
    for method, run_id in fallidas:
        print(f"    falló: {method} run {run_id + 1}")
    print("=" * 58)

    consolidate_all(results_dir=BASELINE_RESULTS_DIR)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baselines simples — piso de comparación frente a los MOEAs.")
    parser.add_argument('--sweep', action='store_true',
                        help="Corre todos los métodos × semillas en paralelo.")
    parser.add_argument('--method', choices=METHODS, default=None,
                        help="Método de una corrida individual.")
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS,
                        help="Solo --sweep: métodos a correr.")
    parser.add_argument('--n-runs', type=int, default=20,
                        help="Solo --sweep: semillas por método.")
    parser.add_argument('-p', '--parallel', type=int, default=3,
                        help="Solo --sweep: corridas concurrentes.")
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--n_gen', type=int, default=N_GEN)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--weights', type=str, default=None,
                        help="Solo weighted_ga: 'w_qed,w_sa,w_lip' (default: iguales).")
    parser.add_argument('--sigma', type=float, default=0.5,
                        help="Solo hill_climber: desvío del paso de mutación.")
    parser.add_argument('--generate_summary', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Dispositivo para el VAE (default: auto → GPU si hay CUDA).")
    args = parser.parse_args()

    if args.generate_summary:
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    if args.sweep:
        sys.stdout.reconfigure(line_buffering=True)
        sweep(args)
        return

    assert args.method is not None, "Se requiere --method (o --sweep)"
    assert args.pop_size is not None, "Se requiere --pop_size"
    assert args.run_id is not None, "Se requiere --run_id"
    if args.device != 'auto':
        set_device(args.device)
    ejecutar(args)


if __name__ == "__main__":
    main()
