"""
Baselines "no tan modernos" — random sampling, LHS y GA de suma ponderada.
Sirven de piso de comparación frente a los MOEAs (NSGA2/NSGA3/MOEAD/AGEMOEA/
MOPSO): ninguno de los tres hace búsqueda multi-objetivo real de Pareto.
Objetivos: QED (↑), SA (↓), Fsp3 (↑)  (mismos que el resto del proyecto)

Guardan en results_baselines/ (NO en results/), para no mezclarse con el grid
de sensibilidad de hiperparámetros de los MOEAs.

Uso:
    python experimento_baselines.py --method random     --pop_size 300 --n_gen 500 --run_id 0
    python experimento_baselines.py --method lhs         --pop_size 300 --n_gen 500 --run_id 0
    python experimento_baselines.py --method weighted_ga --pop_size 300 --n_gen 500 --run_id 0
    python experimento_baselines.py --method weighted_ga --weights 0.5,0.3,0.2 --pop_size 300 --run_id 0  (pesos: qed,sa,fsp3)
    python experimento_baselines.py --method random --generate_summary
"""

import os, time, argparse
import numpy as np
import torch
from scipy.stats import qmc
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
Z_LOW, Z_HIGH = -5.0, 5.0   # mismos bounds que MolecularLatentProblem (utils_mo)


def _fmt(x):
    return f"{x:g}"


def baseline_run_dir(method, pop_size, n_gen, run_id, tag=None):
    """results_baselines/<METHOD>/[tag/]pop{P}_gen{G}/run_k — separado de results/."""
    parts = [BASELINE_RESULTS_DIR, method.upper()]
    if tag:
        parts.append(tag)
    parts += [f"pop{pop_size}_gen{n_gen}", f"run_{run_id + 1:02d}"]
    return os.path.join(*parts)


# ─── Tracker genérico (no depende de una Population de pymoo) ────────────────

class BaselineTracker:
    """Convergencia por lote de evaluaciones ('gen'), calculada sobre los objetivos
    crudos del propio lote. A diferencia de GenerationTracker (utils_mo), no asume
    que exista algorithm.pop: random/LHS no son poblacionales, y weighted_ga solo
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


# ─── Random sampling / LHS: barrido manual, sin optimizador ──────────────────

def run_random(problem, tracker, pop_size, n_gen, run_id):
    rng = np.random.default_rng(run_id)
    for gen in range(1, n_gen + 1):
        X = rng.uniform(Z_LOW, Z_HIGH, size=(pop_size, problem.n_var))
        problem._evaluate(X, {})
        tracker.update(gen)


def run_lhs(problem, tracker, pop_size, n_gen, run_id):
    """Un único diseño LHS para TODO el presupuesto (pop_size*n_gen), partido en
    lotes de pop_size solo para el decode batcheado y el tracking por 'gen'.
    Generarlo en lotes separados perdería la estratificación global, que es la
    ventaja de LHS sobre random puro."""
    n_total = pop_size * n_gen
    sampler = qmc.LatinHypercube(d=problem.n_var, seed=run_id)
    X_all = qmc.scale(sampler.random(n=n_total),
                      [Z_LOW] * problem.n_var, [Z_HIGH] * problem.n_var)
    for gen in range(1, n_gen + 1):
        X = X_all[(gen - 1) * pop_size: gen * pop_size]
        problem._evaluate(X, {})
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baselines simples (random / LHS / GA suma ponderada) — "
                     "piso de comparación frente a los MOEAs.")
    parser.add_argument('--method', choices=['random', 'lhs', 'weighted_ga'], required=True)
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--n_gen', type=int, default=500)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--weights', type=str, default=None,
                        help="Solo weighted_ga: 'w_qed,w_sa,w_fsp3' (default: iguales).")
    parser.add_argument('--generate_summary', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Dispositivo para el VAE (default: auto → GPU si hay CUDA).")
    args = parser.parse_args()

    if args.device != 'auto':
        set_device(args.device)

    if args.generate_summary:
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    assert args.pop_size is not None, "Se requiere --pop_size"
    assert args.run_id is not None, "Se requiere --run_id"
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    model, stoi, itos, latent_dim = load_model()
    train_smiles = load_train_smiles()

    weights = (tuple(float(w) for w in args.weights.split(','))
              if args.weights else (1 / 3, 1 / 3, 1 / 3))
    tag = (f"w{_fmt(weights[0])}_{_fmt(weights[1])}_{_fmt(weights[2])}"
          if args.method == 'weighted_ga' else None)
    run_dir = baseline_run_dir(args.method, args.pop_size, args.n_gen, args.run_id, tag=tag)
    os.makedirs(run_dir, exist_ok=True)
    label = f"{args.method.upper()}/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id + 1:02d}"
    print(f"[{label}] Iniciando...", flush=True)

    hp = {}
    t0 = time.time()
    if args.method == 'random':
        problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
        tracker = BaselineTracker(problem, train_smiles)
        run_random(problem, tracker, args.pop_size, args.n_gen, args.run_id)
    elif args.method == 'lhs':
        problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
        tracker = BaselineTracker(problem, train_smiles)
        run_lhs(problem, tracker, args.pop_size, args.n_gen, args.run_id)
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

    alg_name = args.method.upper()
    metrics, pareto, hv, spacing, validity = postprocess_run(
        alg_name, args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Div={metrics['diversity']:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Fsp3={metrics['best_fsp3']}  t={metrics['time_sec']}s", flush=True)


if __name__ == "__main__":
    main()
