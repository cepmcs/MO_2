"""
Experimento NSGA-III — Optimización multi-objetivo del espacio latente VAE.
Objetivos: QED (↑), SA (↓), Lipinski (↑)

NSGA-III usa vectores de referencia (Das-Dennis), por lo que el pop_size
real se ajusta al número de vectores más cercano al solicitado.

Operadores configurables:
  - Crossover: SBX (default) o PCX
  - Mutation:  PM (default) o Gaussian

Cada combinación guarda en results/<crossover>_<mutation>/
(sbx_pm es el combo base de la comparación entre algoritmos).

Uso:
    python experimento_nsga3.py --pop_size 300 --run_id 0
    python experimento_nsga3.py --pop_size 300 --run_id 0 --crossover pcx --mutation gauss
    python experimento_nsga3.py --pop_size 300 --crossover pcx --mutation gauss --generate_summary
"""

import os, time, argparse
import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles,
    MolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, generate_summary, get_operators, get_results_dir,
)

ALG_NAME = "NSGA3"
N_GEN    = 500


def get_ref_dirs_for_pop(target_pop, n_obj=3):
    """Encuentra n_partitions de Das-Dennis cuyo pop_size sea >= target_pop."""
    for n in range(2, 50):
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n)
        if len(ref_dirs) >= target_pop:
            return ref_dirs
    return get_reference_directions("das-dennis", n_obj, n_partitions=12)


def main():
    parser = argparse.ArgumentParser(description="NSGA-III — Multi-objective optimization")
    parser.add_argument('--pop_size',  type=int, required=True)
    parser.add_argument('--run_id',    type=int, default=None)
    parser.add_argument('--crossover', choices=['sbx', 'pcx'], default='sbx')
    parser.add_argument('--mutation',  choices=['pm', 'gauss'], default='pm')
    parser.add_argument('--generate_summary', action='store_true')
    args = parser.parse_args()

    results_dir = get_results_dir(args.crossover, args.mutation)

    if args.generate_summary:
        generate_summary(ALG_NAME, args.pop_size, results_dir=results_dir)
        return

    assert args.run_id is not None, "Se requiere --run_id"
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    # Directorios (usa pop_size solicitado, no el ajustado)
    alg_dir = os.path.join(results_dir, ALG_NAME, f"pop{args.pop_size}")
    run_dir = os.path.join(alg_dir, f"run_{args.run_id+1:02d}")
    os.makedirs(run_dir, exist_ok=True)

    label = f"{ALG_NAME}[{args.crossover}+{args.mutation}]/pop{args.pop_size}/run_{args.run_id+1:02d}"
    print(f"[{label}] Iniciando...", flush=True)

    # Vectores de referencia Das-Dennis
    ref_dirs = get_ref_dirs_for_pop(args.pop_size)
    actual_pop = len(ref_dirs)
    print(f"[{label}] ref_dirs: solicitado={args.pop_size}, real={actual_pop}", flush=True)

    # Cargar modelo y datos
    model, stoi, itos, latent_dim = load_model()
    mus = load_seed_mus(model, stoi, actual_pop, args.run_id)
    train_smiles = load_train_smiles()

    # Configurar y ejecutar
    crossover, mutation = get_operators(args.crossover, args.mutation)
    problem  = MolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker  = GenerationTracker(problem, train_smiles)
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=actual_pop,
        sampling=LatentSampling(mus),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    t0  = time.time()
    _   = minimize(problem, algorithm, ('n_gen', N_GEN),
                   seed=args.run_id, verbose=True, callback=tracker)
    elapsed = time.time() - t0

    # Post-procesamiento
    metrics, pareto, hv, spacing, validity = postprocess_run(
        ALG_NAME, args.pop_size, N_GEN, args.run_id,
        problem, tracker, elapsed, run_dir, results_dir=results_dir)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Lip={metrics['best_lipinski']}  t={metrics['time_sec']}s", flush=True)


if __name__ == "__main__":
    main()
