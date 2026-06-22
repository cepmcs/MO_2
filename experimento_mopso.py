"""
Experimento MOPSO — Optimización multi-objetivo del espacio latente VAE.
Objetivos: QED (↑), SA (↓), Lipinski (↑)

Uso:
    python experimento_mopso.py --pop_size 300 --run_id 0
    python experimento_mopso.py --pop_size 300 --generate_summary
"""

import os, time, argparse
import numpy as np
import torch
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles,
    NormalizedMolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, generate_summary, get_results_dir, BASELINE_COMBO,
)

ALG_NAME = "MOPSO"
N_GEN    = 500


def main():
    parser = argparse.ArgumentParser(description="MOPSO — Multi-objective optimization")
    parser.add_argument('--pop_size', type=int, required=True)
    parser.add_argument('--run_id',   type=int, default=None)
    parser.add_argument('--generate_summary', action='store_true')
    args = parser.parse_args()

    # MOPSO no tiene crossover/mutation: vive en el combo base sbx_pm.
    results_dir = get_results_dir(*BASELINE_COMBO)

    if args.generate_summary:
        generate_summary(ALG_NAME, args.pop_size, results_dir=results_dir)
        return

    assert args.run_id is not None, "Se requiere --run_id"
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    # Directorios
    alg_dir = os.path.join(results_dir, ALG_NAME, f"pop{args.pop_size}")
    run_dir = os.path.join(alg_dir, f"run_{args.run_id+1:02d}")
    os.makedirs(run_dir, exist_ok=True)

    label = f"{ALG_NAME}/pop{args.pop_size}/run_{args.run_id+1:02d}"
    print(f"[{label}] Iniciando...", flush=True)

    # Cargar modelo y datos
    model, stoi, itos, latent_dim = load_model()
    mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
    train_smiles = load_train_smiles()

    # Configurar y ejecutar
    problem  = NormalizedMolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker  = GenerationTracker(problem, train_smiles)
    algorithm = MOPSO_CD(
        pop_size=args.pop_size,
        sampling=LatentSampling(mus),
    )

    t0  = time.time()
    _   = minimize(problem, algorithm, ('n_gen', N_GEN),
                   seed=args.run_id, verbose=False, callback=tracker)
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
