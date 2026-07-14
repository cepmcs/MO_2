"""
Experimento AGE-MOEA — Optimización multi-objetivo del espacio latente VAE.
Objetivos: QED (↑), SA (↓), Fsp3 (↑)

AGE-MOEA (Adaptive Geometry Estimation MOEA) adapta la presión de
selección según la geometría del frente de Pareto actual.

Operadores configurables:
  - Crossover: SBX (default) o PCX
  - Mutation:  PM (default) o Gaussian

Cada combinación guarda en results/<crossover>_<mutation>/
(sbx_pm es el combo base de la comparación entre algoritmos).

Uso:
    python experimento_agemoea.py --pop_size 300 --run_id 0
    python experimento_agemoea.py --pop_size 300 --run_id 0 --crossover pcx --mutation gauss
    python experimento_agemoea.py --pop_size 300 --crossover pcx --mutation gauss --generate_summary
"""

import os, time, argparse
import numpy as np
import torch
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    MolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, consolidate_all, get_operators, ga_run_dir,
)

ALG_NAME = "AGEMOEA"


def main():
    parser = argparse.ArgumentParser(description="AGE-MOEA — Multi-objective optimization")
    parser.add_argument('--pop_size',  type=int, default=None)
    parser.add_argument('--n_gen',     type=int, default=500)
    parser.add_argument('--run_id',    type=int, default=None)
    parser.add_argument('--crossover', choices=['sbx', 'pcx'], default='sbx')
    parser.add_argument('--mutation',  choices=['pm', 'gauss'], default='pm')
    parser.add_argument('--cx_prob',   type=float, default=0.9,
                        help="Probabilidad de cruce (por apareamiento).")
    parser.add_argument('--mut_prob',  type=float, default=None,
                        help="Probabilidad de mutación POR-GEN (default: 1/n_var).")
    parser.add_argument('--generate_summary', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Dispositivo para el VAE (default: auto → GPU si hay CUDA).")
    args = parser.parse_args()

    # Dispositivo de cómputo: 'auto' respeta el default del módulo (GPU si hay CUDA).
    if args.device != 'auto':
        set_device(args.device)

    if args.generate_summary:
        consolidate_all()
        return

    assert args.pop_size is not None, "Se requiere --pop_size"
    assert args.run_id is not None, "Se requiere --run_id"
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    # Cargar modelo y datos
    model, stoi, itos, latent_dim = load_model()
    mut_prob = args.mut_prob if args.mut_prob is not None else 1.0 / latent_dim
    mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
    train_smiles = load_train_smiles()

    run_dir = ga_run_dir(ALG_NAME, args.crossover, args.mutation, args.cx_prob,
                         mut_prob, args.pop_size, args.n_gen, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{ALG_NAME}[{args.crossover}{args.cx_prob:g}+{args.mutation}{mut_prob:g}]"
             f"/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id+1:02d}")
    print(f"[{label}] Iniciando...", flush=True)

    # Configurar y ejecutar
    crossover, mutation = get_operators(args.crossover, args.mutation, args.cx_prob, mut_prob)
    problem  = MolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker  = GenerationTracker(problem, train_smiles)
    algorithm = AGEMOEA(
        pop_size=args.pop_size,
        sampling=LatentSampling(mus),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    t0  = time.time()
    _   = minimize(problem, algorithm, ('n_gen', args.n_gen),
                   seed=args.run_id, verbose=False, callback=tracker)
    elapsed = time.time() - t0

    # Post-procesamiento (los hiperparámetros barridos van como columnas de metrics.csv)
    hp = {'crossover': args.crossover, 'mutation': args.mutation,
          'cx_prob': args.cx_prob, 'mut_prob': round(mut_prob, 6)}
    metrics, pareto, hv, spacing, validity = postprocess_run(
        ALG_NAME, args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Div={metrics['diversity']:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Fsp3={metrics['best_fsp3']}  t={metrics['time_sec']}s", flush=True)


if __name__ == "__main__":
    main()
