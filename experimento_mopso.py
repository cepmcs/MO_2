"""
Experimento MOPSO — Optimización multi-objetivo del espacio latente VAE.
Objetivos: SA (↓), d(ALOGP) (↑), d(HBD) (↑)

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
    load_model, load_seed_mus, load_train_smiles, set_device,
    NormalizedMolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, consolidate_all, mopso_run_dir,
)

ALG_NAME = "MOPSO"


def main():
    parser = argparse.ArgumentParser(description="MOPSO — Multi-objective optimization")
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--n_gen',    type=int, default=500)
    parser.add_argument('--run_id',   type=int, default=None)
    parser.add_argument('--w',  type=float, default=0.6, help="Peso de inercia.")
    parser.add_argument('--c1', type=float, default=2.0, help="Coeficiente cognitivo (mejor personal).")
    parser.add_argument('--c2', type=float, default=2.0, help="Coeficiente social (mejor global).")
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
    mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
    train_smiles = load_train_smiles()

    run_dir = mopso_run_dir(args.pop_size, args.n_gen, args.w, args.c1, args.c2, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{ALG_NAME}[w{args.w:g}_c1{args.c1:g}_c2{args.c2:g}]"
             f"/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id+1:02d}")
    print(f"[{label}] Iniciando...", flush=True)

    # Configurar y ejecutar
    problem  = NormalizedMolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker  = GenerationTracker(problem, train_smiles)
    algorithm = MOPSO_CD(
        pop_size=args.pop_size,
        w=args.w, c1=args.c1, c2=args.c2,
        archive_size=args.pop_size,   # archivo de líderes a la misma escala que el enjambre (convención MOPSO)
        sampling=LatentSampling(mus),
    )

    t0  = time.time()
    _   = minimize(problem, algorithm, ('n_gen', args.n_gen),
                   seed=args.run_id, verbose=False, callback=tracker)
    elapsed = time.time() - t0

    # Post-procesamiento (los hiperparámetros barridos van como columnas de metrics.csv)
    hp = {'w': args.w, 'c1': args.c1, 'c2': args.c2}
    metrics, pareto, hv, spacing, validity = postprocess_run(
        ALG_NAME, args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  Div={metrics['diversity']:.4f}  Valid={validity:.0%}  "
          f"n={len(pareto)}  SA={metrics['best_sa']}  ALOGPd={metrics['best_alogp_d']}  "
          f"HBDd={metrics['best_hbd_d']}  t={metrics['time_sec']}s", flush=True)


if __name__ == "__main__":
    main()
