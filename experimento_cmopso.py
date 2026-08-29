"""
Experimento CMOPSO — Optimización multi-objetivo del espacio latente VAE.
Objetivos: QED (↑), SA (↓).  Constraint: Fsp3 ≥ FSP3_MIN.

CMOPSO (Zhang et al., Inf. Sci. 427:63-76, 2018) reemplaza al MOPSO_CD de la etapa
anterior: maneja el constraint de forma nativa (dominancia de factibilidad en la
supervivencia SPEA2 y en el archivo de elites), mientras que MOPSO_CD ordena por
NonDominatedSorting crudo sobre F y lo ignora en silencio.

Sus perillas no son las de MOPSO_CD: la ecuación de velocidad es
v' = R1·v + R2·(ganador − p), con R1/R2 aleatorios por dimensión y sin pbest, así que
w, c1 y c2 no existen.  Se barren elite_size, la mutación por-gen y el tope de
velocidad.

Uso:
    python experimento_cmopso.py --pop_size 100 --n_gen 1000 --run_id 0
    python experimento_cmopso.py --generate_summary
"""

import os, time, argparse
import numpy as np
import torch
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    NormalizedMolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, consolidate_all, cmopso_run_dir, FSP3_MIN,
)

ALG_NAME = "CMOPSO"


def main():
    parser = argparse.ArgumentParser(description="CMOPSO — Multi-objective optimization")
    parser.add_argument('--pop_size', type=int, default=None)
    parser.add_argument('--n_gen',    type=int, default=500)
    parser.add_argument('--run_id',   type=int, default=None)
    parser.add_argument('--elite_size', type=int, default=10,
                        help="Tamaño al que se poda el archivo de elites (perilla de pymoo).")
    parser.add_argument('--mut_prob', type=float, default=0.031,
                        help="Probabilidad de mutación POR GEN (prob_var), como en el grid GA.")
    parser.add_argument('--vel_rate', type=float, default=0.2,
                        help="max_velocity_rate: V_max = vel_rate · (xu − xl).")
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

    run_dir = cmopso_run_dir(args.pop_size, args.n_gen, args.elite_size,
                             args.mut_prob, args.vel_rate, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{ALG_NAME}[e{args.elite_size:g}_mut{args.mut_prob:g}_vel{args.vel_rate:g}]"
             f"/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id+1:02d}")
    print(f"[{label}] Iniciando...", flush=True)

    # Configurar y ejecutar
    problem  = NormalizedMolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker  = GenerationTracker(problem, train_smiles)
    algorithm = CMOPSO(
        pop_size=args.pop_size,
        elite_size=args.elite_size,
        max_velocity_rate=args.vel_rate,
        sampling=LatentSampling(mus),
    )
    # CMOPSO fija PolynomialMutation(prob=mutation_rate) en su __init__, sin parámetro
    # para pasar el prob_var: `prob` es por-INDIVIDUO y deja el por-gen en el default
    # 1/n_var (=1/256≈0.0039), ~8x más suave que el 0.031 que ganó en el grid GA.  Se
    # pisa acá para barrer la misma perilla que los GA y que la mutación sea comparable
    # entre las cinco familias.
    algorithm.mutation = PM(prob=1.0, prob_var=args.mut_prob)

    t0  = time.time()
    _   = minimize(problem, algorithm, ('n_gen', args.n_gen),
                   seed=args.run_id, verbose=False, callback=tracker)
    elapsed = time.time() - t0

    # Post-procesamiento (los hiperparámetros barridos van como columnas de metrics.csv)
    hp = {'elite_size': args.elite_size, 'mut_prob': args.mut_prob,
          'vel_rate': args.vel_rate, 'fsp3_min': FSP3_MIN}
    metrics, pareto, hv, spacing, validity = postprocess_run(
        ALG_NAME, args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  "
          f"Valid={validity:.0%}  Feas={metrics['feasibility']:.0%}  n={len(pareto)}  "
          f"QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Fsp3={metrics['mean_fsp3']}  t={metrics['time_sec']}s", flush=True)


if __name__ == "__main__":
    main()
