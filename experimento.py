"""
Los tres algoritmos de la comparación (NSGA-II, NSGA-III y AGE-MOEA) y el cuerpo
de una corrida.

Para agregar un algoritmo se suma a la tabla ALGORITMOS; si necesita argumentos
propios, como NSGA-III con sus ref_dirs, se agregan en correr.  Las perillas son
las mismas en los tres: --crossover --cx_prob --mut_prob (la mutación es siempre PM).

Corre UNA configuración por vez; el grid lo lanza run_experiments.py.
"""
import argparse
import os
import time

import numpy as np
import torch

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    MolecularLatentProblem, LatentSampling, GenerationTracker,
    postprocess_run, consolidate_all, get_operators, get_ref_dirs,
    ga_run_dir, FSP3_MIN, POP_SIZE, N_GEN, CX_PROB, MUTATION, MUT_PROB,
)


# ═══════════════════════════════════════════════════════════════════════════
#   1. Los tres algoritmos
# ═══════════════════════════════════════════════════════════════════════════

ALGORITMOS = {'NSGA2': NSGA2, 'NSGA3': NSGA3, 'AGEMOEA': AGEMOEA}


# ═══════════════════════════════════════════════════════════════════════════
#   2. El cuerpo de una corrida
# ═══════════════════════════════════════════════════════════════════════════

def correr(alg, args):
    """Una corrida completa de cualquiera de los tres."""

    # El run_id da la misma población inicial en los tres: las semillas quedan
    # pareadas y el análisis puede tomarlas como bloque.
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    model, stoi, itos, latent_dim = load_model()
    mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
    train_smiles = load_train_smiles()

    run_dir = ga_run_dir(alg, args.crossover, args.mutation, args.cx_prob,
                         args.mut_prob, args.pop_size, args.n_gen, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{alg}[{args.crossover}{args.cx_prob:g}"
             f"+{args.mutation}{args.mut_prob:g}]"
             f"/pop{args.pop_size}xgen{args.n_gen}/run_{args.run_id + 1:02d}")
    print(f"[{label}] Iniciando...", flush=True)
    hp = {'crossover': args.crossover, 'mutation': args.mutation,
          'cx_prob': args.cx_prob, 'mut_prob': round(args.mut_prob, 6),
          'fsp3_min': FSP3_MIN}

    cruce, mutacion = get_operators(args.crossover, args.mutation,
                                    args.cx_prob, args.mut_prob)
    extra = {'ref_dirs': get_ref_dirs(args.pop_size)} if alg == 'NSGA3' else {}

    problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker = GenerationTracker(problem, train_smiles)
    algoritmo = ALGORITMOS[alg](pop_size=args.pop_size, sampling=LatentSampling(mus),
                                crossover=cruce, mutation=mutacion,
                                eliminate_duplicates=True, **extra)

    t0 = time.time()
    minimize(problem, algoritmo, ('n_gen', args.n_gen),
             seed=args.run_id, verbose=False, callback=tracker)
    elapsed = time.time() - t0

    metrics, pareto, hv, spacing, validity = postprocess_run(
        alg, args.pop_size, args.n_gen, args.run_id,
        problem, tracker, elapsed, run_dir, hp=hp)

    print(f"[{label}] HV={hv:.4f}  Spacing={spacing:.4f}  "
          f"Valid={validity:.0%}  Feas={metrics['feasibility']:.0%}  "
          f"n={len(pareto)}  QED={metrics['best_qed']}  SA={metrics['best_sa']}  "
          f"Fsp3={metrics['mean_fsp3']}  t={metrics['time_sec']}s", flush=True)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#   3. Línea de comandos
# ═══════════════════════════════════════════════════════════════════════════

def _parser():
    ap = argparse.ArgumentParser(
        prog="experimento.py",
        description="Optimización multi-objetivo del espacio latente VAE.")

    ap.add_argument('--alg', choices=list(ALGORITMOS), type=str.upper,
                    help="Algoritmo a correr.")
    ap.add_argument('--pop_size', type=int, default=POP_SIZE)
    ap.add_argument('--n_gen', type=int, default=N_GEN)
    ap.add_argument('--run_id', type=int, default=None)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help="Dispositivo para el VAE (default: auto → GPU si hay CUDA).")
    ap.add_argument('--generate_summary', action='store_true',
                    help="No corre nada: consolida results/exp4/all_metrics.csv y sale.")

    ap.add_argument('--crossover', choices=['sbx', 'pcx'], default='sbx')
    ap.add_argument('--mutation', choices=['pm'], default=MUTATION)
    ap.add_argument('--cx_prob', type=float, default=CX_PROB,
                    help="Probabilidad de cruce (por apareamiento).")
    ap.add_argument('--mut_prob', type=float, default=MUT_PROB,
                    help="Probabilidad de mutación POR-GEN.")
    return ap


def main():
    ap = _parser()
    args = ap.parse_args()

    if args.generate_summary:
        consolidate_all()
        return
    if args.alg is None:
        ap.error("se requiere --alg (o --generate_summary)")
    if args.run_id is None:
        ap.error("se requiere --run_id")

    # 'auto' respeta el default del módulo.
    if args.device != 'auto':
        set_device(args.device)

    correr(args.alg, args)


if __name__ == "__main__":
    main()
