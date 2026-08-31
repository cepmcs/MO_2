"""
Los cinco algoritmos de la comparación y el cuerpo de una corrida.

Para agregar o cambiar un algoritmo se toca la tabla ALGORITMOS.  Las perillas
dependen de la familia; --help muestra las del --alg que pases:
  ga  (NSGA2, NSGA3, MOEAD, AGEMOEA)   --crossover --mutation --cx_prob --mut_prob
  pso (CMOPSO)                         --elite_size --mut_prob --vel_rate

Corre UNA configuración por vez; el grid lo lanza run_experiments.py.
"""
import argparse
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.moead import ParallelMOEAD, default_decomp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.misc import parameter_less

from utils_mo import (
    load_model, load_seed_mus, load_train_smiles, set_device,
    MolecularLatentProblem, NormalizedMolecularLatentProblem,
    LatentSampling, GenerationTracker,
    postprocess_run, consolidate_all, get_operators, get_ref_dirs,
    ga_run_dir, cmopso_run_dir, FSP3_MIN,
)


# ═══════════════════════════════════════════════════════════════════════════
#   1. MOEA/D con constraints
# ═══════════════════════════════════════════════════════════════════════════

class MOEADConstr(ParallelMOEAD):
    """MOEA/D con dominancia de factibilidad.

    El de pymoo aborta con un assert ante constraints.  Acá cada infactible vale
    fmax + CV en el espacio escalarizado: peor que cualquier factible, y entre
    infactibles gana el de menor violación."""

    def _setup(self, problem, **kwargs):
        # Igual que MOEAD._setup pero sin el assert que rechaza constraints.
        if self.ref_dirs is None:
            from pymoo.util.reference_direction import default_ref_dirs
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(
            cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort'
        )[:, :self.n_neighbors]
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _replace(self, k, off):
        pop = self.pop
        N = self.neighbors[k]
        FV = self.decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :],
                                   ideal_point=self.ideal)
        off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :],
                                       ideal_point=self.ideal)

        if self.problem.has_constraints():
            CV = pop[N].get("CV")[:, 0]
            off_CV = np.full(len(off_FV), off.CV[0])
            fmax = max(FV.max(), off_FV.max())
            FV = parameter_less(FV, CV, fmax=fmax)
            off_FV = parameter_less(off_FV, off_CV, fmax=fmax)

        I = np.where(off_FV < FV)[0]
        pop[N[I]] = off


# ═══════════════════════════════════════════════════════════════════════════
#   2. Los cinco algoritmos
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Algoritmo:
    """Lo que distingue a un algoritmo de los otros.

    construir     (args, sampling, operadores, ref_dirs) → algoritmo de pymoo.
    familia       'ga' o 'pso'; decide las perillas del CLI y el run_dir.
    normalizado   el problema entrega F normalizado a [0,1]^2.
    ref_dirs      necesita pop_size direcciones de referencia.
    nota          sale en su --help.
    """
    construir: Callable
    familia: str
    normalizado: bool
    ref_dirs: bool
    nota: str


def _nsga2(args, sampling, operadores, ref_dirs):
    cruce, mutacion = operadores
    return NSGA2(pop_size=args.pop_size, sampling=sampling,
                 crossover=cruce, mutation=mutacion, eliminate_duplicates=True)


def _nsga3(args, sampling, operadores, ref_dirs):
    cruce, mutacion = operadores
    return NSGA3(ref_dirs=ref_dirs, pop_size=args.pop_size, sampling=sampling,
                 crossover=cruce, mutation=mutacion, eliminate_duplicates=True)


def _agemoea(args, sampling, operadores, ref_dirs):
    cruce, mutacion = operadores
    return AGEMOEA(pop_size=args.pop_size, sampling=sampling,
                   crossover=cruce, mutation=mutacion, eliminate_duplicates=True)


def _moead(args, sampling, operadores, ref_dirs):
    # Sin pop_size: sale de len(ref_dirs), un subproblema por dirección.
    cruce, mutacion = operadores
    return MOEADConstr(ref_dirs=ref_dirs, n_neighbors=20,
                       prob_neighbor_mating=0.9, sampling=sampling,
                       crossover=cruce, mutation=mutacion)


def _cmopso(args, sampling, operadores, ref_dirs):
    algoritmo = CMOPSO(pop_size=args.pop_size, elite_size=args.elite_size,
                       max_velocity_rate=args.vel_rate, sampling=sampling)
    # Se pisa la mutación para barrer la misma perilla que los GA.
    algoritmo.mutation = PM(prob=1.0, prob_var=args.mut_prob)
    return algoritmo


ALGORITMOS = {
    'NSGA2': Algoritmo(
        construir=_nsga2, familia='ga', normalizado=False, ref_dirs=False,
        nota="Constraint nativo."),

    'NSGA3': Algoritmo(
        construir=_nsga3, familia='ga', normalizado=False, ref_dirs=True,
        nota="Usa pop_size direcciones Das-Dennis.  Constraint nativo."),

    'MOEAD': Algoritmo(
        construir=_moead, familia='ga', normalizado=True, ref_dirs=True,
        nota="Necesita subclase (MOEADConstr): el de pymoo no acepta constraints. "
             "Va normalizado: la escala de SA domina la descomposición "
             "Tchebycheff."),

    'AGEMOEA': Algoritmo(
        construir=_agemoea, familia='ga', normalizado=False, ref_dirs=False,
        nota="Adapta la presión de selección a la geometría del frente.  "
             "Constraint nativo."),

    'CMOPSO': Algoritmo(
        construir=_cmopso, familia='pso', normalizado=True, ref_dirs=False,
        nota="Reemplaza al MOPSO_CD anterior, que ignoraba el constraint.  Va "
             "normalizado: la escala de SA domina la velocidad."),
}

# Los cuatro genéticos: los que tienen operadores que barrer.
ALGS_GA = [nombre for nombre, a in ALGORITMOS.items() if a.familia == 'ga']


# ═══════════════════════════════════════════════════════════════════════════
#   3. El cuerpo de una corrida
# ═══════════════════════════════════════════════════════════════════════════

def _perillas(alg, args, latent_dim):
    """(run_dir, label, hp, mut_prob) según la familia."""
    if ALGORITMOS[alg].familia == 'pso':
        run_dir = cmopso_run_dir(args.pop_size, args.n_gen, args.elite_size,
                                 args.mut_prob, args.vel_rate, args.run_id)
        label = (f"{alg}[e{args.elite_size:g}_mut{args.mut_prob:g}"
                 f"_vel{args.vel_rate:g}]")
        hp = {'elite_size': args.elite_size, 'mut_prob': args.mut_prob,
              'vel_rate': args.vel_rate, 'fsp3_min': FSP3_MIN}
        return run_dir, label, hp, args.mut_prob

    mut_prob = args.mut_prob if args.mut_prob is not None else 1.0 / latent_dim
    run_dir = ga_run_dir(alg, args.crossover, args.mutation, args.cx_prob,
                         mut_prob, args.pop_size, args.n_gen, args.run_id)
    label = (f"{alg}[{args.crossover}{args.cx_prob:g}"
             f"+{args.mutation}{mut_prob:g}]")
    hp = {'crossover': args.crossover, 'mutation': args.mutation,
          'cx_prob': args.cx_prob, 'mut_prob': round(mut_prob, 6),
          'fsp3_min': FSP3_MIN}
    return run_dir, label, hp, mut_prob


def correr(alg, args):
    """Una corrida completa de cualquiera de los cinco."""
    spec = ALGORITMOS[alg]

    # El run_id da la misma población inicial en los cinco: las semillas quedan
    # pareadas y el análisis puede tomarlas como bloque.
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_id)

    model, stoi, itos, latent_dim = load_model()
    mus = load_seed_mus(model, stoi, args.pop_size, args.run_id)
    train_smiles = load_train_smiles()

    run_dir, etiqueta_cfg, hp, mut_prob = _perillas(alg, args, latent_dim)
    os.makedirs(run_dir, exist_ok=True)
    label = (f"{etiqueta_cfg}/pop{args.pop_size}xgen{args.n_gen}"
             f"/run_{args.run_id + 1:02d}")
    print(f"[{label}] Iniciando...", flush=True)

    ref_dirs = get_ref_dirs(args.pop_size) if spec.ref_dirs else None
    operadores = (get_operators(args.crossover, args.mutation,
                                args.cx_prob, mut_prob)
                  if spec.familia == 'ga' else None)

    clase_problema = (NormalizedMolecularLatentProblem if spec.normalizado
                      else MolecularLatentProblem)
    problem = clase_problema(model, stoi, itos, latent_dim)
    tracker = GenerationTracker(problem, train_smiles)
    algoritmo = spec.construir(args, LatentSampling(mus), operadores, ref_dirs)

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
#   4. Línea de comandos
# ═══════════════════════════════════════════════════════════════════════════

def _parser(alg=None, ayuda=True):
    """Perillas comunes y, si ya se sabe el algoritmo, las de su familia.

    Se construye dos veces: primero sin ayuda, para averiguar el --alg.  Así una
    perilla ajena es error."""
    spec = ALGORITMOS[alg] if alg else None
    ap = argparse.ArgumentParser(
        prog="experimento.py", add_help=ayuda,
        description="Optimización multi-objetivo del espacio latente VAE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(textwrap.fill(f"{alg}: {spec.nota}", 78, subsequent_indent='  ')
                if spec else
                "Pasá --alg <nombre> --help para ver las perillas de ese algoritmo."))

    ap.add_argument('--alg', choices=list(ALGORITMOS), type=str.upper,
                    help="Algoritmo a correr.")
    ap.add_argument('--pop_size', type=int, default=None)
    ap.add_argument('--n_gen', type=int, default=500)
    ap.add_argument('--run_id', type=int, default=None)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help="Dispositivo para el VAE (default: auto → GPU si hay CUDA).")
    ap.add_argument('--generate_summary', action='store_true',
                    help="No corre nada: consolida results/all_metrics.csv y sale.")

    if spec is None:
        return ap

    if spec.familia == 'ga':
        ap.add_argument('--crossover', choices=['sbx', 'pcx'], default='sbx')
        ap.add_argument('--mutation', choices=['pm', 'gauss'], default='pm')
        ap.add_argument('--cx_prob', type=float, default=0.9,
                        help="Probabilidad de cruce (por apareamiento).")
        ap.add_argument('--mut_prob', type=float, default=None,
                        help="Probabilidad de mutación POR-GEN (default: 1/n_var).")
    else:
        ap.add_argument('--elite_size', type=int, default=10,
                        help="Tamaño al que se poda el archivo de elites.")
        ap.add_argument('--mut_prob', type=float, default=0.031,
                        help="Probabilidad de mutación POR-GEN (prob_var), como "
                             "en el grid GA.")
        ap.add_argument('--vel_rate', type=float, default=0.2,
                        help="max_velocity_rate: V_max = vel_rate · (xu − xl).")
    return ap


def main():
    # Sin ayuda propia: así '--alg X --help' llega al parser final.
    conocidos, _ = _parser(ayuda=False).parse_known_args()

    if conocidos.generate_summary:
        consolidate_all()
        return

    ap = _parser(conocidos.alg)
    args = ap.parse_args()
    if args.alg is None:
        ap.error("se requiere --alg (o --generate_summary)")
    if args.pop_size is None:
        ap.error("se requiere --pop_size")
    if args.run_id is None:
        ap.error("se requiere --run_id")

    # 'auto' respeta el default del módulo.
    if args.device != 'auto':
        set_device(args.device)

    correr(args.alg, args)


if __name__ == "__main__":
    main()
