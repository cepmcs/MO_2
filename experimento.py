"""
Los experimentos: los cinco algoritmos de la comparación y cómo se corre uno.

Objetivos: QED (↑), SA (↓)  →  pymoo minimiza [-QED, SA].
Fsp3 (↑) entra como constraint de desigualdad (Fsp3 ≥ FSP3_MIN).

Cuatro de los cinco se usan tal cual vienen en pymoo: NSGA-II, NSGA-III, AGE-MOEA
y CMOPSO manejan constraints de forma nativa por dominancia de factibilidad
(Survival.do → split_by_feasibility; en CMOPSO además el archivo de elites filtra
por factibilidad al insertar).  El único que necesita subclase es MOEA/D.

El archivo tiene cuatro partes, en este orden:

  1. MOEADConstr — la subclase que le falta a pymoo.
  2. ALGORITMOS  — los cinco declarados: cómo se construye cada uno, si su
     problema va normalizado, si necesita direcciones de referencia y a qué
     familia de perillas pertenece.  Es la ÚNICA tabla donde figura eso, y es lo
     único que hay que tocar para agregar o cambiar un algoritmo.
  3. correr()    — el cuerpo de una corrida, idéntico para los cinco: semillas,
     VAE, población inicial, minimize y post-procesamiento.
  4. El CLI      — qué perillas acepta cada familia.

Las perillas dependen de la familia del algoritmo, y `--help` las muestra según
el `--alg` que le pases:

  ga  (NSGA2, NSGA3, MOEAD, AGEMOEA)   --crossover --mutation --cx_prob --mut_prob
  pso (CMOPSO)                         --elite_size --mut_prob --vel_rate

Uso:
    python experimento.py --alg nsga2 --pop_size 300 --run_id 0
    python experimento.py --alg nsga2 --pop_size 300 --run_id 0 --crossover pcx --mutation gauss
    python experimento.py --alg cmopso --pop_size 100 --n_gen 1000 --run_id 0
    python experimento.py --alg moead --help        # las perillas de esa familia
    python experimento.py --generate_summary        # consolidar results/all_metrics.csv

Corre UNA configuración por vez.  El grid completo lo lanza run_experiments.py,
que llama a este script una vez por corrida.
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
#   1. La subclase que le falta a pymoo
# ═══════════════════════════════════════════════════════════════════════════

class MOEADConstr(ParallelMOEAD):
    """MOEA/D con dominancia de factibilidad (criterio parameter-less de Deb, 2002).

    Extiende ParallelMOEAD, que es la variante que usa el experimento; el reemplazo
    corre por _replace igual que en la versión loopwise.

    pymoo aborta con `assert not problem.has_constraints()` en _setup, y el criterio
    que lo resolvería está escrito en moead.py pero comentado («not originally proposed
    though and not tested enough») y sin importar siquiera la función que usa.  Acá se
    activa: cada infactible pasa a valer fmax + CV en el espacio escalarizado, o sea
    peor que cualquier factible, y entre infactibles gana el de menor violación.  Ese es
    el mismo ORDEN que la dominancia de factibilidad de los otros cuatro algoritmos,
    expresado sobre los valores descompuestos en vez de sobre el frente.
    """

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
    """Todo lo que distingue a un algoritmo de los otros cuatro.

    construir     (args, sampling, operadores, ref_dirs) → algoritmo de pymoo.
                  'operadores' es la tupla (cruce, mutación) en la familia 'ga'
                  y None en 'pso'; 'ref_dirs' es None si el algoritmo no las usa.
    familia       'ga'  → se barren cruce, mutación y sus probabilidades.
                  'pso' → se barren elite_size, mutación por-gen y velocidad.
                  Decide las perillas del CLI, el directorio de salida y qué
                  columnas de hiperparámetros van a metrics.csv.
    normalizado   True si el problema debe entregar F normalizado a [0,1]^2.
    ref_dirs      True si necesita direcciones de referencia (una por subproblema
                  o por nicho); se generan exactamente pop_size.
    nota          Por qué este algoritmo se configura así.  Se imprime en el
                  --help de su propio algoritmo.
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
    # No lleva pop_size: MOEA/D tiene un subproblema por dirección de referencia,
    # así que el tamaño de población sale de len(ref_dirs) (ver MOEADConstr._setup).
    cruce, mutacion = operadores
    return MOEADConstr(ref_dirs=ref_dirs, n_neighbors=20,
                       prob_neighbor_mating=0.9, sampling=sampling,
                       crossover=cruce, mutation=mutacion)


def _cmopso(args, sampling, operadores, ref_dirs):
    algoritmo = CMOPSO(pop_size=args.pop_size, elite_size=args.elite_size,
                       max_velocity_rate=args.vel_rate, sampling=sampling)
    # CMOPSO fija PolynomialMutation(prob=mutation_rate) en su __init__, sin parámetro
    # para pasar el prob_var: `prob` es por-INDIVIDUO y deja el por-gen en el default
    # 1/n_var (=1/256≈0.0039), ~8x más suave que el 0.031 que ganó en el grid GA.  Se
    # pisa acá para barrer la misma perilla que los GA y que la mutación sea comparable
    # entre las cinco familias.
    algoritmo.mutation = PM(prob=1.0, prob_var=args.mut_prob)
    return algoritmo


ALGORITMOS = {
    'NSGA2': Algoritmo(
        construir=_nsga2, familia='ga', normalizado=False, ref_dirs=False,
        nota="El constraint lo maneja pymoo de forma nativa (dominancia de "
             "factibilidad); no necesita ningún cambio más allá del problema."),

    'NSGA3': Algoritmo(
        construir=_nsga3, familia='ga', normalizado=False, ref_dirs=True,
        nota="Usa vectores de referencia: se generan exactamente pop_size "
             "direcciones equiespaciadas con Das-Dennis uniforme (ver "
             "utils_mo.get_ref_dirs).  Constraint nativo."),

    'MOEAD': Algoritmo(
        construir=_moead, familia='ga', normalizado=True, ref_dirs=True,
        nota="El único que necesita subclase para el constraint: el MOEA/D de "
             "pymoo aborta con un assert (ver MOEADConstr).  Corre sobre "
             "ParallelMOEAD, la variante síncrona que evalúa todo el offspring "
             "en lote y permite el decode batcheado.  El problema va normalizado "
             "porque la escala cruda de SA domina la descomposición Tchebycheff."),

    'AGEMOEA': Algoritmo(
        construir=_agemoea, familia='ga', normalizado=False, ref_dirs=False,
        nota="Adapta la presión de selección según la geometría del frente de "
             "Pareto actual.  Constraint nativo."),

    'CMOPSO': Algoritmo(
        construir=_cmopso, familia='pso', normalizado=True, ref_dirs=False,
        nota="CMOPSO (Zhang et al., Inf. Sci. 427:63-76, 2018) reemplaza al "
             "MOPSO_CD de la etapa anterior: maneja el constraint de forma "
             "nativa (dominancia de factibilidad en la supervivencia SPEA2 y en "
             "el archivo de elites), mientras que MOPSO_CD ordena por "
             "NonDominatedSorting crudo sobre F y lo ignora en silencio.  Sus "
             "perillas no son las de MOPSO_CD: la ecuación de velocidad es "
             "v' = R1·v + R2·(ganador − p), con R1/R2 aleatorios por dimensión y "
             "sin pbest, así que w, c1 y c2 no existen.  El problema va "
             "normalizado porque la escala cruda de SA domina la velocidad de "
             "las partículas."),
}

# Los cuatro genéticos, en el orden en que se presentan.  Lo usa run_experiments.py
# para armar el grid, y es la definición de "algoritmo con operadores de cruce".
ALGS_GA = [nombre for nombre, a in ALGORITMOS.items() if a.familia == 'ga']


# ═══════════════════════════════════════════════════════════════════════════
#   3. El cuerpo de una corrida
# ═══════════════════════════════════════════════════════════════════════════

def _perillas(alg, args, latent_dim):
    """Lo que depende de la familia: directorio de salida, etiqueta de progreso y
    las columnas de hiperparámetros que van a metrics.csv.

    Devuelve (run_dir, label, hp, mut_prob).  El mut_prob se resuelve acá porque
    en la familia 'ga' su default es 1/latent_dim, que no se conoce hasta cargar
    el VAE."""
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
    """Una corrida completa de cualquiera de los cinco.

    Es el cuerpo que antes estaba copiado en los cinco experimento_*.py: fijar
    las semillas, cargar el VAE y la población inicial, construir el problema y
    el algoritmo, correr minimize y post-procesar.  Lo único que cambia entre
    algoritmos sale de ALGORITMOS y de la familia de perillas."""
    spec = ALGORITMOS[alg]

    # Semillas: la población inicial se muestrea con random_state=run_id, así que
    # el run_id k da la MISMA población en los cinco algoritmos.  De ahí que los
    # tests del análisis puedan tomar la semilla como bloque.
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

    # Direcciones de referencia: exactamente pop_size, equiespaciadas sobre el
    # símplex (Das-Dennis uniforme, exacto con 2 objetivos).  Cuestan ~1 ms.
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
    """Parser con las perillas comunes y, si ya se sabe el algoritmo, las de su
    familia.

    Se construye dos veces: una sin ayuda para averiguar el --alg, y otra ya con
    la familia para el parseo de verdad.  Así `--alg cmopso --help` muestra las
    perillas de CMOPSO, y pasar una ajena (--vel_rate a un genético) es un error
    y no algo que se ignore en silencio."""
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
    # Primera pasada, sin ayuda propia, solo para saber el algoritmo: así un
    # `--alg cmopso --help` llega al parser definitivo y muestra SUS perillas.
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

    # 'auto' respeta el default del módulo (GPU si hay CUDA).
    if args.device != 'auto':
        set_device(args.device)

    correr(args.alg, args)


if __name__ == "__main__":
    main()
