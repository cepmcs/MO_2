"""
Variantes de algoritmos pymoo necesarias para la etapa con constraint (Fsp3 ≥ umbral).

Cuatro de los cinco algoritmos se usan tal cual vienen en pymoo: NSGA-II, NSGA-III,
AGE-MOEA y CMOPSO manejan constraints de forma nativa por dominancia de factibilidad
(Survival.do → split_by_feasibility; en CMOPSO además el archivo de elites filtra por
factibilidad al insertar).  El único que necesita subclase es MOEA/D.

Nota sobre CMOPSO, relevante al describirlo en la tesis: la implementación de pymoo se
desvía del paper original (Zhang et al., Inf. Sci. 427:63-76, 2018) en su contribución
central.  El paper selecciona los elites FRESCOS del enjambre actual en cada generación
y remarca que CMOPSO «does not need any external archive» (Alg. 2 línea 3); pymoo usa un
archivo externo persistente (MultiObjectiveArchive con max_size=pop_size,
truncate_size=elite_size).  Consecuencia práctica: `elite_size` NO es el γ del paper —
el archivo crece hasta pop_size y solo entonces se poda, así que el conjunto real de
elites oscila (medido: 2..100 con pop_size=100, con media 20/29/42 para elite_size
10/25/50).  Sigue siendo una perilla con efecto monótono y barrible, pero conviene
citarla como «elite_size de la implementación de pymoo» y no como el γ del artículo.
"""

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.moo.moead import ParallelMOEAD, default_decomp
from pymoo.util.misc import parameter_less


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
