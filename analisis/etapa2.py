"""
Etapa 2 — cómo se distribuyen las soluciones en el frente.

Tres figuras, una por pregunta, sobre las decisiones que dejó la etapa 1:

  algoritmos   los cinco con la mutación y el reparto elegidos.
  mutacion     las tres configuraciones en un algoritmo.
  reparto      los dos repartos en un algoritmo.

Cada comparación fija un combo de operadores —PCX con polinomial, el que eligió
el experimento anterior— porque el frente cambia mucho entre cruces y superponer
los cuatro combos tapa lo que se quiere mirar.  CMOPSO no tiene operadores y va
con su única configuración.
"""

import glob
import os

from .comun import DISPLAY, Series, _has_runs
from .figuras import (GRID_COLOR_MODES, plot_pareto_comparison,
                      plot_pareto_qed_sa_grid)


# El combo sobre el que se comparan algoritmos, mutaciones y repartos.
CRUCE, TIPO = 'pcx', 'pm'

# Del experimento anterior; la etapa 3 todavía los importa de acá.
COMBO_DIRS = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']

MUT_STD = 'pm'

ORDEN_ALG = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'CMOPSO']


def _dir(results, alg, cruce, tipo, config, pop, gen):
    """El directorio de una configuración, o None si no tiene corridas."""
    mut = '0.05' if config == 'adaptativa' else config
    sufijo = '_adapt' if config == 'adaptativa' else ''
    if alg == 'CMOPSO':
        patron = os.path.join(results, 'CMOPSO',
                              f'pop{pop}_gen{gen}_e*_{tipo}{sufijo}{mut}_vel*')
    else:
        patron = os.path.join(results, alg, f'{cruce}_{tipo}{sufijo}',
                              f'cx*_mut{mut}_pop{pop}_gen{gen}')
    hallados = [d for d in sorted(glob.glob(patron)) if _has_runs(d)]
    return hallados[0] if hallados else None


def _figuras(series, tag, out_dir, superpuestos=True):
    """El grid por serie —que es el que muestra la distribución— y, si se pide,
    los frentes superpuestos."""
    if len(series) < 2:
        print(f"  ⚠ {tag}: {len(series)} serie(s); se omite")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  {tag}: {', '.join(s.label for s in series)}")
    for modo in GRID_COLOR_MODES:
        plot_pareto_qed_sa_grid(series, tag, out_dir, color_by=modo)
    if superpuestos:
        plot_pareto_comparison(series, tag, out_dir)


def etapa2(args):
    pop, gen = (int(x) for x in args.reparto.split('x'))
    out = args.out
    print(f"\n{'='*70}\n  DISTRIBUCIÓN EN EL FRENTE\n"
          f"  mutación {args.mutacion}, reparto {args.reparto}, "
          f"combo {CRUCE.upper()}+{TIPO}\n  {args.results} → {out}\n{'='*70}")

    # 1. Los cinco algoritmos.
    series = []
    for alg in ORDEN_ALG:
        d = _dir(args.results, alg, CRUCE, TIPO, args.mutacion, pop, gen)
        if d:
            series.append(Series(DISPLAY.get(alg, alg), d, color_key=alg))
    _figuras(series, 'algoritmos', os.path.join(out, 'algoritmos'))

    # 2. Las tres configuraciones de mutación, en un algoritmo.
    series = []
    for config in ('0.1', '0.05', 'adaptativa'):
        d = _dir(args.results, args.algoritmo, CRUCE, TIPO, config, pop, gen)
        if d:
            series.append(Series(config, d))
    _figuras(series, 'mutacion', os.path.join(out, 'mutacion'))

    # 3. Los dos repartos, en un algoritmo.
    series = []
    for p, g in ((100, 1000), (200, 500)):
        d = _dir(args.results, args.algoritmo, CRUCE, TIPO, args.mutacion, p, g)
        if d:
            series.append(Series(f'{p}x{g}', d))
    _figuras(series, 'reparto', os.path.join(out, 'reparto'))

    print(f"\n  ✓ {out}")
