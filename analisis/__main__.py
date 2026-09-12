"""
Línea de comandos del análisis.

Las cinco etapas encadenadas (cada una lee lo que dejó la anterior) más las
figuras comparativas, que son el otro entregable y no dependen de las etapas.

  etapa1     resultados/grid/all_metrics.csv          →  plots/hiperparametros/
  etapa2     resultados/winners/                      →  plots/operadores/
  etapa3     resultados/finalistas/                   →  plots/comparacion_final/
                                                      +  plots/frente_conjunto/
  etapa4     resultados/finalistas/ + .../baselines/  →  plots/baselines/
  moleculas  resultados/finalistas/                   →  plots/frente_conjunto/
  figuras    resultados/finalistas/                    →  plots/comparacion_final/

Uso:
    python -m analisis etapa1 [--algorithms NSGA2 CMOPSO] [--metric spacing]
    python -m analisis etapa2 [--algorithms NSGA2 MOEAD] [--metric igd_plus]
    python -m analisis etapa3 [--finalistas otra_carpeta]
    python -m analisis etapa4 [--metric igd_plus]
    python -m analisis moleculas [--out figura.png]
    python -m analisis figuras [--algorithms NSGA2 NSGA3]
"""

import argparse
import os

from .comun import (
    ALGORITHM_ORDER, BASELINES_DIR, FINALISTAS_DIR, METRICS_CSV, OUT_ALGORITMOS,
    OUT_BASELINES, OUT_FRENTE, OUT_HP, OUT_OPERADORES, PLOTS_DIR, RESULTADOS_DIR,
    OUT_TABLAS, RESULTS_DIR, WINNERS_DIR,
    _has_runs, build_finalist_series, winner_cfg_dir,
)
from .figuras import _generate_report
from .etapa1 import etapa1
from .tablas_pdf import comparar_repartos, tablas_pdf
from .etapa2 import etapa2
from .etapa3 import MOLECULAS_OUT, etapa3, moleculas
from .etapa4 import etapa4


# ─── Las figuras comparativas ────────────────────────────────────────────────

def run_algorithm_comparison(algorithms, finalistas):
    """Comparación final entre algoritmos: la configuración elegida de cada uno,
    leída de finalistas/<ALG>/ (symlinks a la ganadora dentro de winners/)."""
    if not os.path.isdir(finalistas):
        print(f"No existe {finalistas}")
        return
    algorithms = algorithms or [a for a in ALGORITHM_ORDER
                                if _has_runs(os.path.join(finalistas, a))]
    series = build_finalist_series(algorithms, finalistas)
    if len(series) < 2:
        print(f"Se necesitan ≥2 algoritmos con datos en {finalistas}")
        return
    print(f"\n{'='*60}")
    print("  Comparación final entre algoritmos")
    print(f"  Origen: {finalistas}")
    print(f"  Algoritmos: {', '.join(s.label for s in series)}")
    print(f"{'='*60}")
    output_dir = os.path.join(PLOTS_DIR, "comparacion_final")
    _generate_report(series, "final", output_dir,
                     "Comparación Final — Todos los Algoritmos")
    print(f"\n{'='*60}\n  ✅ Generación completa: {output_dir}\n{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='etapa', required=True,
                            metavar='etapa1|etapa2|etapa3|etapa4|moleculas|figuras|pdf')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p1 = sub.add_parser('etapa1', formatter_class=fmt,
                        help="Tablas del experimento de mutación: qué mutación, "
                             "qué reparto y los cinco algoritmos.")
    p1.add_argument('--results', default=RESULTS_DIR, help="Raíz de las corridas.")
    p1.add_argument('--out', default=OUT_TABLAS, help="Directorio de salida.")
    p1.add_argument('--metric', default='hypervolume',
                    help="Métrica que ordena los rangos.")
    p1.add_argument('--mutacion', default=None,
                    help="Fija la configuración del bloque 2 (default: la mejor).")
    p1.add_argument('--reparto', default=None,
                    help="Fija el reparto del bloque 3 (default: el mejor).")
    p1.set_defaults(func=etapa1)

    p2 = sub.add_parser('etapa2', formatter_class=fmt,
                        help="Cómo se distribuyen las soluciones en el frente.")
    p2.add_argument('--results', default=RESULTS_DIR)
    p2.add_argument('--out', default=os.path.join(PLOTS_DIR, 'mutacion', 'frentes'))
    p2.add_argument('--mutacion', default='0.1', help="Configuración de mutación.")
    p2.add_argument('--reparto', default='100x1000')
    p2.add_argument('--algoritmo', default='NSGA2',
                    help="Algoritmo de las comparaciones de mutación y reparto.")
    p2.set_defaults(func=etapa2)

    p3 = sub.add_parser('etapa3', formatter_class=fmt,
                        help="Comparación estadística entre algoritmos.")
    p3.add_argument('--finalistas', default=FINALISTAS_DIR)
    p3.add_argument('--out', default=OUT_ALGORITMOS)
    p3.add_argument('--out-frente', default=OUT_FRENTE,
                    help="Directorio del análisis del frente conjunto.")
    p3.add_argument('--winners', default=WINNERS_DIR,
                    help="De acá salen las dos ramas de cruce del pool.")
    p3.set_defaults(func=etapa3)

    p4 = sub.add_parser('etapa4', formatter_class=fmt,
                        help="Baselines vs algoritmos multiobjetivo.")
    p4.add_argument('--finalistas', default=FINALISTAS_DIR)
    p4.add_argument('--baselines', default=BASELINES_DIR)
    p4.add_argument('--out', default=OUT_BASELINES)
    p4.add_argument('--metric', default='hypervolume')
    p4.add_argument('--metric-label', default='hipervolumen')
    p4.set_defaults(func=etapa4)

    pm = sub.add_parser('moleculas', formatter_class=fmt,
                        help="Moléculas representativas del frente de cada "
                             "algoritmo.")
    pm.add_argument('--finalistas', default=FINALISTAS_DIR)
    pm.add_argument('--winners', default=WINNERS_DIR,
                    help="De acá salen las dos ramas de cruce de cada algoritmo.")
    pm.add_argument('--out', default=MOLECULAS_OUT)
    pm.set_defaults(func=moleculas)

    # Solo compara algoritmos.  Existió un modo --operadores que corría este
    # mismo reporte sobre los cuatro combos de cada algoritmo, a
    # plots/operadores/<ALG>/winners/; se quitó porque no se usaba y lo que
    # decide entre operadores son las tablas de la etapa 2.
    pp = sub.add_parser('pdf', formatter_class=fmt,
                        help="Un PDF con las dos tablas completas.")
    pp.add_argument('--results', default=RESULTS_DIR)
    pp.add_argument('--out', default=os.path.join(PLOTS_DIR, 'mutacion'))
    pp.add_argument('--reparto', default='100x1000')
    pp.set_defaults(func=tablas_pdf)

    pr = sub.add_parser('pdf-repartos', formatter_class=fmt,
                        help="PDF comparando los dos presupuestos.")
    pr.add_argument('--results', default=RESULTS_DIR)
    pr.add_argument('--out', default=os.path.join(PLOTS_DIR, 'mutacion'))
    pr.add_argument('--mutacion', default='0.1')
    pr.set_defaults(func=comparar_repartos)

    pf = sub.add_parser('figuras', formatter_class=fmt,
                        help="Figuras comparativas de los cinco algoritmos.")
    pf.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a comparar (auto-detecta si no se especifica).")
    pf.add_argument('--finalistas', default=FINALISTAS_DIR,
                    help="La configuración elegida de cada algoritmo.")
    pf.set_defaults(func=lambda a: run_algorithm_comparison(a.algorithms,
                                                            a.finalistas))

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
