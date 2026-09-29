"""
Línea de comandos del análisis.

Cada comando lee results/exp4/ (--results) y escribe en plots/exp4/.

  operadores  PCX contra SBX, un reporte por algoritmo      →  plots/exp4/operadores/
  algoritmos  los tres algoritmos entre sí, por cruce       →  plots/exp4/comparacion_final/
              y el frente conjunto de las seis configs.     →  plots/exp4/frente_conjunto/
  moleculas   las de mayor QED de cada algoritmo            →  plots/exp4/frente_conjunto/
  figuras     figuras comparativas de los algoritmos        →  plots/exp4/comparacion_final/

Uso:
    python -m analisis operadores [--algorithms NSGA2 NSGA3] [--metric igd_plus]
    python -m analisis algoritmos [--crossovers pcx]
    python -m analisis moleculas [--out figura.png]
    python -m analisis figuras [--crossovers pcx sbx]
"""

import argparse
import os

from .comun import (
    ALGORITHM_ORDER, CRUCES, OUT_ALGORITMOS, OUT_FRENTE, OUT_OPERADORES,
    RESULTADOS_DIR, series_algoritmos,
)
from .figuras import _generate_report
from .indicadores import load_reference_front
from .algoritmos import MOLECULAS_OUT, algoritmos, moleculas
from .operadores import OP_INDICATORS, operadores


# ─── Las figuras comparativas ────────────────────────────────────────────────

def run_algorithm_comparison(cruces, results, out):
    """Comparación entre algoritmos: un reporte de figuras y tablas por cruce."""
    if not os.path.isdir(results):
        print(f"No existe {results}")
        return
    frente = load_reference_front(results)
    for cruce in cruces or CRUCES:
        series = series_algoritmos(results, cruce)
        if len(series) < 2:
            print(f"Cruce {cruce.upper()}: se necesitan ≥2 algoritmos con datos "
                  f"en {results}")
            continue
        print(f"\n{'='*60}")
        print(f"  Comparación entre algoritmos — cruce {cruce.upper()}")
        print(f"  Origen: {results}")
        print(f"  Algoritmos: {', '.join(s.label for s in series)}")
        print(f"{'='*60}")
        output_dir = os.path.join(out, cruce)
        _generate_report(series, cruce.upper(), output_dir,
                         f"Comparación entre algoritmos — cruce {cruce.upper()}",
                         frente)
        print(f"\n{'='*60}\n  ✅ Generación completa: {output_dir}\n{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='comando', required=True,
                            metavar='operadores|algoritmos|moleculas|figuras')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--results', default=RESULTADOS_DIR,
                      help="Carpeta con los resultados del experimento.")

    p1 = sub.add_parser('operadores', parents=[base], formatter_class=fmt,
                        help="Comparación de los cruces PCX y SBX por algoritmo.")
    p1.add_argument('--out', default=OUT_OPERADORES, help="Directorio de salida.")
    p1.add_argument('--algorithms', nargs='+', choices=ALGORITHM_ORDER, default=None,
                    help="Algoritmos a analizar (default: todos).")
    p1.add_argument('--metric', default='hypervolume',
                    choices=[c for c, _, _ in OP_INDICATORS],
                    help="Indicador sobre el que se contrastan los cruces.")
    p1.set_defaults(func=operadores)

    p2 = sub.add_parser('algoritmos', parents=[base], formatter_class=fmt,
                        help="Comparación estadística entre algoritmos y frente "
                             "conjunto.")
    p2.add_argument('--out', default=OUT_ALGORITMOS, help="Directorio de salida.")
    p2.add_argument('--out-frente', default=OUT_FRENTE,
                    help="Directorio del análisis del frente conjunto.")
    p2.add_argument('--crossovers', nargs='+', choices=CRUCES, default=None,
                    help="Cruces a comparar (default: todos).")
    p2.set_defaults(func=algoritmos)

    pm = sub.add_parser('moleculas', parents=[base], formatter_class=fmt,
                        help="Moléculas representativas del frente de cada "
                             "algoritmo.")
    pm.add_argument('--out', default=MOLECULAS_OUT)
    pm.set_defaults(func=moleculas)

    pf = sub.add_parser('figuras', parents=[base], formatter_class=fmt,
                        help="Figuras comparativas de los algoritmos, por cruce.")
    pf.add_argument('--out', default=OUT_ALGORITMOS, help="Directorio de salida.")
    pf.add_argument('--crossovers', nargs='+', choices=CRUCES, default=None,
                    help="Cruces a comparar (default: todos).")
    pf.set_defaults(func=lambda a: run_algorithm_comparison(a.crossovers,
                                                            a.results, a.out))

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
