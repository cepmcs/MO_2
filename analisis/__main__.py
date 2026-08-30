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
  figuras    finalistas/ (o winners/ con --operadores) → las figuras del reporte

Uso:
    python -m analisis etapa1 [--algorithms NSGA2 CMOPSO] [--metric spacing]
    python -m analisis etapa2 [--algorithms NSGA2 MOEAD] [--metric igd_plus]
    python -m analisis etapa3 [--finalistas otra_carpeta]
    python -m analisis etapa4 [--metric igd_plus]
    python -m analisis moleculas [--out figura.png]
    python -m analisis figuras [--operadores] [--algorithms NSGA2 NSGA3]
"""

import argparse
import os

from .comun import (
    ALGORITHM_ORDER, BASELINES_DIR, FINALISTAS_DIR, METRICS_CSV, OUT_ALGORITMOS,
    OUT_BASELINES, OUT_FRENTE, OUT_HP, OUT_OPERADORES, PLOTS_DIR, RESULTADOS_DIR,
    WINNERS_DIR, _has_runs, build_finalist_series, build_operator_series_winners,
    winner_cfg_dir,
)
from .figuras import _generate_report
from .etapa1 import HP_METRICS, etapa1
from .etapa2 import OP_INDICATORS, etapa2
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


def run_operator_comparison(algorithms, winners_dir):
    """Un reporte por algoritmo comparando sus cuatro combos de operadores, sobre
    las configuraciones que ganaron su bloque en la etapa 1.

    Sale a plots/operadores/<ALG>/winners/, un nivel debajo de las tablas que
    deja la etapa 2."""
    if not os.path.isdir(winners_dir):
        print(f"No existe {winners_dir}")
        return
    algorithms = algorithms or sorted(
        d for d in os.listdir(winners_dir)
        if os.path.isdir(os.path.join(winners_dir, d)))
    tag = "winners"

    print(f"\n{'='*60}")
    print("  Comparación de operadores (por algoritmo)")
    print(f"  Origen: {winners_dir}")
    print(f"  Algoritmos candidatos: {', '.join(algorithms)}")
    print(f"{'='*60}")

    generated = []
    for alg in algorithms:
        series = build_operator_series_winners(alg, winners_dir)
        if len(series) < 2:
            print(f"\n  ⚠ {alg}: solo {len(series)} combo(s) con datos; "
                  f"se omite (se requieren ≥2 para comparar).")
            continue
        output_dir = os.path.join(PLOTS_DIR, "operadores", alg, tag)
        _generate_report(series, tag, output_dir,
                         f"Comparación de Operadores — {alg}")
        generated.append((alg, output_dir))

    print(f"\n{'='*60}")
    if generated:
        print("  ✅ Reportes de operadores generados:")
        for alg, out in generated:
            print(f"     {alg}: {out}")
    else:
        print("  ⚠ No se generó ningún reporte (cada algoritmo necesita "
              "≥2 combos de operadores con datos).")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='etapa', required=True,
                            metavar='etapa1|etapa2|etapa3|etapa4|moleculas|figuras')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p1 = sub.add_parser('etapa1', formatter_class=fmt,
                        help="Selección de hiperparámetros por combinación de "
                             "operadores.")
    p1.add_argument('--csv', default=METRICS_CSV, help="CSV consolidado del grid.")
    p1.add_argument('--out', default=OUT_HP, help="Directorio de salida.")
    p1.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a analizar (default: todos).")
    p1.add_argument('--metric', default='hypervolume', choices=list(HP_METRICS),
                    help="Métrica de selección.")
    p1.set_defaults(func=etapa1)

    p2 = sub.add_parser('etapa2', formatter_class=fmt,
                        help="Comparación de operadores por algoritmo.")
    p2.add_argument('--winners', default=WINNERS_DIR)
    p2.add_argument('--out', default=OUT_OPERADORES)
    p2.add_argument('--algorithms', nargs='+', default=None)
    p2.add_argument('--metric', default='hypervolume',
                    choices=[c for c, _, _ in OP_INDICATORS],
                    help="Indicador con el que se elige el combo ganador.")
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

    pf = sub.add_parser('figuras', formatter_class=fmt,
                        help="Figuras comparativas: los cinco algoritmos, o los "
                             "cuatro combos de operadores con --operadores.")
    pf.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a comparar (auto-detecta si no se especifica).")
    pf.add_argument('--operadores', action='store_true',
                    help="Compara variantes de operadores por algoritmo.")
    pf.add_argument('--winners', default=WINNERS_DIR,
                    help="Con --operadores: las ganadoras de la etapa 1.")
    pf.add_argument('--finalistas', default=FINALISTAS_DIR,
                    help="Sin --operadores: la configuración elegida de cada algoritmo.")
    pf.set_defaults(func=lambda a: (run_operator_comparison(a.algorithms, a.winners)
                                    if a.operadores else
                                    run_algorithm_comparison(a.algorithms, a.finalistas)))

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
