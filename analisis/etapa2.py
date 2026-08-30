"""
Etapa 2 — comparación de combinaciones de operadores, por algoritmo.

Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (lo que ganó su bloque en
la etapa 1) y contrasta los cuatro combos de una vez: Friedman con las semillas
como bloques y los seis pares con Wilcoxon corregidos por Holm.  Dentro del
grupo ganador, que el post-hoc no separa, se conserva la mutación estándar.
"""

import os

import numpy as np
import pandas as pd

from .comun import (
    DISPLAY,
    GA_ALGS,
    _build_series_value_getter,
    _fmt_p,
    _latex_escape,
    _num,
    _write_tex,
    build_operator_series_winners,
    compare_indicator,
    fmt_groups,
    generate_latex_comparison_tables,
    homogeneous_groups,
    rank_biserial,
)
from .indicadores import build_reference_front, compute_indicators_per_run
from .figuras import GRID_COLOR_MODES, plot_frente_conjunto, plot_pareto_qed_sa_grid


# Los combos como nombres de directorio bajo winners/ (con guion bajo).
COMBO_DIRS = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']


# (columna, etiqueta, mayor_es_mejor).  Decide el hipervolumen; el resto está
# para poder rehacer la etapa con otro criterio desde --metric.
OP_INDICATORS = [
    ('hypervolume', 'Hipervolumen',      True),
    ('igd_plus',    'IGD$^+$',           False),
    ('epsilon',     r'$\epsilon^+$',     False),
    ('spacing',     'Espaciamiento',     False),
    ('n_pareto',    'Tamaño de Pareto',  True),
    ('validity',    'Validez',           True),
    ('feasibility', 'Factibilidad',      True),
    ('uniqueness',  'Unicidad',          True),
]



# ─── Comparación de operadores ───────────────────────────────────────────────
#
#   El grid es un 2×2 (cruce × mutación) y los cuatro combos se contrastan de una
#   vez: Friedman con las semillas como bloques y los seis pares con Wilcoxon
#   corregidos por Holm.  Decide el indicador de --metric; dentro del grupo
#   ganador, que el post-hoc no separa, se conserva la mutación estándar.
#
#   La contribución al frente conjunto se evaluó como criterio alternativo y se
#   descartó.  Queda solo su figura; el test propio está en la etapa 3.

MUT_STD = 'pm'          # la estándar; se conserva cuando el test no separa



def write_tabla_operadores(res, groups, labels, alg, out_dir, get_values, col):
    """Las seis comparaciones por pares entre los cuatro combos, con su $p$
    corregido y su tamaño de efecto.

    Se listan las seis y no solo las de la decisión: la de las dos mutaciones
    dentro de SBX es significativa en algunos algoritmos.  Quién gana se resuelve
    en los grupos del caption, no par a par.

    Devuelve el campeón: dentro del grupo ganador se conserva la mutación
    estándar si está."""
    orden = sorted(labels, key=lambda l: -res['medians'][l])
    top = groups[0]
    campeon = next((l for l in top if l.endswith(f'_{MUT_STD}')),
                   max(top, key=lambda l: res['medians'][l]))
    vals = {l: np.asarray(get_values(l, col), dtype=float) for l in labels}
    medianas = ', '.join(f'{_latex_escape(l)} {_num(res["medians"][l], 4)}'
                         for l in orden)

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación de operadores en '
        f'{_latex_escape(DISPLAY.get(alg, alg))} sobre hipervolumen.  Test de '
        f'Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm '
        f'($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de '
        f'pares emparejados, con signo positivo cuando gana el primero del par.  '
        f'Medianas: {medianas}.  Grupos homogéneos, de mejor a peor: '
        f'{fmt_groups(groups)}.  Como el post-hoc no separa a los miembros del '
        f'grupo ganador, se selecciona el de mutación polinomial por ser la '
        f'estándar ({_latex_escape(campeon)}).}}',
        f'\\label{{tab:ops_{alg.lower()}}}',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Par & $p$ (Holm) & $r_{rb}$ & Mejor \\', r'\midrule',
    ]
    for p in res['pairs']:
        a, b = p['a'], p['b']
        r = rank_biserial(vals[a], vals[b])
        # Donde el post-hoc no separa no se declara ganador: la mediana ordena
        # igual, pero llamarlo «mejor» afirmaría una diferencia que no hay.
        mejor = (_latex_escape(a if res['medians'][a] > res['medians'][b] else b)
                 if p['p_holm'] < 0.05 else '---')
        lines.append(
            f"{_latex_escape(a)} vs {_latex_escape(b)} & "
            f"{_fmt_p(p['p_holm'])} & "
            f"{('$+$' if r >= 0 else '$-$') + _num(abs(r), 3)} & "
            f"{mejor} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'operadores_{alg}.tex'))

    print("  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))
    print(f"  campeón: {campeon}")
    for p in res['pairs']:
        r = rank_biserial(vals[p['a']], vals[p['b']])
        print(f"    {p['a']:10s} vs {p['b']:11s} p={p['p_holm']:8.4f}  "
              f"r_rb={r:+.3f}")
    return campeon



def analyze_operators(alg, winners_dir, out_root, decision_col):
    # COMBO_DIRS fija el orden en que los combos aparecen en tablas y leyendas.
    series = build_operator_series_winners(alg, winners_dir, COMBO_DIRS)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    # Frente de referencia común a los 4 combos → IGD+ y ε+ comparables.
    pf_F, pf_df = build_reference_front(series)
    indicator_data = {}
    if pf_F is not None:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicator_data = compute_indicators_per_run(series, pf_F)
        pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = _build_series_value_getter(series, indicator_data)

    # Las salidas de la sección 3.2: tablas de indicadores, frentes por combo y
    # la atribución del frente conjunto (tabla + figura).
    generate_latex_comparison_tables(series, alg, out_dir, get_values)
    # Los frentes de cada combo por separado no se dibujan acá: en un mismo panel
    # se tapan entre sí, y la pregunta de esta etapa —quién sobrevive al juntarlos—
    # la responde mejor el frente conjunto de más abajo.
    for modo in GRID_COLOR_MODES:
        plot_pareto_qed_sa_grid(series, alg, out_dir, color_by=modo)

    if pf_df is not None:
        # La tabla de contribución por combo no se emite en esta etapa: lo que
        # decide está en las dos tablas de operadores de más abajo.  La figura
        # del frente conjunto sí, que es la que muestra el mecanismo.
        plot_frente_conjunto(series, alg, out_dir, pf_df)

    # La comparación de operadores, sobre el indicador de decisión.
    label, higher = dict((c, (l, h)) for c, l, h in OP_INDICATORS)[decision_col]
    res = compare_indicator(get_values, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    groups = homogeneous_groups(res, labels, res['medians'], higher)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)
    print(f"  Friedman ({label}): p = {res['p_omnibus']:.4g}   "
          f"({n_sig}/{len(res['pairs'])} pares significativos)")
    campeon = write_tabla_operadores(res, groups, labels, alg, out_dir,
                                     get_values, decision_col)

    return {'algorithm': alg, 'campeon': campeon,
            'p_friedman': res['p_omnibus'],
            'n_pares_sig': n_sig, 'n_pares': len(res['pairs']),
            'grupos': ' > '.join('{' + ', '.join(g) + '}' for g in groups),
            'mejor_grupo': ', '.join(groups[0]),
            'mediana': round(res['medians'][campeon], 6)}



def etapa2(args):
    algs = args.algorithms or GA_ALGS
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*64}")
    print("  ETAPA 2 — COMPARACIÓN DE OPERADORES")
    print(f"  Datos: {args.winners}")
    print(f"  Decisión por: {args.metric}")
    print(f"{'='*64}")

    rows = []
    for alg in algs:
        r = analyze_operators(alg, args.winners, args.out, args.metric)
        if r:
            rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")
