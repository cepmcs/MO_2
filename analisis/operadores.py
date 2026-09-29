"""
Comparación de los operadores de cruce (PCX y SBX), por algoritmo.

Lee results/exp4/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ y contrasta los dos
cruces con Wilcoxon de rangos con signo, pareado por semilla.  Por algoritmo deja
además las tablas de indicadores, los frentes de cada cruce y el frente conjunto.
"""

import os

import numpy as np
import pandas as pd

from .comun import (
    ALGORITHM_ORDER,
    DISPLAY,
    _build_series_value_getter,
    _fmt_p,
    _latex_escape,
    _num,
    _write_tex,
    compare_indicator,
    generate_latex_comparison_tables,
    rank_biserial,
    series_operadores,
)
from .indicadores import (build_reference_front, compute_indicators_per_run,
                          load_reference_front)
from .figuras import (GRID_COLOR_MODES, plot_frente_conjunto,
                      plot_moleculas_operadores, plot_pareto_qed_sa_grid)


# (columna, etiqueta, mayor_es_mejor).  Por defecto decide el hipervolumen; el
# resto está para poder rehacer la comparación con otro criterio desde --metric.
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



def write_tabla_operadores(res, alg, out_dir, get_values, col, label, higher):
    """La comparación de los dos cruces: el $p$ del Wilcoxon pareado y su tamaño de
    efecto.

    Devuelve el cruce ganador, o None si el test no los separa."""
    p = res['pairs'][0]
    a, b = p['a'], p['b']
    va = np.asarray(get_values(a, col), dtype=float)
    vb = np.asarray(get_values(b, col), dtype=float)
    med = res['medians']

    gana_a = (med[a] > med[b]) == higher
    mejor = (a if gana_a else b) if p['p_holm'] < 0.05 else None
    signo = 1 if higher else -1
    r = signo * rank_biserial(va, vb)
    medianas = ', '.join(f'{_latex_escape(l)} {_num(med[l], 4)}' for l in (a, b))

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación de los operadores de cruce en '
        f'{_latex_escape(DISPLAY.get(alg, alg))} sobre el indicador {label}.  '
        f'Wilcoxon de rangos con signo con las {min(len(va), len(vb))} semillas '
        f'pareadas ($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de pares emparejados, con signo positivo cuando gana el '
        f'primero del par.  Donde el test no separa no se declara ganador.  '
        f'Medianas: {medianas}.}}',
        f'\\label{{tab:ops_{alg.lower()}}}',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Par & $p$ & $r_{rb}$ & Mejor \\', r'\midrule',
        f"{_latex_escape(a)} vs {_latex_escape(b)} & {_fmt_p(p['p_holm'])} & "
        f"{('$+$' if r >= 0 else '$-$') + _num(abs(r), 3)} & "
        f"{_latex_escape(mejor) if mejor else '---'} \\\\",
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ]
    _write_tex(lines, os.path.join(out_dir, f'operadores_{alg}.tex'))

    print(f"  {a} vs {b}: p = {p['p_holm']:.4f}   r_rb = {r:+.3f}   "
          f"mejor: {mejor or '---'}")
    return mejor



def analyze_operators(alg, results, out_root, decision_col, frente):
    series = series_operadores(results, alg)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} cruce(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   cruces: {', '.join(labels)}\n{'─'*64}")

    pf_F, pf_df = frente
    print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
    indicator_data = compute_indicators_per_run(series, pf_F)
    pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = _build_series_value_getter(series, indicator_data)

    generate_latex_comparison_tables(series, alg, out_dir, get_values)
    for modo in GRID_COLOR_MODES:
        plot_pareto_qed_sa_grid(series, alg, out_dir, color_by=modo)
    plot_moleculas_operadores(series, alg, out_dir)

    _, pf_propio = build_reference_front(series)
    if pf_propio is not None:
        plot_frente_conjunto(series, alg, out_dir, pf_propio)

    label, higher = dict((c, (l, h)) for c, l, h in OP_INDICATORS)[decision_col]
    res = compare_indicator(get_values, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    mejor = write_tabla_operadores(res, alg, out_dir, get_values, decision_col,
                                   label, higher)
    return {'algorithm': alg, 'indicador': decision_col,
            'mejor': mejor or 'ninguno', 'p': res['pairs'][0]['p_holm'],
            **{f'mediana_{l}': round(res['medians'][l], 6) for l in labels}}



def operadores(args):
    algs = args.algorithms or ALGORITHM_ORDER
    os.makedirs(args.out, exist_ok=True)
    frente = load_reference_front(args.results)

    print(f"\n{'='*64}")
    print("  COMPARACIÓN DE OPERADORES DE CRUCE")
    print(f"  Datos: {args.results}")
    print(f"  Indicador: {args.metric}")
    print(f"{'='*64}")

    rows = []
    for alg in algs:
        r = analyze_operators(alg, args.results, args.out, args.metric, frente)
        if r:
            rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")
