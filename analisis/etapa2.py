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
    DISPLAY_ALG,
    GA_ALGS,
    _build_series_value_getter,
    _fmt_p,
    _latex_escape,
    _num,
    _num_es,
    _write_tex,
    build_operator_series_winners,
    compare_indicator,
    fmt_groups,
    homogeneous_groups,
    rank_biserial,
)
from .indicadores import (_atribucion_por_origen, _familia,
                          build_reference_front, compute_indicators_per_run,
                          load_reference_front)
from .figuras import (plot_frente_conjunto_multi, plot_moleculas_operadores,
                      plot_pareto_qed_sa_grid)


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
    medianas = ', '.join(f'{_latex_escape(l)} {_num(res["medians"][l], 3)}'
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


# ─── Tablas conjuntas de indicadores ─────────────────────────────────────────
#
#   Una tabla por tipo de indicador con los cuatro combos de cada algoritmo, en
#   vez de una por algoritmo.  Conserva los \label por algoritmo para no romper
#   las referencias ya escritas en el documento.

# Orden de los algoritmos en las tablas conjuntas.
ORDEN_TABLAS = ['AGEMOEA', 'MOEAD', 'NSGA2', 'NSGA3']

# (encabezado, columna, formato, mayor_es_mejor).  mayor_es_mejor=None: se
# reporta sin competir (Fsp3 es la restricción, no un objetivo), sin flecha ni
# negrita.
TABLA_MO = [
    ('Hipervolumen',     'hypervolume', '.3f', True),
    ('Espaciamiento',    'spacing',     '.3f', False),
    (r'$\text{IGD}^+$',  'igd_plus',    '.3f', False),
    (r'$\epsilon^+$',    'epsilon',     '.3f', False),
    ('Tamaño de Pareto', 'n_pareto',    '.1f', True),
    ('Tiempo (s)',       'time_sec',    '.1f', False),
]

TABLA_QUIM = [
    ('QED',                'mean_qed',    '.3f', True),
    ('SA',                 'mean_sa',     '.3f', False),
    ('Factibilidad',       'feasibility', '.3f', True),
    (r'$F_{\text{sp3}}$',  'mean_fsp3',   '.3f', None),
    ('Validez',            'validity',    '.3f', True),
    ('Unicidad',           'uniqueness',  '.3f', True),
    ('Novedad',            'novelty',     '.3f', True),
]


def _tabla_conjunta(acumulado, cfg, caption, tag, fname, out_dir):
    """Tabla LaTeX con una fila por combo y un bloque por algoritmo (media ± std
    sobre las semillas).  La negrita marca el mejor combo de cada columna DENTRO
    de su algoritmo: lo que se compara acá son operadores, no algoritmos."""
    algs = [a for a in ORDEN_TABLAS if a in acumulado]
    algs += [a for a in acumulado if a not in algs]
    if not algs:
        return

    flecha = lambda hb: '' if hb is None else (r' $\uparrow$' if hb
                                               else r' $\downarrow$')
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{tab:{tag}_operadores}}%',
    ]
    lines += [f'\\label{{tab:{tag}_{a.lower()}}}' + ('%' if i < len(algs) - 1 else '')
              for i, a in enumerate(algs)]
    lines += [
        r'\resizebox{\textwidth}{!}{%',
        f'\\begin{{tabular}}{{ll{"c" * len(cfg)}}}',
        r'\toprule',
        'Algoritmo & Operadores & '
        + ' & '.join(f'{h}{flecha(hb)}' for h, _, _, hb in cfg) + r' \\',
        r'\midrule',
    ]

    for k, alg in enumerate(algs):
        series, get = acumulado[alg]['series'], acumulado[alg]['get']
        media, desv = {}, {}
        for s in series:
            for _, col, _, _ in cfg:
                v = get(s.label, col)
                if v is not None and len(v):
                    media[(s.label, col)] = float(np.mean(v))
                    desv[(s.label, col)] = (float(np.std(v, ddof=1))
                                            if len(v) > 1 else 0.0)
        mejor = {}
        for _, col, _, hb in cfg:
            cand = [(s.label, media[(s.label, col)]) for s in series
                    if (s.label, col) in media]
            if hb is not None and cand:
                mejor[col] = (max if hb else min)(cand, key=lambda t: t[1])[0]

        lines.append(f'\\multirow{{{len(series)}}}{{*}}{{\\acrshort{{{alg.lower()}}}}} ')
        for s in series:
            celdas = []
            for _, col, fmt, _ in cfg:
                if (s.label, col) not in media:
                    celdas.append('--')
                    continue
                cuerpo = (f'{_num_es(media[(s.label, col)], fmt)} \\pm '
                          f'{_num_es(desv[(s.label, col)], fmt)}')
                celdas.append(f'$\\mathbf{{{cuerpo}}}$'
                              if mejor.get(col) == s.label else f'${cuerpo}$')
            lines.append(f' & {_latex_escape(s.label)} & '
                         + ' & '.join(celdas) + r' \\')
        lines.append(r'\midrule' if k < len(algs) - 1 else r'\bottomrule')

    lines += [r'\end{tabular}%', '}', r'\end{table}']
    with open(os.path.join(out_dir, fname), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {fname}")


def escribir_tablas_conjuntas(acumulado, out_dir):
    """Las dos tablas conjuntas (multiobjetivo y químicos) de todos los
    algoritmos analizados, en out_dir."""
    if not acumulado:
        return
    sufijo = ('para las distintas combinaciones de operadores de cruce y '
              'mutación (media $\\pm$ desviación estándar sobre 20 semillas '
              'independientes).')
    _tabla_conjunta(
        acumulado, TABLA_MO,
        f'Comparación de indicadores multiobjetivo {sufijo}',
        'comparison_multiobjective', 'comparison_multiobjective_operadores.tex',
        out_dir)
    _tabla_conjunta(
        acumulado, TABLA_QUIM,
        f'Comparación de indicadores químicos {sufijo}',
        'comparison_chemical', 'comparison_chemical_operadores.tex', out_dir)



def escribir_frente_conjunto(acumulado, out_dir):
    """Una sola figura con el frente conjunto de cada algoritmo (PCX vs. SBX),
    en lugar de una por algoritmo.  La tabla de contribución por combo no se
    emite en esta etapa: lo que decide está en las tablas de operadores; la
    figura es la que muestra el mecanismo."""
    algs = [a for a in ORDEN_TABLAS if a in acumulado]
    algs += [a for a in acumulado if a not in algs]
    paneles = [(DISPLAY_ALG.get(a, a), acumulado[a].get('atr')) for a in algs]
    plot_frente_conjunto_multi(paneles, out_dir, 'frente_conjunto_operadores.pdf')


def analyze_operators(alg, winners_dir, out_root, decision_col, acumulado=None):
    # COMBO_DIRS fija el orden en que los combos aparecen en tablas y leyendas.
    series = build_operator_series_winners(alg, winners_dir, COMBO_DIRS)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    # IGD+ y ε+ contra el frente común de todo el documento.
    pf_F, pf_df = load_reference_front()
    indicator_data = {}
    if pf_F is not None:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicator_data = compute_indicators_per_run(series, pf_F)
        pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = _build_series_value_getter(series, indicator_data)

    # Las tablas de indicadores no se escriben por algoritmo: se juntan en una
    # sola por tipo (ver escribir_tablas_conjuntas), que etapa2 arma al final.
    if acumulado is not None:
        acumulado[alg] = {'series': series, 'get': get_values}

    # Las demás salidas de la sección 3.2: frentes por combo y la atribución del
    # frente conjunto (figura).
    # Los frentes de cada combo por separado no se dibujan acá: en un mismo panel
    # se tapan entre sí, y la pregunta de esta etapa —quién sobrevive al juntarlos—
    # la responde mejor el frente conjunto de más abajo.
    # Solo la variante por nº de ejecuciones: la de Fsp3 no se usa en esta etapa.
    plot_pareto_qed_sa_grid(series, alg, out_dir, color_by='nruns')
    # Qué química pone cada combo a lo largo del frente.  Las tablas comparan
    # números sobre el frente entero; esto muestra las estructuras y dónde deja
    # de haberlas.
    plot_moleculas_operadores(series, alg, out_dir)

    # El frente conjunto cuenta qué combo aportó cada molécula, así que va con el
    # frente de sus propias series y no con el común de IGD+.
    # No se dibuja acá: la figura junta los cuatro algoritmos (ver
    # escribir_frente_conjunto), así que solo se deja lista su atribución.
    _, pf_propio = build_reference_front(series)
    if pf_propio is not None and acumulado is not None:
        acumulado[alg]['atr'] = _atribucion_por_origen(series, pf_propio, _familia)

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
    acumulado = {}
    for alg in algs:
        r = analyze_operators(alg, args.winners, args.out, args.metric, acumulado)
        if r:
            rows.append(r)

    escribir_tablas_conjuntas(acumulado, args.out)
    escribir_frente_conjunto(acumulado, args.out)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")
