"""
Tests estadísticos y tablas LaTeX.

Las 20 semillas están pareadas en todas las etapas (mismo run_id → misma
población inicial), así que los tests toman la semilla como bloque: Friedman
sobre los métodos y, si resulta significativo, las comparaciones por pares con
Wilcoxon de rangos con signo corregidas por Holm.  El resultado se resume en
grupos homogéneos: dentro de una llave el post-hoc no separa, entre llaves sí.
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import datos


# ─── Formato LaTeX ───────────────────────────────────────────────────────────

def latex_escape(s):
    """Escapa los caracteres especiales de LaTeX (pcx_gauss → pcx\\_gauss)."""
    repl = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            '×': r'$\times$'}
    return ''.join(repl.get(c, c) for c in str(s))


def fmt_p(p):
    if p is None or pd.isna(p):
        return '---'
    return r'$<$0.001' if p < 1e-3 else f'{p:.3f}'


def fmt_groups(grupos):
    """'{A, B} $>$ {C}' con los nombres de presentación."""
    return ' $>$ '.join('\\{' + ', '.join(datos.display(x) for x in g) + '\\}'
                        for g in grupos)


def _factor_label(f):
    """FACTOR_LABELS ya trae matemática ($w$, $c_1$); solo falta el '×'."""
    return datos.FACTOR_LABELS.get(f, f).replace('×', r'$\times$')


def write_tex(lines, path, msg=None):
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ {msg or os.path.basename(path)}")


def _tabla(caption, tex_label, col_spec, header, filas):
    """Envoltorio común de las tablas: entorno, caption, cabecera y filas."""
    return ([r'\begin{table}[htbp]', r'\centering',
             f'\\caption{{{caption}}}', f'\\label{{{tex_label}}}',
             f'\\begin{{tabular}}{{{col_spec}}}', r'\toprule',
             header + r' \\', r'\midrule']
            + filas
            + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


# ─── Tests ───────────────────────────────────────────────────────────────────

def holm(pvals):
    """Corrección de Holm-Bonferroni; devuelve los p ajustados en el orden de
    entrada."""
    n = len(pvals)
    adj = np.empty(n, dtype=float)
    prev = 0.0
    for rango, idx in enumerate(np.argsort(pvals)):
        prev = max(prev, min((n - rango) * pvals[idx], 1.0))   # monotonía
        adj[idx] = prev
    return adj


def holm_nan(pvals):
    """Holm sobre los p que existen; las celdas sin datos vuelven como NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = holm(p[ok])
    return out


def compare_indicator(get_values, labels, col):
    """Friedman con la semilla como bloque más el post-hoc por pares.

    Devuelve {p_omnibus, medians, pairs} o None si falta algún método o hay
    menos de 3 semillas."""
    cols = {}
    for lab in labels:
        v = get_values(lab, col)
        if v is None:
            return None
        cols[lab] = np.asarray(v, dtype=float)

    n = min(len(v) for v in cols.values())
    if n < 3:
        return None
    M = np.column_stack([cols[lab][:n] for lab in labels])   # semillas × métodos
    k = M.shape[1]

    try:
        p_omni = float(stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue)
    except ValueError:
        p_omni = np.nan

    pares, crudos = [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            p = float(stats.wilcoxon(M[:, i], M[:, j]).pvalue)
        except ValueError:      # todas las diferencias son cero
            p = 1.0
        pares.append((labels[i], labels[j]))
        crudos.append(p)
    ajustados = holm(crudos) if crudos else []

    return {'p_omnibus': p_omni,
            'medians': {lab: float(np.median(cols[lab][:n])) for lab in labels},
            'pairs': [{'a': a, 'b': b, 'p_raw': pr, 'p_holm': pa}
                      for (a, b), pr, pa in zip(pares, crudos, ajustados)]}


def homogeneous_groups(res, labels, medians, higher_better):
    """Grupos que el post-hoc no logra separar, de mejor a peor."""
    orden = sorted(labels, key=lambda l: -medians[l] if higher_better else medians[l])
    separa = {(p['a'], p['b']): p['p_holm'] < 0.05 for p in res['pairs']}

    def difieren(a, b):
        return separa.get((a, b), separa.get((b, a), False))

    grupos, actual = [], [orden[0]]
    for lab in orden[1:]:
        if any(difieren(lab, m) for m in actual):
            grupos.append(actual)
            actual = [lab]
        else:
            actual.append(lab)
    grupos.append(actual)
    return grupos


def friedman_by_factor(g, factor, metric):
    """Efecto marginal de un hiperparámetro con la semilla como bloque.

    Friedman sobre (semillas × niveles), donde cada celda es la mediana marginal
    del nivel en esa semilla.  Con 2 niveles Friedman degenera y se usa Wilcoxon.
    Devuelve (delta, W, p): delta es la diferencia entre el mejor y el peor nivel,
    y W es la de Kendall, W = χ²/(m(k−1)), la concordancia del orden entre las m
    semillas (0 = ninguna, 1 = las m ordenan igual)."""
    B = g.pivot_table(index='run', columns=factor, values=metric,
                      aggfunc='median').dropna()
    m, k = B.shape
    if m < 3 or k < 2:
        return np.nan, np.nan, np.nan

    medianas = B.median(axis=0)
    delta = float(medianas.max() - medianas.min())

    R = np.apply_along_axis(stats.rankdata, 1, B.values)
    chi2 = 12 / (m * k * (k + 1)) * np.sum(R.sum(axis=0) ** 2) - 3 * m * (k + 1)
    W = float(max(chi2 / (m * (k - 1)), 0.0))

    cols = [B[c].values for c in B.columns]
    try:
        p = (stats.wilcoxon(cols[0], cols[1]).pvalue if k == 2
             else stats.friedmanchisquare(*cols).pvalue)
    except ValueError:
        return delta, W, np.nan
    return delta, W, float(p)


# ─── Etapa 1: sensibilidad y selección ───────────────────────────────────────

def _fmt_delta(delta, p_holm):
    if pd.isna(delta):
        return '---'
    return f'{delta:.4f}' + ('*' if not pd.isna(p_holm) and p_holm < 0.05 else '')


def write_effects_table(effects, alg, factors, out_dir, metric):
    """Sensibilidad a cada hiperparámetro.

    Δ es el recorrido de la métrica entre el mejor y el peor nivel, en sus
    propias unidades: se lee contra el eje de main_effects sin escala auxiliar.
    Con combos la tabla es una grilla hiperparámetros × combos, porque agrupar
    los combos produce un orden espurio (pcx y sbx viven en regímenes de
    hipervolumen distintos y la mediana agrupada salta entre ellos)."""
    label, _ = datos.HP_METRICS[metric]
    por_combo = any(isinstance(v, dict) for v in effects.values())
    combos = ([c for c in datos.HP_COMBOS
               if any(c in v for v in effects.values())] if por_combo else [])

    # Una tabla son 12 tests (3 factores × 4 combos) o 4 en MOPSO; sin corregir,
    # un p de 0.04 es lo esperable por azar.  Holm sobre la tabla completa, el
    # mismo estándar que el post-hoc de las etapas 2-4.
    if por_combo:
        keys = [(f, c) for f in factors for c in combos]
        crudos = [effects[f].get(c, (np.nan,) * 3)[2] for f, c in keys]
    else:
        keys = [(f, None) for f in factors]
        crudos = [effects.get(f, (np.nan,) * 3)[2] for f, _ in keys]
    p_holm = dict(zip(keys, holm_nan(crudos)))
    n_tests = int(np.sum(~np.isnan(np.asarray(crudos, dtype=float))))

    # La grilla marca la significancia con asterisco; la tabla sin combos tiene
    # columna de p propia.
    sig = (f'* indica $p < 0.05$ en el test de Friedman con las 20 semillas '
           f'como bloques, tras corregir por Holm las {n_tests} comparaciones '
           f'de la tabla.' if por_combo else
           f'$p$: test de Friedman con las 20 semillas como bloques, corregido '
           f'por Holm sobre las {n_tests} comparaciones de la tabla.')
    caption = (f'Sensibilidad de {latex_escape(label)} a cada hiperparámetro '
               f'de {latex_escape(alg)}.  $\\Delta$: diferencia entre el mejor '
               f'y el peor nivel, en unidades de {latex_escape(label).lower()}.  '
               f'{sig}')
    if por_combo:
        caption += ('  Cada columna es una combinación de operadores, evaluada '
                    'por separado: agrupar los combos produce un orden espurio, '
                    'porque los operadores de cruce operan en regímenes de '
                    'hipervolumen distintos.')
        col_spec = 'l' + 'c' * len(combos)
        header = 'Hiperparámetro & ' + ' & '.join(latex_escape(c) for c in combos)
        filas = [
            f'{_factor_label(f)} & '
            + ' & '.join(_fmt_delta(effects[f].get(c, (np.nan,) * 3)[0],
                                    p_holm[(f, c)]) for c in combos)
            + r' \\'
            for f in factors]
    else:
        col_spec = 'lcc'
        header = r'Hiperparámetro & $\Delta$ & $p$ (Holm)'
        filas = []
        for f in sorted(factors,
                        key=lambda x: -np.nan_to_num(effects.get(x, (0, 0, 1))[0])):
            delta = effects.get(f, (np.nan,) * 3)[0]
            filas.append(f'{_factor_label(f)} & {delta:.4f} & '
                         f'{fmt_p(p_holm[(f, None)])} \\\\')

    write_tex(_tabla(caption, f'tab:hp_effects_{alg.lower()}', col_spec, header,
                     filas),
              os.path.join(out_dir, f'effects_{alg}.tex'))


def write_selection_summary(per_alg, metric, out_dir):
    """CSV y tabla de las configuraciones seleccionadas.  El CSV usa los nombres
    de columna que consume run_experiments.py."""
    label, _ = datos.HP_METRICS[metric]
    filas = []
    for alg in [a for a in datos.ALGORITHM_ORDER if a in per_alg]:
        d = per_alg[alg]
        for nombre, b in d['blocks'].items():
            niveles = dict(zip(d['sub_factors'], b['cfg']))
            if d['por_combo'] and nombre:
                niveles['crossover'], niveles['mutation'] = nombre.split('/')
            pop, gen = str(niveles['budget']).split('×')
            fila = {'algorithm': alg, 'operators': nombre or '',
                    'pop_size': int(pop), 'n_gen': int(gen),
                    'config': b['label'], 'avg_rank': b['rank'],
                    'mean': b['mean'], 'std': b['std'],
                    'n_configs': b['n_configs']}
            fila.update({f: niveles[f] for f in
                         ('crossover', 'mutation', 'cx_prob', 'mut_prob',
                          'w', 'c1', 'c2') if f in niveles})
            filas.append(fila)

    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(out_dir, 'selected_configs.csv'), index=False)
    print(f"  ✓ selected_configs.csv  ({len(out)} configuraciones)")

    lines, prev = [], None
    for _, r in out.iterrows():
        celda = '' if r['algorithm'] == prev else latex_escape(r['algorithm'])
        if celda and prev is not None:
            lines.append(r'\midrule')
        prev = r['algorithm']
        lines.append(
            f"{celda} & {latex_escape(r['operators']) or '---'} & "
            f"{latex_escape(r['config'])} & {r['avg_rank']:.2f} & "
            f"{r['mean']:.4f} $\\pm$ {r['std']:.4f} \\\\")

    tabla = _tabla(
        f'Configuraciones seleccionadas: la mejor de cada combinación de '
        f'operadores en los algoritmos genéticos, y la mejor global en MOPSO. '
        f'Gana la de menor rango medio de {latex_escape(label)}, rankeando las '
        f'configuraciones dentro de cada una de las 20 semillas.',
        'tab:hp_seleccionadas', 'lllcc',
        r'Algoritmo & Operadores & Configuración & Rango medio & $\mu \pm \sigma$',
        lines)
    tabla.insert(2, r'\small')
    write_tex(tabla, os.path.join(out_dir, 'selected_configs.tex'))


# ─── Tablas de comparación (etapas 2 y 3) ────────────────────────────────────

COMPARISON_MO = [
    ('Hipervolumen',     'hypervolume', '.4f', True),
    ('Espaciamiento',    'spacing',     '.4f', False),
    ('IGD$^+$',          'igd_plus',    '.4f', False),
    (r'$\epsilon^+$',    'epsilon',     '.4f', False),
    ('Tamaño de Pareto', 'n_pareto',    '.1f', True),
    ('Tiempo (s)',       'time_sec',    '.1f', False),
]
COMPARISON_CHEM = [
    ('QED',      'mean_qed',      '.4f', True),
    ('SA',       'mean_sa',       '.2f', False),
    ('Lipinski', 'mean_lipinski', '.4f', True),
    ('Validez',  'validity',      '.4f', True),
    ('Unicidad', 'uniqueness',    '.4f', True),
    ('Novedad',  'novelty',       '.4f', True),
]


def _comparison_table(series, get_values, cfg, caption, tex_label, out_dir, fname):
    """Series en filas, métricas en columnas, media ± desvío; el mejor de cada
    columna en negrita."""
    cols = [c for c in cfg
            if any(get_values(s.label, c[1]) is not None for s in series)]
    if not cols:
        print(f"  ⚠ {fname}: sin métricas con datos")
        return

    medias, desvios = {}, {}
    for s in series:
        for _, col, _, _ in cols:
            vals = get_values(s.label, col)
            if vals is not None and len(vals):
                medias[(s.label, col)] = float(np.mean(vals))
                desvios[(s.label, col)] = (float(np.std(vals, ddof=1))
                                           if len(vals) > 1 else 0.0)

    mejor = {}
    for _, col, _, higher in cols:
        candidatos = [(s.label, medias[(s.label, col)]) for s in series
                      if (s.label, col) in medias]
        if candidatos:
            mejor[col] = (max if higher else min)(candidatos, key=lambda t: t[1])[0]

    flecha = lambda h: r'$\uparrow$' if h else r'$\downarrow$'
    filas = []
    for s in series:
        celdas = [latex_escape(s.label)]
        for _, col, fmt, _ in cols:
            if (s.label, col) not in medias:
                celdas.append('--')
                continue
            cuerpo = f'{medias[(s.label, col)]:{fmt}} \\pm {desvios[(s.label, col)]:{fmt}}'
            celdas.append(f'$\\mathbf{{{cuerpo}}}$' if mejor.get(col) == s.label
                          else f'${cuerpo}$')
        filas.append(' & '.join(celdas) + r' \\')

    write_tex(_tabla(caption, tex_label, 'l' + 'c' * len(cols),
                     ' & '.join(['Algoritmo']
                                + [f'{h} {flecha(hb)}' for h, _, _, hb in cols]),
                     filas),
              os.path.join(out_dir, fname))


def comparison_tables(series, out_dir, get_values, contexto=None):
    """Las dos tablas descriptivas: indicadores multiobjetivo y químicos.
    contexto (p. ej. el algoritmo en la etapa 2) desambigua caption y \\label."""
    if len(series) < 2:
        return
    cap = f' — {contexto}' if contexto else ''
    lab = f'_{contexto.lower()}' if contexto else ''
    _comparison_table(series, get_values, COMPARISON_MO,
                      f'Comparación de indicadores multiobjetivo{cap}',
                      f'tab:comparison_multiobjective{lab}',
                      out_dir, 'comparison_multiobjective.tex')
    _comparison_table(series, get_values, COMPARISON_CHEM,
                      f'Comparación de indicadores químicos '
                      f'(media del frente final){cap}',
                      f'tab:comparison_chemical{lab}',
                      out_dir, 'comparison_chemical.tex')


def summary_csv(series, out_dir, get_values):
    """Resumen numérico (media y desvío por serie) de las métricas per-run."""
    cols = ['hypervolume', 'spacing', 'validity', 'novelty', 'igd_plus',
            'epsilon', 'best_qed', 'best_sa', 'n_pareto', 'time_sec']
    filas = []
    for s in series:
        fila = {'series': s.label}
        for col in cols:
            vals = get_values(s.label, col)
            if vals is None or not len(vals):
                continue
            fila['n_runs'] = len(vals)
            fila[f'{col}_mean'] = float(np.mean(vals))
            fila[f'{col}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        filas.append(fila)
    pd.DataFrame(filas).to_csv(os.path.join(out_dir, 'comparison_summary.csv'),
                               index=False)
    print("  ✓ comparison_summary.csv")


# ─── Tablas de tests (etapas 2, 3 y 4) ───────────────────────────────────────

def _filas_pares(res, nombre=lambda x: x):
    return [f"{nombre(p['a'])} vs {nombre(p['b'])} & {fmt_p(p['p_holm'])} & "
            f"{'sí' if p['p_holm'] < 0.05 else 'no'} \\\\"
            for p in res['pairs']]


def write_tests_table(res, alg, out_dir, label):
    """Etapa 2: las 6 comparaciones por pares sobre el indicador de decisión.
    La magnitud del efecto está en las tablas de comparación."""
    write_tex(
        _tabla(f'Comparación de operadores en {latex_escape(alg)} sobre '
               f'{label.lower()}.  Test de Friedman con las 20 semillas como '
               f'bloques ($p$ = {fmt_p(res["p_omnibus"])}), seguido de las '
               f'comparaciones por pares con Wilcoxon de rangos con signo y '
               f'corrección de Holm.',
               f'tab:ops_tests_{alg.lower()}', 'lcc',
               r'Par & $p$ (Holm) & Significativo',
               _filas_pares(res, latex_escape)),
        os.path.join(out_dir, f'tests_{alg}.tex'))


def write_groups_table(res, grupos, out_dir, metric_label):
    """Etapa 3: las 10 comparaciones entre algoritmos, resumidas en grupos."""
    path = os.path.join(out_dir, 'grupos_homogeneos.tex')
    print()
    write_tex(
        _tabla(f'Comparación entre algoritmos sobre {metric_label}.  Test de '
               f'Friedman con las 20 semillas como bloques '
               f'($p$ = {fmt_p(res["p_omnibus"])}), seguido de las comparaciones '
               f'por pares con Wilcoxon de rangos con signo y corrección de '
               f'Holm.  Grupos homogéneos, de mejor a peor: {fmt_groups(grupos)}.',
               'tab:comparacion_grupos', 'lcc',
               r'Par & $p$ (Holm) & Significativo',
               _filas_pares(res, datos.display)),
        path, msg=path)


def write_baseline_tables(res, grupos, medianas, desvios, labels, metric_label,
                          out_dir):
    """Etapa 4: una tabla descriptiva y otra de cada MOEA contra cada baseline."""
    n_moea = sum(1 for l in labels if l not in datos.BASELINE_KEYS)

    filas = []
    for i, lab in enumerate(labels):
        if i == n_moea:
            filas.append(r'\midrule')
        tipo = ''
        if i == 0:
            tipo = r'\multirow{%d}{*}{MOEA}' % n_moea
        elif i == n_moea:
            tipo = r'\multirow{%d}{*}{Baseline}' % (len(labels) - n_moea)
        filas.append(f"{tipo} & {datos.display(lab)} & "
                     f"${medianas[lab]:.4f} \\pm {desvios[lab]:.4f}$ \\\\")
    write_tex(
        _tabla(f'Algoritmos multiobjetivo frente a las baselines sobre '
               f'{metric_label}, con idéntico presupuesto de 100.000 '
               f'evaluaciones y las mismas 20 semillas.  Mediana y desvío '
               f'estándar entre ejecuciones.',
               'tab:baselines_desc', 'llc',
               '& Método & ' + metric_label.capitalize(), filas),
        os.path.join(out_dir, 'comparacion.tex'))

    moeas = [l for l in labels if l not in datos.BASELINE_KEYS]
    bases = [l for l in labels if l in datos.BASELINE_KEYS]
    p_por_par = {}
    for p in res['pairs']:
        p_por_par[(p['a'], p['b'])] = p['p_holm']
        p_por_par[(p['b'], p['a'])] = p['p_holm']

    filas = [f'{datos.display(m)} & '
             + ' & '.join(fmt_p(p_por_par.get((m, b))) for b in bases) + r' \\'
             for m in moeas]
    write_tex(
        _tabla(f'Comparaciones por pares entre cada algoritmo multiobjetivo y '
               f'cada baseline sobre {metric_label}.  Test de Friedman global '
               f'($p$ = {fmt_p(res["p_omnibus"])}) y Wilcoxon de rangos con '
               f'signo por pares con corrección de Holm; se indica el $p$ '
               f'ajustado.  Grupos homogéneos: {fmt_groups(grupos)}.',
               'tab:baselines_tests', 'l' + 'c' * len(bases),
               'Algoritmo & ' + ' & '.join(datos.display(b) for b in bases),
               filas),
        os.path.join(out_dir, 'grupos_homogeneos.tex'))


def write_pairs_csv(res, out_dir):
    """Los pares con su p crudo y ajustado, para revisar fuera del documento."""
    path = os.path.join(out_dir, 'tests_pares.csv')
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(path, index=False)
    print(f"  ✓ {path}")
