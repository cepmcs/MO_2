"""
Análisis de los experimentos: las cuatro etapas de la comparación, más la figura
de moléculas representativas.  Cada etapa lee lo que produjo la anterior y deja
sus tablas y figuras bajo plots/.

  etapa1     resultados/grid/all_metrics.csv          →  plots/hiperparametros/
  etapa2     resultados/winners/                      →  plots/operadores/
  etapa3     resultados/finalistas/                   →  plots/comparacion_final/
  etapa4     resultados/finalistas/ + .../baselines/  →  plots/baselines/
  moleculas  resultados/finalistas/                   →  plots/comparacion_final/

Las 20 semillas están pareadas en todas las etapas (mismo run_id → misma
población inicial), así que los tests toman la semilla como bloque: Friedman
sobre los métodos y, si resulta significativo, las comparaciones por pares con
Wilcoxon de rangos con signo corregidas por Holm.  El resultado se resume en
grupos homogéneos: dentro de una llave el post-hoc no separa, entre llaves sí.

Uso:
    python analisis.py etapa1 [--algorithms NSGA2 MOPSO] [--metric spacing]
    python analisis.py etapa2 [--algorithms NSGA2 MOEAD] [--metric igd_plus]
    python analisis.py etapa3 [--finalistas otra_carpeta]
    python analisis.py etapa4 [--metric igd_plus]
    python analisis.py moleculas [--out figura.png]

La carga de resultados y las gráficas comunes viven en plot_comparison.py.
"""

import io
import os
import math
import glob
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Importar plot_comparison también fija el estilo global de matplotlib.
import plot_comparison as pc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Entradas de cada etapa, todas bajo resultados/.  Los experimentos escriben en
# el cluster a results/ y results_baselines/ (ver utils_mo.RESULTS_DIR y
# experimento_baselines.BASELINE_RESULTS_DIR); lo que baja al PC vive acá:
#
#   grid/        copia ligera de results/ del cluster (sin convergence.csv ni
#                all_molecules.csv.gz), con el all_metrics.csv del grid completo
#   winners/     las 17 configuraciones que ganaron su bloque en la etapa 1
#   finalistas/  symlinks relativos a la ganadora de cada algoritmo en winners/
#   baselines/   copia de results_baselines/ del cluster
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
METRICS_CSV    = os.path.join(RESULTADOS_DIR, "grid", "all_metrics.csv")
WINNERS_DIR    = os.path.join(RESULTADOS_DIR, "winners")
FINALISTAS_DIR = os.path.join(RESULTADOS_DIR, "finalistas")
BASELINES_DIR  = os.path.join(RESULTADOS_DIR, "baselines")

# Salidas, todas bajo plots/.  El componente 'operadores' de la etapa 2 es
# significativo: plot_comparison deduce de él el nombre del algoritmo para
# titular las figuras de frentes.
PLOTS_DIR      = os.path.join(ROOT_DIR, "plots")
OUT_HP         = os.path.join(PLOTS_DIR, "hiperparametros")
OUT_OPERADORES = os.path.join(PLOTS_DIR, "operadores")
OUT_ALGORITMOS = os.path.join(PLOTS_DIR, "comparacion_final")
OUT_BASELINES  = os.path.join(PLOTS_DIR, "baselines")

# Nombres para el documento (los directorios usan la forma corta).
DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO',
           'RANDOM': 'Aleatorio', 'LHS': 'LHS', 'WEIGHTED_GA': 'GA ponderado',
           'SCREENING': 'Cribado MOSES', 'HILL_CLIMBER': 'Escalador'}


# ═══════════════════════════════════════════════════════════════════════════
#   Utilidades compartidas
# ═══════════════════════════════════════════════════════════════════════════

def _latex_escape(s):
    """Escapa caracteres especiales de LaTeX en texto (p. ej. el guion bajo de
    nombres de operadores como pcx_gauss → pcx\\_gauss)."""
    repl = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            '×': r'$\times$'}
    return ''.join(repl.get(c, c) for c in str(s))


def _fmt_p(p):
    if p is None or pd.isna(p):
        return '---'
    return r'$<$0.001' if p < 1e-3 else f'{p:.3f}'


def fmt_groups(groups):
    """'{A, B} $>$ {C}' con los nombres de presentación."""
    return ' $>$ '.join('\\{' + ', '.join(DISPLAY.get(x, x) for x in g) + '\\}'
                        for g in groups)


def holm(pvals):
    """Corrección de Holm-Bonferroni.  Devuelve los p ajustados en el orden de
    entrada."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        prev = max(prev, min(val, 1.0))    # monotonía
        adj[idx] = prev
    return adj


def compare_indicator(get_values, labels, col):
    """Friedman sobre los métodos con la semilla como bloque + post-hoc.

    Devuelve dict con el p de Friedman, el tamaño de efecto W de Kendall, y la
    lista de comparaciones por pares con su p corregido por Holm."""
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

    m, k = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M)
    chi2 = 12 / (m * k * (k + 1)) * np.sum(R.sum(axis=0) ** 2) - 3 * m * (k + 1)
    W = float(max(chi2 / (m * (k - 1)), 0.0))
    try:
        p_omni = float(stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue)
    except ValueError:
        p_omni = np.nan

    pairs, raw = [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            p = float(stats.wilcoxon(M[:, i], M[:, j]).pvalue)
        except ValueError:      # todas las diferencias son cero
            p = 1.0
        pairs.append((labels[i], labels[j]))
        raw.append(p)
    adj = holm(raw) if raw else []

    return {'p_omnibus': p_omni, 'W': W, 'n_blocks': m,
            'medians': {lab: float(np.median(cols[lab][:n])) for lab in labels},
            'pairs': [{'a': a, 'b': b, 'p_raw': pr, 'p_holm': pa}
                      for (a, b), pr, pa in zip(pairs, raw, adj)]}


def rank_combos(get_values, labels, col, higher_better):
    """Rango medio de cada método, rankeando dentro de cada semilla."""
    cols = []
    for lab in labels:
        v = get_values(lab, col)
        if v is None:
            return None
        cols.append(np.asarray(v, dtype=float))
    n = min(len(v) for v in cols)
    M = np.column_stack([v[:n] for v in cols])
    A = -M if higher_better else M
    R = np.apply_along_axis(stats.rankdata, 1, A)
    return dict(zip(labels, R.mean(axis=0)))


def homogeneous_groups(res, labels, medians, higher_better):
    """Grupos de métodos que el post-hoc no logra separar, ordenados de mejor a
    peor.  Notación estándar para resumir las comparaciones en una línea."""
    order = sorted(labels, key=lambda l: -medians[l] if higher_better else medians[l])
    sep = {(p['a'], p['b']): p['p_holm'] < 0.05 for p in res['pairs']}

    def differ(a, b):
        return sep.get((a, b), sep.get((b, a), False))

    groups, cur = [], [order[0]]
    for lab in order[1:]:
        if any(differ(lab, m) for m in cur):
            groups.append(cur)
            cur = [lab]
        else:
            cur.append(lab)
    groups.append(cur)
    return groups


def _write_tex(lines, path, msg=None):
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ {msg or os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 1 — selección de hiperparámetros
#
#   Lee resultados/grid/all_metrics.csv (513 configs × 20 semillas) y elige, para
#   cada algoritmo genético, la mejor configuración dentro de cada combinación
#   de operadores (4 combos × 27 configs).  MOPSO se selecciona de una vez sobre
#   sus 81.  Total: 17 configuraciones.
#
#   Criterio: gana la de menor rango medio de hipervolumen, rankeando las
#   configuraciones dentro de cada una de las 20 semillas (que están pareadas:
#   la población inicial se muestrea con random_state=run_id).
#
#   Salida (plots/hiperparametros/)
#     <ALG>/main_effects_<ALG>.png  efecto de cada hiperparámetro, una curva por
#                                   combo
#     <ALG>/effects_<ALG>.tex       Friedman por bloques y W de Kendall por
#                                   hiperparámetro (los operadores no se
#                                   testean: se comparan en la etapa siguiente,
#                                   ya con su configuración afinada)
#     selected_configs.csv/.tex     las 17 configuraciones
# ═══════════════════════════════════════════════════════════════════════════

# MOPSO no tiene operadores; se colorea por inercia, que es su factor dominante.
W_COLORS = {0.4: '#9ECAE1', 0.6: '#4292C6', 0.9: '#08519C'}

# columna: (etiqueta, higher_better)
HP_METRICS = {
    'hypervolume': ('Hipervolumen', True),
    'spacing':     ('Espaciamiento', False),
    'n_pareto':    ('Tamaño de Pareto', True),
    'validity':    ('Validez', True),
    'novelty':     ('Novedad', True),
    'best_sa':     ('Mejor SA', False),
    'time_sec':    ('Tiempo (s)', False),
}

# 'budget' agrupa pop_size×n_gen: están acoplados por el presupuesto fijo de 100k
# evaluaciones.
FACTORS_GA = ['budget', 'crossover', 'mutation', 'cx_prob', 'mut_prob']
FACTORS_PSO = ['budget', 'w', 'c1', 'c2']

COMBO_FACTORS = ['crossover', 'mutation']
SUB_FACTORS_GA = ['budget', 'cx_prob', 'mut_prob']

OPERATOR_COLORS = {
    'pcx/pm':    '#1F4E79',
    'pcx/gauss': '#6FA8DC',
    'sbx/pm':    '#B45F06',
    'sbx/gauss': '#F6B26B',
}
# Combos tal como aparecen en el CSV del grid (la etapa 2 los lee de winners/,
# donde son nombres de directorio con guion bajo: ver COMBO_DIRS).
HP_COMBOS = ['pcx/pm', 'pcx/gauss', 'sbx/pm', 'sbx/gauss']

FACTOR_LABELS = {
    'budget':    'Población × Generaciones',
    'crossover': 'Cruce',
    'mutation':  'Mutación',
    'cx_prob':   'Prob. de cruce',
    'mut_prob':  'Prob. de mutación',
    'w':         'Inercia $w$',
    'c1':        'Cognitivo $c_1$',
    'c2':        'Social $c_2$',
}

BUDGET_ORDER = ['100×1000', '200×500', '400×250']


def factors_for(alg):
    return FACTORS_PSO if alg == 'MOPSO' else FACTORS_GA


# ─── Carga y selección ───────────────────────────────────────────────────────

def load_grid(csv_path):
    """Carga el CSV consolidado y añade la columna derivada 'budget'."""
    df = pd.read_csv(csv_path)
    df['budget'] = (df['pop_size'].astype(int).astype(str) + '×'
                    + df['n_gen'].astype(int).astype(str))
    return df


def run_matrix(g, factors, metric):
    """Matriz (configuraciones × semillas) de la métrica, sin las configs
    incompletas."""
    M = g.pivot_table(index=factors, columns='run', values=metric)
    complete = M.notna().all(axis=1)
    if not complete.all():
        print(f"  ⚠ {int((~complete).sum())} config(s) con semillas faltantes; se omiten")
    return M[complete]


def select_config(M, higher_better):
    """Rango medio de cada configuración, rankeando dentro de cada semilla
    (rango 1 = mejor de esa semilla).  Devuelve (elegida, rangos)."""
    A = -M.values if higher_better else M.values
    R = np.apply_along_axis(stats.rankdata, 0, A)
    ranks = pd.Series(R.mean(axis=1), index=M.index)
    return ranks.idxmin(), ranks


def config_label(cfg, factors):
    """Etiqueta compacta, p. ej. '400×250 pcx/pm cx=1 mut=0.031'."""
    cfg = cfg if isinstance(cfg, tuple) else (cfg,)
    parts, cx, mu = [], None, None
    for f, v in zip(factors, cfg):
        if f == 'budget':
            parts.append(str(v))
        elif f == 'crossover':
            cx = v
        elif f == 'mutation':
            mu = v
        elif f == 'cx_prob':
            parts.append(f'cx={v:g}')
        elif f == 'mut_prob':
            parts.append(f'mut={v:g}')
        else:
            parts.append(f'{f}={v:g}')
    if cx is not None:
        parts.insert(1, f'{cx}/{mu}' if mu is not None else cx)
    return ' '.join(parts)


def friedman_by_factor(g, factor, metric):
    """Efecto marginal de un hiperparámetro con la semilla como bloque.

    Friedman sobre la matriz (semillas × niveles), donde cada celda es la mediana
    marginal del nivel en esa semilla.  Tamaño de efecto: W de Kendall,
    W = χ²/(m(k−1)), la concordancia del orden entre las m semillas
    (0 = ninguna, 1 = las m ordenan igual).  Con 2 niveles Friedman degenera y se
    usa Wilcoxon de rangos con signo.  Devuelve (delta, W, p), donde delta es la
    diferencia entre el mejor y el peor nivel (mediana entre semillas)."""
    B = g.pivot_table(index='run', columns=factor, values=metric,
                      aggfunc='median').dropna()
    m, k = B.shape
    if m < 3 or k < 2:
        return np.nan, np.nan, np.nan

    level_medians = B.median(axis=0)
    delta = float(level_medians.max() - level_medians.min())

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


# ─── Gráficas ────────────────────────────────────────────────────────────────

def _level_order(factor, levels):
    """Orden natural de los niveles de un hiperparámetro."""
    if factor == 'budget':
        return [b for b in BUDGET_ORDER if b in levels]
    try:
        return sorted(levels, key=float)
    except (TypeError, ValueError):
        return sorted(levels)


def plot_main_effects(g, alg, factors, metric, out_dir, por_combo=True):
    """Un panel por hiperparámetro; en los GA, una curva por combinación de
    operadores.  En MOPSO, una sola curva."""
    label, higher = HP_METRICS[metric]
    n = len(factors)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False, sharey=True)
    axes = axes.flatten()

    if por_combo:
        g = g.copy()
        g['_combo'] = g['crossover'].astype(str) + '/' + g['mutation'].astype(str)
        series = [c for c in HP_COMBOS if c in set(g['_combo'])]
    else:
        series = [None]

    for ax, f in zip(axes, factors):
        levels = _level_order(f, g[f].dropna().unique().tolist())
        x = np.arange(len(levels))
        for s in series:
            gs = g if s is None else g[g['_combo'] == s]
            color = (pc.COLORS.get(alg, '#333333') if s is None
                     else OPERATOR_COLORS.get(s, '#888888'))
            meds = [np.median(gs.loc[gs[f] == lv, metric].dropna().values)
                    for lv in levels]
            ax.plot(x, meds, color=color, linewidth=2, zorder=3,
                    label=s or 'Mediana')
            ax.scatter(x, meds, s=45, color=color, zorder=4,
                       edgecolors='white', linewidths=1.0)

        # Δ y p no van acá: se reportan en effects_<ALG>.tex.
        ax.set_title(FACTOR_LABELS.get(f, f), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lv) for lv in levels], fontsize=10)
        ax.set_ylabel(f'{label} ({"↑" if higher else "↓"})')

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    order = [l for l in (series if por_combo else labels) if l in seen]
    order += [l for l in seen if l not in order]
    fig.legend([seen[l] for l in order], order, loc='lower center',
               ncol=len(order), framealpha=0.9, edgecolor='#cccccc',
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    sub = ' por combinación de operadores' if por_combo else ''
    fig.suptitle(f'Efecto de los hiperparámetros{sub} — {alg}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = f'main_effects_{alg}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


def plot_metric_vs_validity(g, alg, metric, out_dir, chosen, sub_factors,
                            por_combo):
    """Cada configuración del algoritmo como un punto: validez contra la métrica
    de selección, ambas medianas sobre las semillas.  El color separa las
    combinaciones de operadores (la inercia en MOPSO)."""
    label, _ = HP_METRICS[metric]
    if metric == 'validity':
        return
    fs = [f for f in factors_for(alg) if g[f].notna().any()]
    m = g.groupby(fs, observed=True)[[metric, 'validity']].median()

    if por_combo:
        key = (m.index.get_level_values('crossover').astype(str) + '/'
               + m.index.get_level_values('mutation').astype(str))
        groups = [(c, OPERATOR_COLORS[c]) for c in HP_COMBOS if c in set(key)]
    else:
        key = m.index.get_level_values('w')
        groups = [(w, c) for w, c in sorted(W_COLORS.items()) if w in set(key)]

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for name, color in groups:
        sel = key == name
        lab = name if por_combo else f'$w$ = {name:g}'
        ax.scatter(m.loc[sel, 'validity'], m.loc[sel, metric], s=34, alpha=0.65,
                   color=color, linewidths=0, label=lab, zorder=2)

    for name, b in chosen.items():
        cfg = b['cfg'] if isinstance(b['cfg'], tuple) else (b['cfg'],)
        lv = dict(zip(sub_factors, cfg))
        if name:
            lv['crossover'], lv['mutation'] = name.split('/')
        r = m.loc[tuple(lv[f] for f in fs)]
        color = (OPERATOR_COLORS[name] if por_combo
                 else W_COLORS.get(lv.get('w'), '#333333'))
        ax.scatter(r['validity'], r[metric], s=190, color=color,
                   edgecolors='black', linewidths=1.6, zorder=4)

    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker='o', linestyle='none', markersize=12,
                        markerfacecolor='#bbbbbb', markeredgecolor='black',
                        markeredgewidth=1.6))
    l.append('Seleccionada')
    leg = ax.legend(h, l, framealpha=0.9, edgecolor='#cccccc', fontsize=10,
                    loc='best')
    for lh in leg.legend_handles[:len(groups)]:
        lh.set_alpha(1.0)

    ax.margins(x=0.06, y=0.06)
    ax.set_xlabel('Validez (fracción de moléculas válidas) →')
    ax.set_ylabel(f'{label} →')
    ax.set_title(f'{label} contra validez — {alg}\n'
                 f'una configuración por punto, mediana de 20 semillas',
                 fontsize=12)
    plt.tight_layout()
    fname = f'{metric}_vs_validity_{alg}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# ─── Tablas ──────────────────────────────────────────────────────────────────

def _latex_label(f):
    """FACTOR_LABELS ya trae matemática ($w$, $c_1$); solo hay que traducir el '×'."""
    return FACTOR_LABELS.get(f, f).replace('×', r'$\times$')


def _holm_nan(pvals):
    """Holm sobre los p que existen; las celdas sin datos vuelven como NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = holm(p[ok])
    return out


def _fmt_delta(delta, p_holm):
    """Celda de la grilla: cuánto mueve la métrica entre el mejor y el peor
    nivel, con asterisco si el efecto sobrevive la corrección de Holm."""
    if pd.isna(delta):
        return '---'
    return f'{delta:.4f}' + ('*' if not pd.isna(p_holm) and p_holm < 0.05 else '')


def write_effects_table(effects, alg, factors, out_dir, metric):
    """Tabla LaTeX de sensibilidad a cada hiperparámetro.

    Reporta Δ, el recorrido de la métrica entre el mejor y el peor nivel, en sus
    propias unidades: es la magnitud del efecto y se lee contra el eje de
    main_effects sin necesidad de una escala auxiliar.

    Con combos de operadores la tabla es una grilla hiperparámetros × combos,
    porque agrupar los combos produce un orden espurio: pcx y sbx viven en
    regímenes de hipervolumen distintos y la mediana agrupada salta entre ellos."""
    label, _ = HP_METRICS[metric]
    por_combo = any(isinstance(v, dict) for v in effects.values())
    combos = ([c for c in HP_COMBOS
               if any(c in v for v in effects.values())] if por_combo else [])

    # Una tabla son 12 tests (3 factores × 4 combos) o 4 en MOPSO; sin corregir,
    # un p de 0.04 es lo que se espera por azar.  Se aplica Holm a la tabla
    # completa, el mismo estándar que el post-hoc de las etapas 2-4.
    if por_combo:
        keys = [(f, c) for f in factors for c in combos]
        raw = [effects[f].get(c, (np.nan,) * 3)[2] for f, c in keys]
    else:
        keys = [(f, None) for f in factors]
        raw = [effects.get(f, (np.nan,) * 3)[2] for f, _ in keys]
    p_holm = dict(zip(keys, _holm_nan(raw)))
    n_tests = int(np.sum(~np.isnan(np.asarray(raw, dtype=float))))

    # La grilla por combos marca la significancia con asterisco; la tabla sin
    # combos tiene columna de p propia, así que el caption no debe hablar de un
    # asterisco que no aparece.
    sig = (f'* indica $p < 0.05$ en el test de Friedman con las 20 semillas '
           f'como bloques, tras corregir por Holm las {n_tests} comparaciones '
           f'de la tabla.' if por_combo else
           f'$p$: test de Friedman con las 20 semillas como bloques, corregido '
           f'por Holm sobre las {n_tests} comparaciones de la tabla.')
    caption = (f'Sensibilidad de {_latex_escape(label)} a cada hiperparámetro '
               f'de {_latex_escape(alg)}.  $\\Delta$: diferencia entre el mejor '
               f'y el peor nivel, en unidades de {_latex_escape(label).lower()}.  '
               f'{sig}')
    if por_combo:
        caption += ('  Cada columna es una combinación de operadores, evaluada '
                    'por separado: agrupar los combos produce un orden espurio, '
                    'porque los operadores de cruce operan en regímenes de '
                    'hipervolumen distintos.')

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{tab:hp_effects_{alg.lower()}}}',
    ]

    if por_combo:
        lines += [
            r'\begin{tabular}{l' + 'c' * len(combos) + '}', r'\toprule',
            'Hiperparámetro & ' + ' & '.join(_latex_escape(c) for c in combos)
            + r' \\', r'\midrule',
        ]
        for f in factors:
            cells = [_fmt_delta(effects[f].get(c, (np.nan,) * 3)[0], p_holm[(f, c)])
                     for c in combos]
            lines.append(f'{_latex_label(f)} & ' + ' & '.join(cells) + r' \\')
    else:
        lines += [
            r'\begin{tabular}{lcc}', r'\toprule',
            r'Hiperparámetro & $\Delta$ & $p$ (Holm) \\', r'\midrule',
        ]
        for f in sorted(factors, key=lambda x: -np.nan_to_num(
                effects.get(x, (0, 0, 1))[0])):
            delta = effects.get(f, (np.nan,) * 3)[0]
            lines.append(f'{_latex_label(f)} & {delta:.4f} & '
                         f'{_fmt_p(p_holm[(f, None)])} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'effects_{alg}.tex'))


def write_selection_summary(per_alg, metric, out_dir):
    """CSV y tabla LaTeX de las configuraciones seleccionadas.  El CSV usa los
    nombres de columna que consume run_experiments.py."""
    label, _ = HP_METRICS[metric]
    rows = []
    for alg in [a for a in pc.ALGORITHM_ORDER if a in per_alg]:
        d = per_alg[alg]
        for name, b in d['blocks'].items():
            lv = dict(zip(d['sub_factors'], b['cfg']))
            if d['por_combo'] and name:
                lv['crossover'], lv['mutation'] = name.split('/')
            pop, gen = str(lv['budget']).split('×')
            row = {'algorithm': alg, 'operators': name or '',
                   'pop_size': int(pop), 'n_gen': int(gen),
                   'config': b['label'], 'avg_rank': b['rank'],
                   'mean': b['mean'], 'std': b['std'],
                   'n_configs': b['n_configs']}
            for f in ('crossover', 'mutation', 'cx_prob', 'mut_prob',
                      'w', 'c1', 'c2'):
                if f in lv:
                    row[f] = lv[f]
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'selected_configs.csv'), index=False)
    print(f"  ✓ selected_configs.csv  ({len(out)} configuraciones)")

    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        f'\\caption{{Configuraciones seleccionadas: la mejor de cada combinación '
        f'de operadores en los algoritmos genéticos, y la mejor global en MOPSO. '
        f'Gana la de menor rango medio de {_latex_escape(label)}, rankeando las '
        f'configuraciones dentro de cada una de las 20 semillas.}}',
        r'\label{tab:hp_seleccionadas}',
        r'\begin{tabular}{lllcc}', r'\toprule',
        r'Algoritmo & Operadores & Configuración & Rango medio & '
        r'$\mu \pm \sigma$ \\',
        r'\midrule',
    ]
    prev = None
    for _, r in out.iterrows():
        alg_cell = '' if r['algorithm'] == prev else _latex_escape(r['algorithm'])
        if alg_cell and prev is not None:
            lines.append(r'\midrule')
        prev = r['algorithm']
        lines.append(
            f"{alg_cell} & {_latex_escape(r['operators']) or '---'} & "
            f"{_latex_escape(r['config'])} & {r['avg_rank']:.2f} & "
            f"{r['mean']:.4f} $\\pm$ {r['std']:.4f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'selected_configs.tex'))


# ─── Orquestación ────────────────────────────────────────────────────────────

def analyze_algorithm(g, alg, metric, out_dir):
    """Elige la configuración de cada combinación de operadores y mide la
    sensibilidad a cada hiperparámetro sobre el grid completo."""
    factors = [f for f in factors_for(alg) if g[f].notna().any()]
    _, higher = HP_METRICS[metric]
    por_combo = set(COMBO_FACTORS).issubset(factors)
    sub_factors = SUB_FACTORS_GA if por_combo else factors

    print(f"\n{'─'*66}")
    print(f"  {alg}")
    print(f"{'─'*66}")

    if por_combo:
        blocks = [(f'{cx}/{mu}', gg)
                  for (cx, mu), gg in g.groupby(COMBO_FACTORS, observed=True)]
        blocks.sort(key=lambda t: HP_COMBOS.index(t[0])
                    if t[0] in HP_COMBOS else len(HP_COMBOS))
    else:
        blocks = [(None, g)]

    chosen = {}
    for name, gg in blocks:
        M = run_matrix(gg, sub_factors, metric)
        if M.shape[0] < 2:
            continue
        best, ranks = select_config(M, higher)
        vals = M.loc[best].values
        chosen[name] = {'cfg': best, 'label': config_label(best, sub_factors),
                        'rank': float(ranks.loc[best]),
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals, ddof=1)),
                        'n_configs': M.shape[0]}
        print(f"  {(name or 'todas'):11s} {chosen[name]['label']}")
        print(f"              rango {chosen[name]['rank']:.2f} de {M.shape[0]} "
              f"configs   μ {chosen[name]['mean']:.5f} ± "
              f"{chosen[name]['std']:.5f}")

    if not chosen:
        print("  ⚠ sin configuraciones completas; se omite")
        return None

    # Los operadores no se testean acá: se barren dentro de cada bloque y su
    # comparación va en la etapa siguiente, entre las configuraciones ya
    # seleccionadas de cada combo.  El resto se mide dentro de cada combo, no
    # agrupando: pcx y sbx están en regímenes de hipervolumen distintos y al
    # agruparlos la mediana salta entre ellos, generando un orden espurio.
    if por_combo:
        gg = g.copy()
        gg['_combo'] = gg['crossover'].astype(str) + '/' + gg['mutation'].astype(str)
        effects = {f: {c: friedman_by_factor(sub, f, metric)
                       for c, sub in gg.groupby('_combo', observed=True)}
                   for f in sub_factors}
    else:
        effects = {f: friedman_by_factor(g, f, metric) for f in sub_factors}

    def _max_W(e):
        """Con combos, un factor tiene un W por combo; se resume por el mayor."""
        if isinstance(e, dict):
            vals = [w for _, w, _ in e.values() if not pd.isna(w)]
            return max(vals) if vals else np.nan
        return e[1]

    top_f = max(effects, key=lambda f: np.nan_to_num(_max_W(effects[f])))
    print(f"  Hiperparámetro dominante: {FACTOR_LABELS.get(top_f, top_f)}  "
          f"(W = {_max_W(effects[top_f]):.3f})")

    plot_main_effects(g, alg, sub_factors, metric, out_dir, por_combo=por_combo)
    plot_metric_vs_validity(g, alg, metric, out_dir, chosen, sub_factors,
                            por_combo)
    write_effects_table(effects, alg, sub_factors, out_dir, metric)

    return {'blocks': chosen, 'factors': factors, 'sub_factors': sub_factors,
            'effects': effects, 'por_combo': por_combo}


def etapa1(args):
    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}.\n"
              f"  En el cluster:  python run_experiments.py --summary-only\n"
              f"  Sobre una copia ya bajada:  python -c "
              f"\"from utils_mo import consolidate_all; "
              f"consolidate_all('{os.path.dirname(args.csv)}')\"")
        return

    df = load_grid(args.csv)
    algs = args.algorithms or [a for a in pc.ALGORITHM_ORDER
                               if a in set(df['algorithm'])]

    print(f"\n{'='*66}")
    print(f"  ETAPA 1a — SELECCIÓN DE HIPERPARÁMETROS")
    print(f"  Datos: {args.csv}  ({len(df)} ejecuciones)")
    print(f"  Métrica: {HP_METRICS[args.metric][0]} "
          f"({'↑' if HP_METRICS[args.metric][1] else '↓'})")
    print(f"  Criterio: menor rango medio, rankeando dentro de cada semilla")
    print(f"  Algoritmos: {', '.join(algs)}")
    print(f"{'='*66}")

    os.makedirs(args.out, exist_ok=True)
    per_alg = {}
    for alg in algs:
        g = df[df['algorithm'] == alg].copy()
        if g.empty:
            print(f"\n  ⚠ {alg}: sin datos")
            continue
        out_dir = os.path.join(args.out, alg)
        os.makedirs(out_dir, exist_ok=True)
        r = analyze_algorithm(g, alg, args.metric, out_dir)
        if r:
            per_alg[alg] = r

    if per_alg:
        print(f"\n{'─'*66}\n  Resumen global\n{'─'*66}")
        write_selection_summary(per_alg, args.metric, args.out)

    print(f"\n{'='*66}")
    print(f"  ✅ Listo: {args.out}")
    print(f"{'='*66}\n")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 2 — comparación de combinaciones de operadores, por algoritmo
#
#   Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (las configuraciones
#   que ganaron su bloque en la etapa 1) y compara los 4 combos entre sí.
#
#   Salida por algoritmo (plots/operadores/<ALG>/)
#     comparison_multiobj_*.tex     indicadores multiobjetivo (HV, spacing,
#                                   IGD+, ε+)
#     comparison_chemical_*.tex     indicadores químicos (QED, SA, validez,
#                                   unicidad…)
#     pareto_comparison_*.png       frentes de Pareto superpuestos
#     pareto_qed_sa_grid_*.png      frentes QED-SA en paneles
#     tests_<ALG>.tex               Friedman + Wilcoxon post-hoc con Holm
#     resumen_tests.csv             el combo elegido de cada algoritmo
# ═══════════════════════════════════════════════════════════════════════════

GA_ALGS = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA']

# Los combos como nombres de directorio bajo winners/ (con guion bajo).
COMBO_DIRS = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']

# (columna, etiqueta, mayor_es_mejor)
OP_INDICATORS = [
    ('hypervolume', 'Hipervolumen',      True),
    ('igd_plus',    'IGD$^+$',           False),
    ('epsilon',     r'$\epsilon^+$',     False),
    ('spacing',     'Espaciamiento',     False),
    ('n_pareto',    'Tamaño de Pareto',  True),
    ('validity',    'Validez',           True),
    ('uniqueness',  'Unicidad',          True),
]


def _series_operadores(alg, winners_dir):
    """Una serie por combo de operadores.  El nivel de configuración se resuelve
    con glob porque cada combo ganó con hiperparámetros distintos."""
    series = []
    for combo in COMBO_DIRS:
        matches = sorted(glob.glob(os.path.join(winners_dir, alg, combo, '*')))
        cfg_dirs = [d for d in matches if pc._has_runs(d)]
        if not cfg_dirs:
            continue
        series.append(pc.Series(combo, cfg_dirs[0], color_key=combo))
    return series


def write_tests_table(res, alg, out_dir, label):
    """Tabla LaTeX de las 6 comparaciones por pares sobre el indicador de
    decisión.  La magnitud del efecto no se repite acá: está en la tabla de
    indicadores, como media ± desvío por combo."""
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación de operadores en {_latex_escape(alg)} sobre '
        f'{label.lower()}.  Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm.}}',
        f'\\label{{tab:ops_tests_{alg.lower()}}}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Par & $p$ (Holm) & Significativo \\', r'\midrule',
    ]
    for pr in res['pairs']:
        sig = 'sí' if pr['p_holm'] < 0.05 else 'no'
        lines.append(f"{_latex_escape(pr['a'])} vs {_latex_escape(pr['b'])} & "
                     f"{_fmt_p(pr['p_holm'])} & {sig} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'tests_{alg}.tex'))


def analyze_operators(alg, winners_dir, out_root, decision_col):
    series = _series_operadores(alg, winners_dir)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    # Frente de referencia común a los 4 combos → IGD+ y ε+ comparables.
    pf_F, pf_df = pc.build_reference_front(series, None)
    indicator_data = {}
    if pf_F is not None:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicator_data = pc.compute_indicators_per_run(series, None, pf_F)
        pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = pc._build_series_value_getter(series, indicator_data)

    # Las 4 salidas de la sección 3.2: 2 tablas + 2 figuras de frentes.
    pc.generate_latex_comparison_tables(series, alg, out_dir, get_values)
    pc.plot_pareto_comparison(series, alg, out_dir)
    pc.plot_pareto_qed_sa_grid(series, alg, out_dir)

    # Test: solo sobre el indicador de decisión.  Los demás indicadores se
    # reportan de forma descriptiva en las tablas de comparación.
    label, higher = dict((c, (l, h)) for c, l, h in OP_INDICATORS)[decision_col]
    res = compare_indicator(get_values, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)
    print(f"  Friedman ({label}): p = {res['p_omnibus']:.4g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}")

    write_tests_table(res, alg, out_dir, label)

    groups = homogeneous_groups(res, labels, res['medians'], higher)
    txt = ' > '.join('{' + ', '.join(g) + '}' for g in groups)
    print(f"  grupos homogéneos: {txt}")

    return {'algorithm': alg, 'p_friedman': res['p_omnibus'],
            'n_pares_sig': n_sig, 'n_pares': len(res['pairs']),
            'grupos': txt,
            'mejor_grupo': ', '.join(groups[0])}


def etapa2(args):
    algs = args.algorithms or GA_ALGS
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  ETAPA 2 — COMPARACIÓN DE OPERADORES")
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


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 3 — comparación estadística final entre algoritmos
#
#   Lee finalistas/<ALG>/run_XX/ (la configuración elegida de cada algoritmo
#   tras las etapas 1 y 2) y contrasta los cinco entre sí.
#
#   Salida (plots/comparacion_final/)
#     grupos_homogeneos.tex   una fila por indicador
#     tests_pares.csv         las 10 comparaciones, con p de Holm
# ═══════════════════════════════════════════════════════════════════════════

ALG_METRIC = 'hypervolume'
ALG_METRIC_LABEL = 'hipervolumen'
ALG_HIGHER_BETTER = True


def write_groups_table(res, groups, out_dir):
    """Las 10 comparaciones por pares, con el resultado resumido en grupos."""
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación entre algoritmos sobre {ALG_METRIC_LABEL}.  '
        f'Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm.  '
        f'Grupos homogéneos, de mejor a peor: {fmt_groups(groups)}.}}',
        r'\label{tab:comparacion_grupos}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Par & $p$ (Holm) & Significativo \\', r'\midrule',
    ]
    for p in res['pairs']:
        sig = 'sí' if p['p_holm'] < 0.05 else 'no'
        lines.append(f"{DISPLAY.get(p['a'], p['a'])} vs {DISPLAY.get(p['b'], p['b'])} "
                     f"& {_fmt_p(p['p_holm'])} & {sig} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    path = os.path.join(out_dir, 'grupos_homogeneos.tex')
    print()
    _write_tex(lines, path, msg=path)


def etapa3(args):
    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    series = pc.build_finalist_series(algs, args.finalistas)
    if len(series) < 3:
        print(f"Se necesitan ≥3 algoritmos en {args.finalistas}")
        return
    labels = [s.label for s in series]
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ETAPA 3 — COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = pc.build_reference_front(series, None)
    ind = pc.compute_indicators_per_run(series, None, pf) if pf is not None else {}
    print(f"  frente de referencia: {len(pf) if pf is not None else 0} soluciones\n")
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, ALG_METRIC)
    if res is None:
        print(f"  ⚠ sin datos de {ALG_METRIC}")
        return
    groups = homogeneous_groups(res, labels, res['medians'], ALG_HIGHER_BETTER)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}\n")
    for lab in sorted(labels, key=lambda l: -res['medians'][l]):
        print(f"    {DISPLAY.get(lab, lab):10s} mediana = {res['medians'][lab]:.5f}")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_groups_table(res, groups, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print(f"  ✓ {os.path.join(args.out, 'tests_pares.csv')}")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 4 — baselines contra los algoritmos multiobjetivo
#
#   Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de
#   suma ponderada) con los cinco MOEAs ya seleccionados, sobre el mismo
#   presupuesto de 100.000 evaluaciones y las mismas 20 semillas.
#
#   Salida (plots/baselines/)
#     grupos_homogeneos.tex   los grupos y las comparaciones contra cada baseline
#     comparacion.tex         mediana ± desvío de cada método
#     tests_pares.csv         todas las comparaciones con su p de Holm
# ═══════════════════════════════════════════════════════════════════════════

# Orden de peor a mejor esperado; LHS quedó fuera por ser indistinguible de RANDOM.
BASELINE_KEYS = ['WEIGHTED_GA', 'HILL_CLIMBER', 'RANDOM', 'SCREENING']


def _series_baselines(finalistas, baselines):
    """Los cinco MOEAs y las tres baselines como series comparables."""
    series = []
    for alg in pc.ALGORITHM_ORDER:
        d = os.path.join(finalistas, alg)
        if pc._has_runs(d):
            series.append(pc.Series(alg, d, color_key=alg))
    for m in BASELINE_KEYS:
        # <baselines>/<METHOD>/[tag/]pop{P}_gen{G}/
        for d in sorted(glob.glob(os.path.join(baselines, m, '*', '*')) +
                        glob.glob(os.path.join(baselines, m, '*'))):
            if pc._has_runs(d):
                series.append(pc.Series(m, d, color_key=m))
                break
    return series


def write_baseline_tables(res, groups, medians, stds, labels, metric_label,
                          out_dir):
    n_moea = sum(1 for l in labels if l not in BASELINE_KEYS)

    # Tabla 1: descriptiva
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Algoritmos multiobjetivo frente a las baselines sobre '
        f'{metric_label}, con idéntico presupuesto de 100.000 evaluaciones y las '
        f'mismas 20 semillas.  Mediana y desvío estándar entre ejecuciones.}}',
        r'\label{tab:baselines_desc}',
        r'\begin{tabular}{llc}', r'\toprule',
        r'& Método & ' + metric_label.capitalize() + r' \\', r'\midrule',
    ]
    for i, lab in enumerate(labels):
        if i == n_moea:
            lines.append(r'\midrule')
        tipo = ''
        if i == 0:
            tipo = r'\multirow{%d}{*}{MOEA}' % n_moea
        elif i == n_moea:
            tipo = r'\multirow{%d}{*}{Baseline}' % (len(labels) - n_moea)
        lines.append(f"{tipo} & {DISPLAY.get(lab, lab)} & "
                     f"${medians[lab]:.4f} \\pm {stds[lab]:.4f}$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'comparacion.tex'))

    # Tabla 2: cada MOEA contra cada baseline
    moeas = [l for l in labels if l not in BASELINE_KEYS]
    bases = [l for l in labels if l in BASELINE_KEYS]
    pmap = {}
    for p in res['pairs']:
        pmap[(p['a'], p['b'])] = p['p_holm']
        pmap[(p['b'], p['a'])] = p['p_holm']

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparaciones por pares entre cada algoritmo multiobjetivo y '
        f'cada baseline sobre {metric_label}.  Test de Friedman global '
        f'($p$ = {_fmt_p(res["p_omnibus"])}) y Wilcoxon de rangos con signo por '
        f'pares con corrección de Holm; se indica el $p$ ajustado.  '
        f'Grupos homogéneos: {fmt_groups(groups)}.}}',
        r'\label{tab:baselines_tests}',
        r'\begin{tabular}{l' + 'c' * len(bases) + '}', r'\toprule',
        'Algoritmo & ' + ' & '.join(DISPLAY.get(b, b) for b in bases) + r' \\',
        r'\midrule',
    ]
    for m in moeas:
        cells = [_fmt_p(pmap.get((m, b))) for b in bases]
        lines.append(f'{DISPLAY.get(m, m)} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'grupos_homogeneos.tex'))


def etapa4(args):
    series = _series_baselines(args.finalistas, args.baselines)
    labels = [s.label for s in series]
    faltan = [m for m in BASELINE_KEYS if m not in labels]
    if faltan:
        print(f"  ⚠ sin datos de: {', '.join(faltan)}")
    if len(series) < 3:
        print("Se necesitan más series")
        return
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ETAPA 4 — BASELINES vs ALGORITMOS MULTIOBJETIVO")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = pc.build_reference_front(series, None)
    ind = pc.compute_indicators_per_run(series, None, pf) if pf is not None else {}
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, args.metric)
    if res is None:
        print(f"  ⚠ sin datos de {args.metric}")
        return
    stds = {l: float(np.std(np.asarray(get(l, args.metric), float), ddof=1))
            for l in labels}
    groups = homogeneous_groups(res, labels, res['medians'], True)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}   "
          f"({sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)} de "
          f"{len(res['pairs'])} pares significativos)\n")
    for l in sorted(labels, key=lambda x: -res['medians'][x]):
        tipo = 'baseline' if l in BASELINE_KEYS else 'MOEA'
        print(f"    {DISPLAY.get(l, l):14s} {res['medians'][l]:.5f}  ({tipo})")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_baseline_tables(res, groups, res['medians'], stds, labels,
                          args.metric_label, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print("  ✓ tests_pares.csv")


# ═══════════════════════════════════════════════════════════════════════════
#   Moléculas representativas
#
#   Una imagen con las moléculas de mayor QED del frente de cada algoritmo.  El
#   frente se arma juntando las moléculas de las 20 ejecuciones, deduplicando
#   por SMILES y recalculando la dominancia global.
# ═══════════════════════════════════════════════════════════════════════════

MOLECULAS_OUT = os.path.join(OUT_ALGORITMOS, "moleculas_representativas.png")

N_MOLECULAS = 5      # por algoritmo


def load_front(alg, finalistas):
    """Frente no dominado global de un algoritmo, sobre sus 20 ejecuciones."""
    df = pc.load_pareto_molecules(os.path.join(finalistas, alg))
    if df.empty:
        return df
    return pc._compute_non_dominated(df.drop_duplicates(subset='smiles'))


def pick(front, n=N_MOLECULAS):
    """Las n moléculas de mayor QED, desempatando por menor SA."""
    f = front.assign(_qed=front['qed'].round(3))
    return (f.sort_values(['_qed', 'sa'], ascending=[False, True])
            .head(n).drop(columns='_qed').reset_index(drop=True))


def render(smiles, size=(420, 320)):
    """PNG de la estructura, como array para matplotlib."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format='png')


def moleculas(args):
    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    fronts = {a: load_front(a, args.finalistas) for a in algs}
    fronts = {a: f for a, f in fronts.items() if not f.empty}
    if not fronts:
        print(f"No se encontraron frentes en {args.finalistas}")
        return

    seleccion = {a: pick(f) for a, f in fronts.items()}

    nrows, ncols = len(fronts), N_MOLECULAS
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for i, (alg, sel) in enumerate(seleccion.items()):
        for j in range(ncols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
            if j >= len(sel):
                ax.set_visible(False)
                continue

            m = sel.iloc[j]
            img = render(m['smiles'])
            if img is not None:
                ax.imshow(img)
            ax.set_xlabel(f"QED {m['qed']:.3f}   ·   SA {m['sa']:.2f}",
                          fontsize=9.5, labelpad=3)
            if j == 0:
                ax.set_ylabel(DISPLAY.get(alg, alg), fontsize=13,
                              fontweight='bold', labelpad=10)

    fig.suptitle(f'Las {N_MOLECULAS} moléculas de mayor QED del frente de cada '
                 f'algoritmo',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {args.out}")

    # Detalle de las moléculas elegidas
    rows = []
    for alg, sel in seleccion.items():
        for j, m in sel.iterrows():
            rows.append({'algoritmo': DISPLAY.get(alg, alg), 'puesto': j + 1,
                         'qed': round(m['qed'], 4), 'sa': round(m['sa'], 2),
                         'lipinski': m['lipinski'], 'smiles': m['smiles']})
    out = pd.DataFrame(rows)
    csv = os.path.splitext(args.out)[0] + '.csv'
    out.to_csv(csv, index=False)
    print(f"✓ {csv}\n")
    print(out.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='etapa', required=True,
                            metavar='etapa1|etapa2|etapa3|etapa4|moleculas')
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
    pm.add_argument('--out', default=MOLECULAS_OUT)
    pm.set_defaults(func=moleculas)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
