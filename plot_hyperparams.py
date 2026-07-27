"""
Etapa 1a: selección de hiperparámetros.

Lee results_light/all_metrics.csv (513 configs × 20 semillas) y elige, para cada
algoritmo genético, la mejor configuración dentro de cada combinación de
operadores (4 combos × 27 configs).  MOPSO se selecciona de una vez sobre sus 81.
Total: 17 configuraciones.

Criterio: gana la de menor rango medio de hipervolumen, rankeando las
configuraciones dentro de cada una de las 20 semillas (que están pareadas: la
población inicial se muestrea con random_state=run_id).

Salida (plots_hp/)
  <ALG>/main_effects_<ALG>.png  efecto de cada hiperparámetro, una curva por combo
  <ALG>/effects_<ALG>.tex       Friedman por bloques y W de Kendall por hiperparámetro
                               (los operadores no se testean: se comparan en la
                                etapa siguiente, ya con su configuración afinada)
  selected_configs.csv/.tex     las 17 configuraciones

Uso:
    python plot_hyperparams.py
    python plot_hyperparams.py --algorithms NSGA2 MOPSO
    python plot_hyperparams.py --metric spacing
"""

import os
import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.color': 'grey',
})

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(ROOT_DIR, "results_light", "all_metrics.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots_hp")

COLORS = {
    'NSGA2':   '#000000',
    'MOPSO':   '#FF0000',
    'AGEMOEA': '#008000',
    'MOEAD':   '#1F77B4',
    'NSGA3':   '#7B1FA2',
}
ALG_ORDER = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'MOPSO']

# MOPSO no tiene operadores; se colorea por inercia, que es su factor dominante.
W_COLORS = {0.4: '#9ECAE1', 0.6: '#4292C6', 0.9: '#08519C'}

# columna: (etiqueta, higher_better)
METRICS = {
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
COMBO_ORDER = ['pcx/pm', 'pcx/gauss', 'sbx/pm', 'sbx/gauss']

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


def plot_main_effects(g, alg, factors, metric, out_dir, effect_stats,
                      por_combo=True):
    """Un panel por hiperparámetro; en los GA, una curva por combinación de
    operadores.  En MOPSO, una sola curva."""
    label, higher = METRICS[metric]
    n = len(factors)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False, sharey=True)
    axes = axes.flatten()

    if por_combo:
        g = g.copy()
        g['_combo'] = g['crossover'].astype(str) + '/' + g['mutation'].astype(str)
        series = [c for c in COMBO_ORDER if c in set(g['_combo'])]
    else:
        series = [None]

    for ax, f in zip(axes, factors):
        levels = _level_order(f, g[f].dropna().unique().tolist())
        x = np.arange(len(levels))
        for s in series:
            gs = g if s is None else g[g['_combo'] == s]
            color = (COLORS.get(alg, '#333333') if s is None
                     else OPERATOR_COLORS.get(s, '#888888'))
            meds = [np.median(gs.loc[gs[f] == lv, metric].dropna().values)
                    for lv in levels]
            ax.plot(x, meds, color=color, linewidth=2, zorder=3,
                    label=s or 'Mediana')
            ax.scatter(x, meds, s=45, color=color, zorder=4,
                       edgecolors='white', linewidths=1.0)

        e = effect_stats.get(f, (np.nan, np.nan, np.nan))
        if isinstance(e, dict):
            # Un W por combo; van en la tabla, no caben cuatro en el título.
            ax.set_title(FACTOR_LABELS.get(f, f), fontsize=11)
        else:
            _, W, p = e
            ptxt = '< 0.001' if p < 1e-3 else f'= {p:.3f}'
            ax.set_title(f'{FACTOR_LABELS.get(f, f)}\n'
                         f'$W$ = {W:.3f},  $p$ {ptxt}', fontsize=11)
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
    label, _ = METRICS[metric]
    if metric == 'validity':
        return
    fs = [f for f in factors_for(alg) if g[f].notna().any()]
    m = g.groupby(fs, observed=True)[[metric, 'validity']].median()

    if por_combo:
        key = (m.index.get_level_values('crossover').astype(str) + '/'
               + m.index.get_level_values('mutation').astype(str))
        groups = [(c, OPERATOR_COLORS[c]) for c in COMBO_ORDER if c in set(key)]
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

def _latex_escape(s):
    repl = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            '×': r'$\times$'}
    return ''.join(repl.get(c, c) for c in str(s))


def _latex_label(f):
    """FACTOR_LABELS ya trae matemática ($w$, $c_1$); solo hay que traducir el '×'."""
    return FACTOR_LABELS.get(f, f).replace('×', r'$\times$')


def _fmt_p(p):
    if pd.isna(p):
        return '---'
    return r'$<$0.001' if p < 1e-3 else f'{p:.3f}'


def _fmt_delta(e):
    """Celda de la grilla: cuánto mueve la métrica entre el mejor y el peor
    nivel, con asterisco si el efecto es significativo."""
    delta, _, p = e
    if pd.isna(delta):
        return '---'
    return f'{delta:.4f}' + ('*' if not pd.isna(p) and p < 0.05 else '')


def write_effects_table(effects, alg, factors, out_dir, metric):
    """Tabla LaTeX de sensibilidad a cada hiperparámetro.

    Reporta Δ, el recorrido de la métrica entre el mejor y el peor nivel, en sus
    propias unidades: es la magnitud del efecto y se lee contra el eje de
    main_effects sin necesidad de una escala auxiliar.

    Con combos de operadores la tabla es una grilla hiperparámetros × combos,
    porque agrupar los combos produce un orden espurio: pcx y sbx viven en
    regímenes de hipervolumen distintos y la mediana agrupada salta entre ellos."""
    label, _ = METRICS[metric]
    por_combo = any(isinstance(v, dict) for v in effects.values())
    combos = ([c for c in COMBO_ORDER
               if any(c in v for v in effects.values())] if por_combo else [])

    caption = (f'Sensibilidad de {_latex_escape(label)} a cada hiperparámetro '
               f'de {_latex_escape(alg)}.  $\\Delta$: diferencia entre el mejor '
               f'y el peor nivel, en unidades de {_latex_escape(label).lower()}.  '
               f'* indica $p < 0.05$ en el test de Friedman con las 20 semillas '
               f'como bloques.')
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
            cells = [_fmt_delta(effects[f].get(c, (np.nan,) * 3)) for c in combos]
            lines.append(f'{_latex_label(f)} & ' + ' & '.join(cells) + r' \\')
    else:
        lines += [
            r'\begin{tabular}{lcc}', r'\toprule',
            r'Hiperparámetro & $\Delta$ & $p$ \\', r'\midrule',
        ]
        for f in sorted(factors, key=lambda x: -np.nan_to_num(
                effects.get(x, (0, 0, 1))[0])):
            delta, _, p = effects.get(f, (np.nan, np.nan, np.nan))
            lines.append(f'{_latex_label(f)} & {delta:.4f} & {_fmt_p(p)} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(out_dir, f'effects_{alg}.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ effects_{alg}.tex")


def write_selection_summary(per_alg, metric, out_dir):
    """CSV y tabla LaTeX de las configuraciones seleccionadas.  El CSV usa los
    nombres de columna que consume run_experiments.py."""
    label, _ = METRICS[metric]
    rows = []
    for alg in [a for a in ALG_ORDER if a in per_alg]:
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
    with open(os.path.join(out_dir, 'selected_configs.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ selected_configs.tex")


# ─── Orquestación ────────────────────────────────────────────────────────────

def analyze_algorithm(g, alg, metric, out_dir):
    """Elige la configuración de cada combinación de operadores y mide la
    sensibilidad a cada hiperparámetro sobre el grid completo."""
    factors = [f for f in factors_for(alg) if g[f].notna().any()]
    _, higher = METRICS[metric]
    por_combo = set(COMBO_FACTORS).issubset(factors)
    sub_factors = SUB_FACTORS_GA if por_combo else factors

    print(f"\n{'─'*66}")
    print(f"  {alg}")
    print(f"{'─'*66}")

    if por_combo:
        blocks = [(f'{cx}/{mu}', gg)
                  for (cx, mu), gg in g.groupby(COMBO_FACTORS, observed=True)]
        blocks.sort(key=lambda t: COMBO_ORDER.index(t[0])
                    if t[0] in COMBO_ORDER else len(COMBO_ORDER))
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

    plot_main_effects(g, alg, sub_factors, metric, out_dir, effects,
                      por_combo=por_combo)
    plot_metric_vs_validity(g, alg, metric, out_dir, chosen, sub_factors,
                            por_combo)
    write_effects_table(effects, alg, sub_factors, out_dir, metric)

    return {'blocks': chosen, 'factors': factors, 'sub_factors': sub_factors,
            'effects': effects, 'por_combo': por_combo}


def main():
    ap = argparse.ArgumentParser(
        description="Etapa 1a: selección de hiperparámetros por combinación de "
                    "operadores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--csv', default=DEFAULT_CSV, help="CSV consolidado del grid.")
    ap.add_argument('--out', default=PLOTS_DIR, help="Directorio de salida.")
    ap.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a analizar (default: todos).")
    ap.add_argument('--metric', default='hypervolume', choices=list(METRICS),
                    help="Métrica de selección.")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}. Corré: python run_experiments.py --summary-only")
        return

    df = load_grid(args.csv)
    algs = args.algorithms or [a for a in ALG_ORDER if a in set(df['algorithm'])]

    print(f"\n{'='*66}")
    print(f"  ETAPA 1a — SELECCIÓN DE HIPERPARÁMETROS")
    print(f"  Datos: {args.csv}  ({len(df)} ejecuciones)")
    print(f"  Métrica: {METRICS[args.metric][0]} "
          f"({'↑' if METRICS[args.metric][1] else '↓'})")
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


if __name__ == "__main__":
    main()
