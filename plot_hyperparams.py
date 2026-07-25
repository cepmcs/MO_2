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
  <ALG>/effects_<ALG>.tex       Kruskal-Wallis y ε² por hiperparámetro
  effect_sizes_overview.png     ε² de cada hiperparámetro en cada algoritmo
  selected_configs.csv/.tex     las 17 configuraciones

Uso:
    python plot_hyperparams.py
    python plot_hyperparams.py --algorithms NSGA2 MOPSO
    python plot_hyperparams.py --metric spacing
"""

import os
import argparse
import itertools
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


def kruskal_by_factor(g, factor, metric):
    """Kruskal-Wallis del efecto marginal de un hiperparámetro + tamaño de
    efecto ε² = (H − k + 1)/(n − k).  0.01/0.06/0.14 = pequeño/medio/grande."""
    groups = [x[metric].dropna().values for _, x in g.groupby(factor, observed=True)]
    groups = [x for x in groups if len(x)]
    k, n = len(groups), sum(len(x) for x in groups)
    if k < 2:
        return np.nan, np.nan, np.nan
    H, p = stats.kruskal(*groups)
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
    return float(H), float(p), float(max(eps2, 0.0))


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

        H, p, eps2 = effect_stats.get(f, (np.nan, np.nan, np.nan))
        ptxt = '< 0.001' if p < 1e-3 else f'= {p:.3f}'
        ax.set_title(f'{FACTOR_LABELS.get(f, f)}\n'
                     f'$\\varepsilon^2$ marginal = {eps2:.3f},  $p$ {ptxt}',
                     fontsize=11)
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


def plot_effect_sizes_overview(per_alg, metric, out_dir):
    """ε² de cada hiperparámetro en cada algoritmo."""
    algs = [a for a in ALG_ORDER if a in per_alg]
    if not algs:
        return
    all_factors = list(dict.fromkeys(
        itertools.chain.from_iterable(per_alg[a]['factors'] for a in algs)))

    fig, ax = plt.subplots(figsize=(1.35 * len(all_factors) + 5, 5.4))
    width = 0.8 / len(algs)
    x = np.arange(len(all_factors))
    for i, a in enumerate(algs):
        vals = [per_alg[a]['effects'].get(f, (np.nan, np.nan, np.nan))[2]
                for f in all_factors]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               color=COLORS.get(a, '#333333'), alpha=0.8, label=a)

    for y, tag in [(0.01, 'pequeño'), (0.06, 'medio'), (0.14, 'grande')]:
        ax.axhline(y, color='#888888', linestyle=':', linewidth=1, zorder=1)
        ax.text(len(all_factors) - 0.45, y, f' {tag}', va='bottom', ha='right',
                fontsize=8.5, color='#666666')

    ax.set_xticks(x)
    ax.set_xticklabels([FACTOR_LABELS.get(f, f) for f in all_factors],
                       rotation=18, ha='right', fontsize=10)
    ax.set_ylabel('$\\varepsilon^2$ (Kruskal-Wallis)')
    ax.set_title(f'Importancia de cada hiperparámetro sobre '
                 f'{METRICS[metric][0]}\n(fracción de varianza de rangos explicada)',
                 fontsize=12)
    ax.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=10)
    plt.tight_layout()
    fname = 'effect_sizes_overview.png'
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


def _fmt_p(p):
    if pd.isna(p):
        return '---'
    return r'$<$0.001' if p < 1e-3 else f'{p:.3f}'


def write_effects_table(effects, alg, factors, out_dir, metric):
    """Tabla LaTeX de sensibilidad a cada hiperparámetro."""
    label, _ = METRICS[metric]
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Sensibilidad de {_latex_escape(label)} a cada hiperparámetro '
        f'de {_latex_escape(alg)} (Kruskal-Wallis sobre el efecto marginal; '
        f'$\\varepsilon^2$: 0.01 pequeño, 0.06 medio, 0.14 grande).}}',
        f'\\label{{tab:hp_effects_{alg.lower()}}}',
        r'\begin{tabular}{lccc}', r'\toprule',
        r'Hiperparámetro & $H$ & $p$ & $\varepsilon^2$ \\', r'\midrule',
    ]
    for f in sorted(factors, key=lambda x: -effects.get(x, (0, 1, 0))[2]):
        H, p, eps2 = effects.get(f, (np.nan, np.nan, np.nan))
        lines.append(f'{_latex_escape(FACTOR_LABELS.get(f, f))} & {H:.1f} & '
                     f'{_fmt_p(p)} & {eps2:.4f} \\\\')
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

    effects = {f: kruskal_by_factor(g, f, metric) for f in factors}
    top_f = max(effects, key=lambda f: effects[f][2])
    print(f"  Hiperparámetro dominante: {FACTOR_LABELS.get(top_f, top_f)} "
          f"(ε² = {effects[top_f][2]:.3f})")

    plot_main_effects(g, alg, sub_factors, metric, out_dir, effects,
                      por_combo=por_combo)
    plot_metric_vs_validity(g, alg, metric, out_dir, chosen, sub_factors,
                            por_combo)
    write_effects_table(effects, alg, factors, out_dir, metric)

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
        plot_effect_sizes_overview(per_alg, args.metric, args.out)
        write_selection_summary(per_alg, args.metric, args.out)

    print(f"\n{'='*66}")
    print(f"  ✅ Listo: {args.out}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
