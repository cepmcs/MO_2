"""
Gráficas del post-procesamiento.  Importar este módulo fija el estilo global de
matplotlib.
"""

import io
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from . import datos

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

# Los algoritmos tienen color propio; las demás series (operadores, baselines)
# caen al ciclo por índice.
COLORS = {
    'NSGA2':   '#000000',
    'MOPSO':   '#FF0000',
    'AGEMOEA': '#008000',
    'MOEAD':   '#1F77B4',
    'NSGA3':   '#7B1FA2',
}
DEFAULT_COLORS = ['#000000', '#FF0000', '#008000', '#1F77B4', '#7B1FA2', '#8C564B']

OPERATOR_COLORS = {
    'pcx/pm':    '#1F4E79',
    'pcx/gauss': '#6FA8DC',
    'sbx/pm':    '#B45F06',
    'sbx/gauss': '#F6B26B',
}
# MOPSO no tiene operadores; se colorea por inercia, su factor dominante.
W_COLORS = {0.4: '#9ECAE1', 0.6: '#4292C6', 0.9: '#08519C'}

# Las series se distinguen por color, no por forma.
PARETO_MARKER = 'o'

# (columna, etiqueta, mayor_es_mejor); espejan las tablas de comparación.
BOXPLOT_MO = [
    ('hypervolume', 'Hipervolumen (↑)',  True),
    ('spacing',     'Espaciamiento (↓)', False),
    ('igd_plus',    'IGD+ (↓)',          False),
    ('epsilon',     'ε+ Aditivo (↓)',    False),
    ('n_pareto',    'Tamaño de Pareto',  True),
]
BOXPLOT_CHEM = [
    ('mean_qed',      'QED (↑)',         True),
    ('mean_sa',       'SA (↓)',          False),
    ('mean_lipinski', 'Lipinski (↑)',    True),
    ('validity',      'Tasa de Validez', True),
    ('uniqueness',    'Unicidad (↑)',    True),
    ('novelty',       'Novedad (↑)',     True),
]


def get_color(key, idx=0):
    return COLORS.get(key, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def _guardar(fig, out_dir, fname, dpi=200):
    plt.savefig(os.path.join(out_dir, fname), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


def _leyenda_al_pie(fig, handles, labels):
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               framealpha=0.9, edgecolor='#cccccc', fontsize=11,
               bbox_to_anchor=(0.5, -0.02))


def _grilla(n, ancho, alto, **kwargs):
    """Ejes en grilla de hasta 3 columnas, ya aplanados."""
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ancho * ncols, alto * nrows),
                             squeeze=False, **kwargs)
    return fig, axes.flatten()


# ─── Convergencia ────────────────────────────────────────────────────────────

def convergence_grid(series, out_dir, panels, fname, suptitle):
    """Un panel por curva.  panels: [(ylabel, título, {label: (gens, vals)})]."""
    panels = [p for p in panels if p[2]]
    if not panels:
        print(f"  ⚠ {fname}: sin datos de convergencia")
        return

    fig, axes = _grilla(len(panels), 6, 5.5)
    handles, labels = [], []
    for ax, (ylabel, title, curvas) in zip(axes, panels):
        for idx, s in enumerate(series):
            if s.label not in curvas:
                continue
            gens, vals = curvas[s.label]
            linea, = ax.plot(gens, vals, color=get_color(s.color_key, idx),
                             linewidth=1.2, label=s.label, zorder=3)
            if s.label not in labels:
                handles.append(linea)
                labels.append(s.label)
        ax.set_xlabel('Generación')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    if handles:
        _leyenda_al_pie(fig, handles, labels)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _guardar(fig, out_dir, fname)


# ─── Boxplots ────────────────────────────────────────────────────────────────

def boxplots(series, out_dir, get_values, configs, fname, suptitle):
    """Distribución entre runs de cada métrica, una caja por serie."""
    disponibles = [c for c in configs
                   if any(get_values(s.label, c[0]) is not None for s in series)]
    if not disponibles:
        print(f"  ⚠ {fname}: sin métricas con datos")
        return

    fig, axes = _grilla(len(disponibles), 5, 6)
    for ax, (col, label, _) in zip(axes, disponibles):
        data, colores = [], []
        for idx, s in enumerate(series):
            vals = get_values(s.label, col)
            if vals is not None and len(vals):
                data.append(vals)
                colores.append(get_color(s.color_key, idx))
        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=[''] * len(data), patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=6))
        for caja, color in zip(bp['boxes'], colores):
            caja.set_facecolor(color)
            caja.set_alpha(0.6)
        for mediana in bp['medians']:
            mediana.set_color('black')
            mediana.set_linewidth(2)
        ax.set_ylabel(label)
        ax.set_title(label)

    for ax in axes[len(disponibles):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    _leyenda_al_pie(
        fig,
        [mpatches.Patch(color=get_color(s.color_key, i), alpha=0.6)
         for i, s in enumerate(series)],
        [s.label for s in series])
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _guardar(fig, out_dir, fname)


# ─── Frentes de Pareto ───────────────────────────────────────────────────────

def _marcador_compartido(ax, x, y, colores, size):
    """Marcador circular partido en sectores, para los puntos donde coinciden
    moléculas de varias series."""
    n = len(colores)
    for i, color in enumerate(colores):
        t = np.linspace(2 * np.pi * i / n, 2 * np.pi * (i + 1) / n, 30)
        xs = np.concatenate([[0], np.cos(t)])
        ys = np.concatenate([[0], np.sin(t)])
        ax.scatter([x], [y], marker=np.column_stack([xs, ys]), s=size,
                   facecolor=color, edgecolors='white', linewidths=0.3, zorder=5)


def pareto_comparison(series, out_dir, titulo):
    """Frentes globales de todas las series superpuestos en QED vs SA."""
    frentes = []
    for idx, s in enumerate(series):
        frente = datos.frente_global(s.path)
        if not frente.empty:
            frentes.append((idx, s, frente))
    if not frentes:
        print("  ⚠ sin datos de Pareto para la comparación")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    xs, ys = [], []
    colores_por_punto = {}
    for idx, s, frente in frentes:
        color = get_color(s.color_key, idx)
        ax.scatter(frente['qed'], frente['sa'], c=color, marker=PARETO_MARKER,
                   s=45, edgecolors='white', linewidths=0.4,
                   label=f'{s.label} ({len(frente)})', zorder=3)
        xs.extend(frente['qed'].values)
        ys.extend(frente['sa'].values)
        for xv, yv in zip(frente['qed'].round(4), frente['sa'].round(4)):
            colores_por_punto.setdefault((xv, yv), []).append(color)

    for (xv, yv), cols in colores_por_punto.items():
        unicos = list(dict.fromkeys(cols))
        if len(unicos) >= 2:
            _marcador_compartido(ax, xv, yv, unicos, size=58)

    ax.set_xlabel('QED (↑)')
    ax.set_ylabel('SA (↓)')
    ax.set_title('QED (↑) vs SA (↓)')
    for lim, vals in ((ax.set_xlim, xs), (ax.set_ylim, ys)):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 if hi > lo else 0.05
        lim(lo - pad, hi + pad)
    ax.legend(framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _guardar(fig, out_dir, "pareto_comparison.png")


def pareto_qed_sa_grid(series, out_dir, titulo):
    """Un panel por serie con su frente global.  El color indica en cuántas
    ejecuciones apareció cada molécula."""
    frentes = []
    for s in series:
        df = datos.load_pareto_molecules(s.path)
        if df.empty:
            continue
        apariciones = df.groupby('smiles')['run'].nunique().rename('n_runs_appeared')
        frente = datos.compute_non_dominated(df.drop_duplicates(subset='smiles'))
        if frente.empty:
            continue
        frente = frente.merge(apariciones, on='smiles', how='left')
        frente['n_runs_appeared'] = frente['n_runs_appeared'].fillna(1).astype(int)
        frentes.append((s, frente, df['run'].nunique()))
    if not frentes:
        print("  ⚠ sin datos de Pareto para la grilla QED vs SA")
        return

    fig, axes = _grilla(len(frentes), 6.5, 5.5, constrained_layout=True)
    norm = mcolors.Normalize(vmin=1, vmax=max(2, max(n for _, _, n in frentes)))
    sc = None
    for ax, (s, frente, n_runs) in zip(axes, frentes):
        sc = ax.scatter(frente['qed'], frente['sa'], c=frente['n_runs_appeared'],
                        cmap='plasma', norm=norm, s=90, alpha=0.85,
                        edgecolors='#333333', linewidths=0.4, zorder=3)
        ax.set_xlabel('QED (↑)')
        ax.set_ylabel('SA (↓)')
        ax.set_title(f'{s.label}  ({n_runs} ejecuciones, {len(frente)} no-dom.)',
                     fontsize=12)
    for ax in axes[len(frentes):]:
        ax.axis('off')

    cbar = fig.colorbar(sc, ax=axes.tolist(), orientation='horizontal',
                        shrink=0.6, pad=0.06, aspect=35)
    cbar.set_label('Nº de ejecuciones en que aparece')
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    _guardar(fig, out_dir, "pareto_qed_sa_grid.png", dpi=150)


# ─── Sensibilidad de hiperparámetros (etapa 1) ───────────────────────────────

def _orden_niveles(factor, niveles):
    if factor == 'budget':
        return [b for b in datos.BUDGET_ORDER if b in niveles]
    try:
        return sorted(niveles, key=float)
    except (TypeError, ValueError):
        return sorted(niveles)


def main_effects(g, alg, factors, metric, out_dir, por_combo=True):
    """Un panel por hiperparámetro con la mediana de cada nivel.  En los GA, una
    curva por combinación de operadores.  Δ y p van en effects_<ALG>.tex."""
    label, higher = datos.HP_METRICS[metric]
    fig, axes = _grilla(len(factors), 5.4, 4.6, sharey=True)

    if por_combo:
        g = g.copy()
        g['_combo'] = g['crossover'].astype(str) + '/' + g['mutation'].astype(str)
        curvas = [c for c in datos.HP_COMBOS if c in set(g['_combo'])]
    else:
        curvas = [None]

    for ax, f in zip(axes, factors):
        niveles = _orden_niveles(f, g[f].dropna().unique().tolist())
        x = np.arange(len(niveles))
        for c in curvas:
            gs = g if c is None else g[g['_combo'] == c]
            color = (COLORS.get(alg, '#333333') if c is None
                     else OPERATOR_COLORS.get(c, '#888888'))
            medianas = [np.median(gs.loc[gs[f] == lv, metric].dropna().values)
                        for lv in niveles]
            ax.plot(x, medianas, color=color, linewidth=2, zorder=3,
                    label=c or 'Mediana')
            ax.scatter(x, medianas, s=45, color=color, zorder=4,
                       edgecolors='white', linewidths=1.0)
        ax.set_title(datos.FACTOR_LABELS.get(f, f), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(lv) for lv in niveles], fontsize=10)
        ax.set_ylabel(f'{label} ({"↑" if higher else "↓"})')

    for ax in axes[len(factors):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    vistos = dict(zip(labels, handles))
    orden = [l for l in (curvas if por_combo else labels) if l in vistos]
    orden += [l for l in vistos if l not in orden]
    fig.legend([vistos[l] for l in orden], orden, loc='lower center',
               ncol=len(orden), framealpha=0.9, edgecolor='#cccccc',
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    sub = ' por combinación de operadores' if por_combo else ''
    fig.suptitle(f'Efecto de los hiperparámetros{sub} — {alg}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _guardar(fig, out_dir, f'main_effects_{alg}.png')


def metric_vs_validity(g, alg, metric, out_dir, elegidas, sub_factors, factors,
                       por_combo):
    """Cada configuración como un punto: validez contra la métrica de selección,
    ambas medianas sobre las semillas.  Las elegidas van resaltadas."""
    if metric == 'validity':
        return
    label, _ = datos.HP_METRICS[metric]
    m = g.groupby(factors, observed=True)[[metric, 'validity']].median()

    if por_combo:
        clave = (m.index.get_level_values('crossover').astype(str) + '/'
                 + m.index.get_level_values('mutation').astype(str))
        grupos = [(c, OPERATOR_COLORS[c]) for c in datos.HP_COMBOS
                  if c in set(clave)]
    else:
        clave = m.index.get_level_values('w')
        grupos = [(w, c) for w, c in sorted(W_COLORS.items()) if w in set(clave)]

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for nombre, color in grupos:
        sel = clave == nombre
        ax.scatter(m.loc[sel, 'validity'], m.loc[sel, metric], s=34, alpha=0.65,
                   color=color, linewidths=0,
                   label=nombre if por_combo else f'$w$ = {nombre:g}', zorder=2)

    for nombre, b in elegidas.items():
        cfg = b['cfg'] if isinstance(b['cfg'], tuple) else (b['cfg'],)
        niveles = dict(zip(sub_factors, cfg))
        if nombre:
            niveles['crossover'], niveles['mutation'] = nombre.split('/')
        r = m.loc[tuple(niveles[f] for f in factors)]
        color = (OPERATOR_COLORS[nombre] if por_combo
                 else W_COLORS.get(niveles.get('w'), '#333333'))
        ax.scatter(r['validity'], r[metric], s=190, color=color,
                   edgecolors='black', linewidths=1.6, zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker='o', linestyle='none', markersize=12,
                              markerfacecolor='#bbbbbb', markeredgecolor='black',
                              markeredgewidth=1.6))
    labels.append('Seleccionada')
    leg = ax.legend(handles, labels, framealpha=0.9, edgecolor='#cccccc',
                    fontsize=10, loc='best')
    for lh in leg.legend_handles[:len(grupos)]:
        lh.set_alpha(1.0)

    ax.margins(x=0.06, y=0.06)
    ax.set_xlabel('Validez (fracción de moléculas válidas) →')
    ax.set_ylabel(f'{label} →')
    ax.set_title(f'{label} contra validez — {alg}\n'
                 f'una configuración por punto, mediana de 20 semillas',
                 fontsize=12)
    plt.tight_layout()
    _guardar(fig, out_dir, f'{metric}_vs_validity_{alg}.png')


# ─── Moléculas representativas ───────────────────────────────────────────────

def _render(smiles, size=(420, 320)):
    """PNG de la estructura, como array para matplotlib."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().clearBackground = False
    d.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format='png')


def figura_moleculas(seleccion, out_path, n_moleculas):
    """Una fila por algoritmo con sus moléculas seleccionadas."""
    nrows, ncols = len(seleccion), n_moleculas
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for i, (alg, sel) in enumerate(seleccion.items()):
        for j in range(ncols):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
            if j >= len(sel):
                ax.set_visible(False)
                continue
            m = sel.iloc[j]
            img = _render(m['smiles'])
            if img is not None:
                ax.imshow(img)
            ax.set_xlabel(f"QED {m['qed']:.3f}   ·   SA {m['sa']:.2f}",
                          fontsize=9.5, labelpad=3)
            if j == 0:
                ax.set_ylabel(datos.display(alg), fontsize=13,
                              fontweight='bold', labelpad=10)

    fig.suptitle(f'Las {n_moleculas} moléculas de mayor QED del frente de cada '
                 f'algoritmo', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {out_path}")
