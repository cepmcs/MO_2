"""
Gráficas comparativas entre algoritmos MOO.
Genera automáticamente todas las gráficas estándar de la literatura.

Layout en disco: cada combinación de operadores es una carpeta uniforme bajo
results/<combo>/<ALGO>/pop{N}/, donde sbx_pm es el combo base.

Dos modos de comparación (mismo suite de gráficas):

  1. Algoritmos (default): superpone los algoritmos entre sí, leyendo el combo
     base de results/sbx_pm/<ALGO>/pop{N}/.  Salida en results/comparison/pop{N}/.

  2. Operadores (--operadores): para cada algoritmo, superpone las variantes
     de operadores genéticos (sbx+pm vs. pcx_gauss vs. ...), leyendo todos los
     combos de results/<combo>/<ALGO>/pop{N}/.  Salida en
     results/comparison/operadores/<ALGO>/pop{N}/.

Uso:
    python plot_comparison.py                          # Algoritmos (auto-detecta)
    python plot_comparison.py --algorithms NSGA2 MOPSO AGEMOEA
    python plot_comparison.py --pop_size 200
    python plot_comparison.py --operadores             # Operadores por algoritmo
    python plot_comparison.py --operadores --algorithms NSGA2 NSGA3
"""

import os, argparse, glob, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.igd_plus import IGDPlus

# ─── Configuración visual ────────────────────────────────────────────────────
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
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
BASELINE_COMBO = "sbx_pm"                              # combo canónico (base)
BASELINE_DIR = os.path.join(RESULTS_DIR, BASELINE_COMBO)

# Paleta de colores distinguibles.  Para algoritmos se usa el nombre como clave;
# para operadores (claves no presentes aquí) se cae al ciclo DEFAULT por índice.
COLORS = {
    'NSGA2':   '#000000',   # Negro
    'MOPSO':   '#FF0000',   # Rojo 100%
    'AGEMOEA': '#008000',   # Verde
    'MOEAD':   '#1F77B4',   # Azul
    'NSGA3':   '#7B1FA2',   # Violeta
}
DEFAULT_COLORS = ['#000000', '#FF0000', '#008000', '#1F77B4', '#7B1FA2', '#8C564B']

# Todas las series usan el mismo marcador (punto): se distinguen por color,
# no por forma.
PARETO_MARKER = 'o'


def get_color(key, idx=0):
    return COLORS.get(key, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ─── Serie a comparar ────────────────────────────────────────────────────────

class Series:
    """Una serie a comparar: una curva/caja/frente en las gráficas.

    En el modo algoritmos cada algoritmo es una serie; en el modo operadores
    cada variante de operadores (de un mismo algoritmo) es una serie.

      label     → texto en leyendas y títulos.
      pop_dir   → ruta absoluta a .../pop{N} de donde se cargan los datos.
      color_key → clave para color/marcador (default: label).
    """
    def __init__(self, label, pop_dir, color_key=None):
        self.label = label
        self.pop_dir = pop_dir
        self.color_key = color_key if color_key is not None else label


def _has_runs(pop_dir):
    """True si pop_dir contiene al menos un run con convergence.csv."""
    return bool(glob.glob(os.path.join(pop_dir, "run_*", "convergence.csv")))


# ─── Carga de datos ──────────────────────────────────────────────────────────

def discover_algorithms(pop_size, base_dir=None):
    """Descubre qué algoritmos tienen resultados para un pop_size dado,
    dentro de un combo concreto (default: el combo base)."""
    if base_dir is None:
        base_dir = BASELINE_DIR
    algorithms = []
    if not os.path.exists(base_dir):
        return algorithms
    for alg_name in sorted(os.listdir(base_dir)):
        alg_path = os.path.join(base_dir, alg_name)
        if not os.path.isdir(alg_path) or alg_name == "comparison":
            continue
        pop_path = os.path.join(alg_path, f"pop{pop_size}")
        if os.path.isdir(pop_path) and _has_runs(pop_path):
            algorithms.append(alg_name)
    return algorithms


def discover_combos():
    """Lista los combos de operadores (results/<combo>/), con el base primero."""
    combos = []
    if not os.path.isdir(RESULTS_DIR):
        return combos
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(path) and name != "comparison":
            combos.append(name)
    if BASELINE_COMBO in combos:   # el base siempre primero
        combos.remove(BASELINE_COMBO)
        combos.insert(0, BASELINE_COMBO)
    return combos


def load_convergence_data(pop_dir):
    """Carga convergence.csv de todas las runs de una serie.
    Retorna DataFrame con columnas: run, gen, hv, validity, uniqueness, ..."""
    all_dfs = []
    for run_dir in sorted(glob.glob(os.path.join(pop_dir, "run_*"))):
        csv_path = os.path.join(run_dir, "convergence.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['run'] = os.path.basename(run_dir)
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def load_metrics(pop_dir):
    """Carga metrics.csv de una serie."""
    path = os.path.join(pop_dir, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_pareto_molecules(pop_dir):
    """Carga molecules.csv de todas las runs de una serie."""
    all_dfs = []
    for run_dir in sorted(glob.glob(os.path.join(pop_dir, "run_*"))):
        csv_path = os.path.join(run_dir, "molecules.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['run'] = os.path.basename(run_dir)
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# ─── Gráficas ────────────────────────────────────────────────────────────────

def _smooth(values, window):
    """Media móvil para suavizar curvas de convergencia."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def _conv_csv_curves(series, metric):
    """Curva de convergencia de una métrica de convergence.csv.
    Devuelve {label: (gens, mean_suavizada)} (media sobre runs)."""
    curves = {}
    for s in series:
        df = load_convergence_data(s.pop_dir)
        if df.empty or metric not in df.columns:
            print(f"  ⚠ {s.label}: sin datos de '{metric}'")
            continue
        grouped = df.groupby('gen')[metric].mean().reset_index()
        curves[s.label] = (grouped['gen'].values,
                           _smooth(grouped[metric].values, 20))
    return curves


def _objective_curves(series, objective):
    """Curva de convergencia del promedio de un objetivo (all_molecules.csv.gz).
    Devuelve {label: (gens, mean_suavizada)} (media sobre runs)."""
    curves = {}
    for s in series:
        all_means = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            gz_path = os.path.join(run_dir, "all_molecules.csv.gz")
            if not os.path.exists(gz_path):
                continue
            try:
                df = pd.read_csv(gz_path, usecols=['gen', objective])
                all_means.append(df.groupby('gen')[objective].mean())
            except Exception:
                continue
        if not all_means:
            print(f"  ⚠ {s.label}: sin datos de '{objective}' en all_molecules.csv.gz")
            continue
        mean_over_runs = pd.concat(all_means, axis=1).mean(axis=1)
        curves[s.label] = (mean_over_runs.index.values,
                           _smooth(mean_over_runs.values, 20))
    return curves


def _plot_convergence_grid(series, pop_size, output_dir, panels, fname, suptitle):
    """Dibuja una grilla de paneles de convergencia (3 por fila).
    panels: lista de (ylabel, title, curves) donde
            curves = {label: (gens, vals)}."""
    panels = [p for p in panels if p[2]]   # descarta paneles sin datos
    if not panels:
        print(f"  ⚠ {fname}: sin datos de convergencia")
        return

    n_plots = len(panels)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, (ylabel, title, curves) in zip(axes, panels):
        for idx, s in enumerate(series):
            if s.label not in curves:
                continue
            gens, vals = curves[s.label]
            ax.plot(gens, vals, color=get_color(s.color_key, idx),
                    linewidth=1.2, label=s.label, zorder=3)
        ax.set_xlabel('Generación')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(framealpha=0.9, edgecolor='#cccccc')

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f'{suptitle} (pop={pop_size})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# Configuración de boxplots separada por tipo de indicador, espejando las
# tablas de comparación.  Cada entrada: (columna, etiqueta, higher_better).
BOXPLOT_MO_CONFIGS = [
    ('hypervolume', 'Hipervolumen (↑)',  True),
    ('spacing',     'Espaciamiento (↓)', False),
    ('igd_plus',    'IGD+ (↓)',          False),
    ('epsilon',     'ε+ Aditivo (↓)',    False),
    ('n_pareto',    'Tamaño de Pareto',  True),
]
BOXPLOT_CHEM_CONFIGS = [
    ('mean_qed',      'QED (↑)',         True),
    ('mean_sa',       'SA (↓)',          False),
    ('mean_lipinski', 'Lipinski (↑)',    True),
    ('validity',      'Tasa de Validez', True),
    ('uniqueness',    'Unicidad (↑)',    True),
    ('novelty',       'Novedad (↑)',     True),
]


def plot_boxplots(series, pop_size, output_dir, get_values, plot_configs,
                  fname, suptitle):
    """Boxplots comparativos de un grupo de métricas finales.
    get_values(label, col) → array de valores per-run (o None).
    plot_configs: lista de (col, etiqueta, higher_better)."""
    available = [(col, label, hb) for col, label, hb in plot_configs
                 if any(get_values(s.label, col) is not None for s in series)]
    if not available:
        print(f"  ⚠ {fname}: sin métricas con datos")
        return

    n_plots = len(available)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, (col, label, _) in zip(axes, available):
        data, labels, colors = [], [], []
        for idx, s in enumerate(series):
            vals = get_values(s.label, col)
            if vals is not None and len(vals):
                data.append(vals)
                labels.append(s.label)
                colors.append(get_color(s.color_key, idx))

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=6))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        ax.set_ylabel(label)
        ax.set_title(label)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f'{suptitle} (pop={pop_size})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")




def _df_to_F(df):
    """Convierte DataFrame con qed, sa, lipinski a matriz F de minimización."""
    return np.array([[-r['qed'], r['sa'], -r['lipinski']] for _, r in df.iterrows()])


def _compute_non_dominated(df):
    """Recalcula el frente no-dominado de un DataFrame con qed, sa, lipinski."""
    required = {'qed', 'sa', 'lipinski'}
    if not required.issubset(df.columns) or df.empty:
        return df
    F = _df_to_F(df)
    front_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return df.iloc[front_idx].reset_index(drop=True)


def _additive_epsilon(F, pf):
    """Additive Epsilon indicator (manual, pymoo 0.6 no lo incluye).

    ε+ = max_j  min_i  max_k  (F_i_k - PF_j_k)

    Mide el mínimo desplazamiento uniforme necesario para que F domine
    a todo el frente de referencia PF.  Menor es mejor (0 = iguala PF).
    """
    # F  : (n, m)  frente obtenido
    # pf : (p, m)  frente de referencia
    # Para cada punto j del PF, buscar el punto i de F que lo "cubre" mejor
    eps_per_ref = []
    for j in range(len(pf)):
        # max_k (F_i_k - PF_j_k) para cada i
        diff = F - pf[j]           # (n, m)
        worst_obj = diff.max(axis=1)  # (n,)  peor exceso por punto de F
        eps_per_ref.append(worst_obj.min())  # mejor cobertura para este punto PF
    return float(np.max(eps_per_ref))  # peor caso sobre todo el PF


# ─── Frente de referencia combinado e indicadores ────────────────────────────

def build_reference_front(series, pop_size):
    """Construye frente de Pareto de referencia combinando todas las runs
    de todas las series.  Retorna (pf_F, pf_df) donde pf_F es la
    matriz de objetivos de minimización y pf_df el DataFrame con SMILES.

    Procedimiento estándar cuando el frente verdadero es desconocido:
    juntar todas las soluciones no-dominadas, eliminar duplicados,
    recalcular NDS global."""
    all_dfs = []
    for s in series:
        df = load_pareto_molecules(s.pop_dir)
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset='smiles', keep='first')
    pf_df = _compute_non_dominated(combined)
    if pf_df.empty:
        return None, None
    pf_F = _df_to_F(pf_df)
    return pf_F, pf_df


def compute_indicators_per_run(series, pop_size, pf_F):
    """Computa IGD+ y ε+ para cada run de cada serie.
    Retorna dict[label] → DataFrame con columnas [run, igd_plus, epsilon]."""
    results = {}
    igd_plus_ind = IGDPlus(pf_F)

    for s in series:
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            required = {'qed', 'sa', 'lipinski'}
            if df.empty or not required.issubset(df.columns):
                continue
            F_run = _df_to_F(df)
            rows.append({
                'run': os.path.basename(run_dir),
                'igd_plus': float(igd_plus_ind(F_run)),
                'epsilon': _additive_epsilon(F_run, pf_F),
            })
        if rows:
            results[s.label] = pd.DataFrame(rows)
    return results


def _pie_marker(ax, x, y, colors, size):
    """Dibuja en (x, y) un marcador circular dividido en sectores iguales,
    uno por color, para señalar que ahí coinciden moléculas de varias series."""
    n = len(colors)
    for i, color in enumerate(colors):
        t = np.linspace(2 * np.pi * i / n, 2 * np.pi * (i + 1) / n, 30)
        xs = np.concatenate([[0], np.cos(t)])
        ys = np.concatenate([[0], np.sin(t)])
        ax.scatter([x], [y], marker=np.column_stack([xs, ys]),
                   s=size, facecolor=color, edgecolors='white',
                   linewidths=0.3, zorder=5)


def plot_pareto_comparison(series, pop_size, output_dir):
    """Superpone frentes de Pareto combinados (todas las runs) de cada serie.
    Junta las moléculas de las 20 runs, elimina duplicados, y recalcula
    el frente no-dominado global por serie.
    Usa diferentes formas de marcadores por serie y auto-escala ejes."""
    combined_paretos = {}     # label → DataFrame
    series_order = []         # preserva orden e info de color
    for idx, s in enumerate(series):
        df_all = load_pareto_molecules(s.pop_dir)
        if df_all.empty:
            continue
        # Eliminar SMILES duplicados, quedarse con la mejor versión
        df_unique = df_all.drop_duplicates(subset='smiles', keep='first')
        # Recalcular frente no-dominado global
        pareto = _compute_non_dominated(df_unique)
        if not pareto.empty:
            combined_paretos[s.label] = pareto
            series_order.append((idx, s))

    if not combined_paretos:
        print("  ⚠ Sin datos de Pareto para comparación")
        return

    # Contar moléculas por serie para el título
    counts = {label: len(df) for label, df in combined_paretos.items()}

    panels = [
        ('qed', 'sa',       'QED (↑)',      'SA (↓)'),
        ('qed', 'lipinski', 'QED (↑)',      'Lipinski (↑)'),
        ('sa',  'lipinski', 'SA (↓)',       'Lipinski (↑)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, panels):
        all_x, all_y = [], []
        coord_colors = {}   # (xr, yr) → colores (uno por molécula) que caen ahí
        for idx, s in series_order:
            df = combined_paretos[s.label]
            if xcol not in df.columns or ycol not in df.columns:
                continue
            color = get_color(s.color_key, idx)
            ax.scatter(df[xcol], df[ycol], c=color, marker=PARETO_MARKER,
                       s=22, alpha=1.0,
                       edgecolors='white', linewidths=0.4,
                       label=f'{s.label} ({counts[s.label]})', zorder=3)
            all_x.extend(df[xcol].values)
            all_y.extend(df[ycol].values)
            for xv, yv in zip(df[xcol].round(4), df[ycol].round(4)):
                coord_colors.setdefault((xv, yv), []).append(color)

        # Donde coinciden moléculas de 2+ series (colores distintos), superponer
        # un marcador "pastel" con los colores presentes.
        for (xv, yv), cols in coord_colors.items():
            uniq = list(dict.fromkeys(cols))
            if len(uniq) >= 2:
                _pie_marker(ax, xv, yv, uniq, size=58)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')

        # Auto-escalar ejes al rango de datos con padding para ver bien el frente
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 0.05
            y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.05
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.legend(framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(f'Frentes de Pareto Combinados — Todas las Runs (pop={pop_size})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f"pareto_comparison_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")




def generate_statistical_table(series, pop_size, output_dir,
                               indicator_data=None):
    """Genera tabla resumen (media ± std) de todas las métricas por serie.
    Incluye métricas per-run (metrics.csv) e indicadores basados en frente de
    referencia (IGD+, ε+) si están disponibles."""
    if indicator_data is None:
        indicator_data = {}

    all_metrics = {}     # label → DataFrame
    for s in series:
        df = load_metrics(s.pop_dir)
        if not df.empty:
            # Merge indicadores si existen para esta serie
            if s.label in indicator_data:
                df_ind = indicator_data[s.label].copy()
                if len(df_ind) == len(df):
                    df = df.copy()
                    for ind_col in ['igd_plus', 'epsilon']:
                        if ind_col in df_ind.columns:
                            df[ind_col] = df_ind[ind_col].values
            all_metrics[s.label] = df

    if len(all_metrics) < 2:
        print("  ⚠ Se necesitan ≥2 series para tabla estadística")
        return

    cols = ['hypervolume', 'spacing', 'validity', 'novelty',
            'igd_plus', 'epsilon',
            'best_qed', 'best_sa', 'n_pareto']
    cols = [c for c in cols if all(c in df.columns for df in all_metrics.values())]

    # Generar CSV resumen
    rows = []
    for label, df in all_metrics.items():
        row = {'series': label, 'n_runs': len(df)}
        for col in cols:
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_std']  = df[col].std()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, f"comparison_summary_pop{pop_size}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ comparison_summary_pop{pop_size}.csv")


def _compute_chemical_means(series):
    """Media por run de QED, SA y Lipinski sobre el frente de Pareto final
    (molecules.csv).  Retorna dict[label] → DataFrame con columnas
    [run, mean_qed, mean_sa, mean_lipinski].

    Nota: usa la MEDIA del frente final, no el mejor valor individual."""
    results = {}
    for s in series:
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            if df.empty or not {'qed', 'sa', 'lipinski'}.issubset(df.columns):
                continue
            rows.append({
                'run': os.path.basename(run_dir),
                'mean_qed': df['qed'].mean(),
                'mean_sa': df['sa'].mean(),
                'mean_lipinski': df['lipinski'].mean(),
            })
        if rows:
            results[s.label] = pd.DataFrame(rows)
    return results


def _compute_uniqueness(series):
    """Uniqueness de la última generación por run, leída de convergence.csv
    (la uniqueness por-generación que ya trackea GenerationTracker).
    Representa la diversidad de la población final.
    Retorna dict[label] → DataFrame con columnas [run, uniqueness]."""
    results = {}
    for s in series:
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            conv_path = os.path.join(run_dir, "convergence.csv")
            if not os.path.exists(conv_path):
                continue
            df = pd.read_csv(conv_path)
            if df.empty or 'uniqueness' not in df.columns:
                continue
            rows.append({
                'run': os.path.basename(run_dir),
                'uniqueness': float(df['uniqueness'].iloc[-1]),
            })
        if rows:
            results[s.label] = pd.DataFrame(rows)
    return results


def _latex_escape(s):
    """Escapa caracteres especiales de LaTeX en texto (p. ej. el guion bajo
    de nombres de operadores como pcx_gauss → pcx\\_gauss)."""
    repl = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}'}
    return ''.join(repl.get(c, c) for c in str(s))


def _write_latex_comparison_table(series, col_values, metrics_cfg,
                                  caption, tex_label, output_dir, fname,
                                  pop_size):
    """Escribe una tabla LaTeX comparando algoritmos (filas) por métricas
    (columnas), con media ± std.  Resalta en negrita el mejor por columna.

    col_values(label, col) → np.array de valores per-run (o None).
    metrics_cfg: lista de (header, col, fmt, higher_better).
    """
    # Conserva solo métricas con datos en al menos una serie
    cols = [(h, c, f, hb) for (h, c, f, hb) in metrics_cfg
            if any(col_values(s.label, c) is not None for s in series)]
    if not cols:
        print(f"  ⚠ {fname}: sin métricas con datos")
        return

    # Precalcula media y std (muestral) por serie/columna
    means, stds = {}, {}
    for s in series:
        for _, col, _, _ in cols:
            vals = col_values(s.label, col)
            if vals is not None and len(vals):
                means[(s.label, col)] = float(np.mean(vals))
                stds[(s.label, col)] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Mejor serie por columna (según dirección de optimización)
    best = {}
    for _, col, _, hb in cols:
        cand = [(s.label, means[(s.label, col)]) for s in series
                if (s.label, col) in means]
        if cand:
            best[col] = (max if hb else min)(cand, key=lambda t: t[1])[0]

    arrow = lambda hb: r'$\uparrow$' if hb else r'$\downarrow$'
    col_spec = 'l' + 'c' * len(cols)
    header_cells = ['Algoritmo'] + [f'{h} {arrow(hb)}' for h, _, _, hb in cols]

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption} ($N={pop_size}$)}}',
        f'\\label{{{tex_label}}}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        ' & '.join(header_cells) + r' \\',
        r'\midrule',
    ]
    for s in series:
        cells = [_latex_escape(s.label)]
        for _, col, fmt, _ in cols:
            if (s.label, col) not in means:
                cells.append('--')
                continue
            m, sd = means[(s.label, col)], stds[(s.label, col)]
            body = f'{m:{fmt}} \\pm {sd:{fmt}}'
            cell = f'$\\mathbf{{{body}}}$' if best.get(col) == s.label else f'${body}$'
            cells.append(cell)
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(output_dir, fname), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓ {fname}")


def _build_series_value_getter(series, indicator_data=None):
    """Construye un getter unificado get(label, col) → array de valores per-run
    (o None), consultando la fuente correcta según la columna:
      - mean_qed/mean_sa/mean_lipinski → medias del frente final (molecules.csv)
      - igd_plus/igd/epsilon           → indicadores vs frente de referencia
      - uniqueness                     → última generación (convergence.csv)
      - resto (hypervolume, spacing, validity, novelty, n_pareto…) → metrics.csv
    Calcula las fuentes una sola vez para reusarse en tablas y boxplots."""
    if indicator_data is None:
        indicator_data = {}
    metrics_by_series = {s.label: load_metrics(s.pop_dir) for s in series}
    chem_by_series = _compute_chemical_means(series)
    uniq_by_series = _compute_uniqueness(series)

    def get(label, col):
        if col in ('mean_qed', 'mean_sa', 'mean_lipinski'):
            df = chem_by_series.get(label)
        elif col in ('igd_plus', 'epsilon'):
            df = indicator_data.get(label)
        elif col == 'uniqueness':
            df = uniq_by_series.get(label)
        else:
            df = metrics_by_series.get(label)
        if df is None or df.empty or col not in df.columns:
            return None
        vals = df[col].dropna().values
        return vals if len(vals) else None

    return get


def generate_latex_comparison_tables(series, pop_size, output_dir, col_values):
    """Genera dos tablas LaTeX de comparación entre algoritmos:
      1) indicadores multiobjetivo (HV, Spacing, IGD+, ε+, Pareto size)
      2) indicadores químicos (QED, SA, Lipinski medios, Validez, Unicidad, Novedad)
    col_values(label, col) → array de valores per-run (de _build_series_value_getter).
    """
    if len(series) < 2:
        return

    multiobj_cfg = [
        ('Hipervolumen',     'hypervolume', '.4f', True),
        ('Espaciamiento',    'spacing',     '.4f', False),
        ('IGD$^+$',          'igd_plus',    '.4f', False),
        (r'$\epsilon^+$',    'epsilon',     '.4f', False),
        ('Tamaño de Pareto', 'n_pareto',    '.1f', True),
    ]
    chem_cfg = [
        ('QED',      'mean_qed',      '.4f', True),
        ('SA',       'mean_sa',       '.2f', False),
        ('Lipinski', 'mean_lipinski', '.4f', True),
        ('Validez',  'validity',      '.4f', True),
        ('Unicidad', 'uniqueness',    '.4f', True),
        ('Novedad',  'novelty',       '.4f', True),
    ]

    # Contexto: en modo operadores output_dir es .../comparison/operadores/<ALG>/pop{N},
    # así que el padre del dir pop es el algoritmo; en modo algoritmos es
    # "comparison".  Se usa para desambiguar caption y \label entre reportes.
    ctx = os.path.basename(os.path.dirname(output_dir))
    cap_ctx = '' if ctx == 'comparison' else f' — {ctx}'
    lab_ctx = '' if ctx == 'comparison' else f'_{ctx.lower()}'

    _write_latex_comparison_table(
        series, col_values, multiobj_cfg,
        f'Comparación de indicadores multiobjetivo{cap_ctx}',
        f'tab:comparison_multiobjective{lab_ctx}_pop{pop_size}',
        output_dir, f'comparison_multiobjective_pop{pop_size}.tex', pop_size)
    _write_latex_comparison_table(
        series, col_values, chem_cfg,
        f'Comparación de indicadores químicos (media del frente final){cap_ctx}',
        f'tab:comparison_chemical{lab_ctx}_pop{pop_size}',
        output_dir, f'comparison_chemical_pop{pop_size}.tex', pop_size)


def plot_pareto_per_series(series, pop_size, output_dir):
    """Para cada serie, combina las moléculas de las 20 runs,
    elimina duplicados de SMILES, recalcula el frente no-dominado global
    y genera la gráfica con colormap verde→azul según el nº de runs en
    que aparece cada molécula (1 run = verde, 20 runs = azul)."""
    # Colormap verde→azul para frecuencia de aparición en runs
    cmap_runs = mcolors.LinearSegmentedColormap.from_list(
        'green_blue', ['#00C853', '#1565C0'])

    for s in series:
        df_all = load_pareto_molecules(s.pop_dir)
        if df_all.empty:
            print(f"  ⚠ {s.label}: sin datos de moléculas")
            continue

        n_runs = df_all['run'].nunique()

        # Contar en cuántas runs aparece cada SMILES (antes de deduplicar)
        run_counts = df_all.groupby('smiles')['run'].nunique().reset_index()
        run_counts.columns = ['smiles', 'n_runs_appeared']

        # Eliminar SMILES duplicados y recalcular frente no-dominado
        df_unique = df_all.drop_duplicates(subset='smiles', keep='first')
        pareto = _compute_non_dominated(df_unique)
        if pareto.empty:
            print(f"  ⚠ {s.label}: frente vacío tras NDS")
            continue

        # Asociar el conteo de runs a cada molécula del frente
        pareto = pareto.merge(run_counts, on='smiles', how='left')
        pareto['n_runs_appeared'] = pareto['n_runs_appeared'].fillna(1).astype(int)

        qed = pareto['qed'].values
        sa  = pareto['sa'].values
        lip = pareto['lipinski'].values
        n_appeared = pareto['n_runs_appeared'].values

        # Normalización: 1 run → 0 (verde), n_runs → 1 (azul)
        norm = mcolors.Normalize(vmin=1, vmax=max(n_runs, 2))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        panels = [
            (qed, sa,  'QED (↑)', 'SA (↓)',       'QED vs SA'),
            (qed, lip, 'QED (↑)', 'Lipinski (↑)', 'QED vs Lipinski'),
            (sa,  lip, 'SA (↓)',  'Lipinski (↑)', 'SA vs Lipinski'),
        ]
        for ax, (x, y, xl, yl, t) in zip(axes, panels):
            sc = ax.scatter(x, y, c=n_appeared, cmap=cmap_runs, norm=norm,
                            s=90, alpha=0.85,
                            edgecolors='#333333', linewidths=0.4, zorder=3)
            ax.set_xlabel(xl, fontsize=11)
            ax.set_ylabel(yl, fontsize=11)
            ax.set_title(t, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.25, color='grey')

        # Colorbar compartida
        cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('Nº de runs en que aparece', fontsize=11)

        # Nombre de archivo seguro (sin '+' ni espacios)
        safe = s.label.replace('+', '').replace(' ', '_')
        fig.suptitle(f'{s.label} | pop={pop_size} | {n_runs} runs combinadas | '
                     f'{len(pareto)} mol no-dominadas',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = f"pareto_{safe}_pop{pop_size}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {fname}  ({n_runs} runs, {len(pareto)} no-dominadas)")


def plot_pareto_qed_sa_grid(series, pop_size, output_dir):
    """Genera UNA imagen con un panel por serie (los 5 separados, no
    superpuestos), mostrando solo el frente de Pareto QED vs SA.
    Cada serie combina las moléculas de sus runs, elimina duplicados de
    SMILES y recalcula el frente no-dominado global.
    Color: verde→azul según nº de runs en que aparece cada molécula."""
    # Colormap verde→azul para frecuencia de aparición en runs
    cmap_runs = mcolors.LinearSegmentedColormap.from_list(
        'green_blue', ['#00C853', '#1565C0'])

    # Recolectar frente por serie
    paretos = []   # (s, pareto_df, n_runs)
    for s in series:
        df_all = load_pareto_molecules(s.pop_dir)
        if df_all.empty:
            continue
        n_runs = df_all['run'].nunique()
        # Contar en cuántas runs aparece cada SMILES
        run_counts = df_all.groupby('smiles')['run'].nunique().reset_index()
        run_counts.columns = ['smiles', 'n_runs_appeared']
        df_unique = df_all.drop_duplicates(subset='smiles', keep='first')
        pareto = _compute_non_dominated(df_unique)
        if pareto.empty or not {'qed', 'sa'}.issubset(pareto.columns):
            continue
        pareto = pareto.merge(run_counts, on='smiles', how='left')
        pareto['n_runs_appeared'] = pareto['n_runs_appeared'].fillna(1).astype(int)
        paretos.append((s, pareto, n_runs))

    if not paretos:
        print("  ⚠ Sin datos de Pareto para grid QED vs SA")
        return

    # Normalización global: 1 run → verde, max_runs → azul
    global_max_runs = max(nr for _, _, nr in paretos)

    n_plots = len(paretos)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    norm = mcolors.Normalize(vmin=1, vmax=max(global_max_runs, 2))
    sc = None
    for ax, (s, pareto, n_runs) in zip(axes, paretos):
        qed = pareto['qed'].values
        sa  = pareto['sa'].values
        n_appeared = pareto['n_runs_appeared'].values
        sc = ax.scatter(qed, sa, c=n_appeared, cmap=cmap_runs, norm=norm,
                        s=90, alpha=0.85,
                        edgecolors='#333333', linewidths=0.4, zorder=3)
        ax.set_xlabel('QED (↑)', fontsize=11)
        ax.set_ylabel('SA (↓)', fontsize=11)
        ax.set_title(f'{s.label}  ({n_runs} runs, {len(pareto)} no-dom.)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25, color='grey')

    for ax in axes[n_plots:]:
        ax.axis('off')

    # Colorbar compartida
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[:n_plots], shrink=0.8, pad=0.02)
        cbar.set_label('Nº de runs en que aparece', fontsize=11)

    fig.suptitle(f'Frentes de Pareto QED vs SA por algoritmo (pop={pop_size})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f"pareto_qed_sa_grid_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# Etiqueta legible por objetivo (con dirección de optimización)
OBJECTIVE_LABELS = {
    'qed':      'QED (↑)',
    'sa':       'SA (↓)',
    'lipinski': 'Lipinski (↑)',
}


def _indicator_curves(series, pop_size, output_dir, pf_F, gen_stride=10):
    """Curvas de convergencia de IGD+ y ε+ por generación, SIN re-entrenar.

    Reconstruye el frente no-dominado de cada generación a partir de
    all_molecules.csv.gz (que guarda la población completa por gen) y mide
    los indicadores contra el frente de referencia combinado pf_F.
    Promedia sobre todas las runs de cada serie y guarda las curvas en CSV.

    gen_stride submuestrea generaciones (1 = todas) para acelerar el cómputo.

    Devuelve {col: {label: (gens, vals_suavizadas)}} para 'igd_plus' y 'epsilon'.
    """
    igd_plus_ind = IGDPlus(pf_F)

    # Acumula curva media por serie: label → DataFrame indexado por gen
    series_curves = {}
    for s in series:
        per_run_curves = []   # cada elemento: DataFrame indexado por gen
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            gz_path = os.path.join(run_dir, "all_molecules.csv.gz")
            if not os.path.exists(gz_path):
                continue
            try:
                df = pd.read_csv(gz_path,
                                 usecols=['gen', 'qed', 'sa', 'lipinski', 'valid'])
            except Exception:
                continue
            df = df[df['valid'].astype(bool)].dropna(subset=['qed', 'sa', 'lipinski'])
            if df.empty:
                continue

            gens = sorted(df['gen'].unique())
            gens = gens[::gen_stride]
            rows = []
            for g in gens:
                df_g = df[df['gen'] == g]
                front = _compute_non_dominated(df_g)
                if front.empty:
                    continue
                F_g = _df_to_F(front)
                rows.append({
                    'gen': g,
                    'igd_plus': float(igd_plus_ind(F_g)),
                    'epsilon': _additive_epsilon(F_g, pf_F),
                })
            if rows:
                per_run_curves.append(pd.DataFrame(rows).set_index('gen'))

        if per_run_curves:
            # Alinea por gen y promedia sobre runs
            series_curves[s.label] = pd.concat(per_run_curves).groupby(level=0).mean()

    if not series_curves:
        print("  ⚠ Sin datos de all_molecules.csv.gz para convergencia de indicadores")
        return {'igd_plus': {}, 'epsilon': {}}

    # Reorganiza a {col: {label: (gens, vals_suavizadas)}}
    out = {'igd_plus': {}, 'epsilon': {}}
    for label, curve in series_curves.items():
        for col in out:
            out[col][label] = (curve.index.values,
                               _smooth(curve[col].values, 5))

    # Guardar las curvas en CSV (formato largo)
    long_rows = []
    for label, curve in series_curves.items():
        for g, row in curve.iterrows():
            long_rows.append({'series': label, 'gen': g, **row.to_dict()})
    csv_path = os.path.join(output_dir, f"convergence_indicators_pop{pop_size}.csv")
    pd.DataFrame(long_rows).to_csv(csv_path, index=False)
    print(f"  ✓ convergence_indicators_pop{pop_size}.csv")

    return out


# ─── Generación de gráficas para un grupo de series ─────────────────────────

def _generate_report(series, pop_size, output_dir, report_label):
    """Genera el conjunto completo de gráficas para un grupo de series."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  {report_label}")
    print(f"  Series: {', '.join(s.label for s in series)}")
    print(f"  Salida: {output_dir}")
    print(f"{'─'*60}\n")

    # 1. Indicadores basados en frente de referencia (IGD+, ε+).
    #    Se computan ANTES de las convergencias/boxplots porque alimentan
    #    tanto la gráfica de convergencia MO como las tablas/boxplots.
    indicator_data = {}
    ind_curves = {'igd_plus': {}, 'epsilon': {}}
    if len(series) >= 2:
        print("📐 Construyendo frente de referencia combinado...")
        pf_F, pf_df = build_reference_front(series, pop_size)
        if pf_F is not None:
            print(f"   Frente de referencia: {len(pf_F)} soluciones no-dominadas")
            indicator_data = compute_indicators_per_run(series, pop_size, pf_F)

            # Guardar frente de referencia
            pf_path = os.path.join(output_dir, f"reference_front_pop{pop_size}.csv")
            pf_df.to_csv(pf_path, index=False)
            print(f"  ✓ reference_front_pop{pop_size}.csv")

            # Guardar indicadores por run
            ind_rows = []
            for label, df_ind in indicator_data.items():
                for _, row in df_ind.iterrows():
                    ind_rows.append({'series': label, **row.to_dict()})
            if ind_rows:
                ind_path = os.path.join(output_dir, f"indicators_pop{pop_size}.csv")
                pd.DataFrame(ind_rows).to_csv(ind_path, index=False)
                print(f"  ✓ indicators_pop{pop_size}.csv")

            # Curvas de convergencia de indicadores por generación (sin re-entrenar)
            print("📈 Curvas de convergencia de indicadores (IGD+, ε+)...")
            ind_curves = _indicator_curves(series, pop_size, output_dir, pf_F)
        else:
            print("  ⚠ No se pudo construir frente de referencia")

    # 2. Convergencia de indicadores multiobjetivo (HV, IGD+, ε+).
    print("📈 Convergencia de indicadores MO (HV, IGD+, ε+)...")
    _plot_convergence_grid(
        series, pop_size, output_dir,
        panels=[
            ('Hipervolumen', 'Convergencia de Hipervolumen (↑)',
             _conv_csv_curves(series, 'hv')),
            ('IGD+ (↓)', 'Convergencia IGD+ (↓)', ind_curves['igd_plus']),
            ('ε+ Aditivo (↓)', 'Convergencia ε+ Aditivo (↓)', ind_curves['epsilon']),
        ],
        fname=f"convergence_mo_pop{pop_size}.png",
        suptitle="Convergencia de Indicadores Multiobjetivo")

    # 3. Convergencia de indicadores químicos
    #    (Validez, Unicidad, Novedad desde convergence.csv;
    #     QED, SA, Lipinski como promedio de objetivo por generación).
    print("📈 Convergencia de indicadores químicos...")
    _plot_convergence_grid(
        series, pop_size, output_dir,
        panels=[
            ('Tasa de Validez', 'Convergencia de Validez',
             _conv_csv_curves(series, 'validity')),
            ('Tasa de Unicidad', 'Convergencia de Unicidad',
             _conv_csv_curves(series, 'uniqueness')),
            ('Tasa de Novedad', 'Convergencia de Novedad',
             _conv_csv_curves(series, 'novelty')),
            (f"Promedio de {OBJECTIVE_LABELS.get('qed', 'QED')}",
             f"Convergencia de {OBJECTIVE_LABELS.get('qed', 'QED')}",
             _objective_curves(series, 'qed')),
            (f"Promedio de {OBJECTIVE_LABELS.get('sa', 'SA')}",
             f"Convergencia de {OBJECTIVE_LABELS.get('sa', 'SA')}",
             _objective_curves(series, 'sa')),
            (f"Promedio de {OBJECTIVE_LABELS.get('lipinski', 'Lipinski')}",
             f"Convergencia de {OBJECTIVE_LABELS.get('lipinski', 'Lipinski')}",
             _objective_curves(series, 'lipinski')),
        ],
        fname=f"convergence_chemical_pop{pop_size}.png",
        suptitle="Convergencia de Indicadores Químicos")

    # 5. Boxplots + tablas (requiere ≥2 series).  Comparten un único getter
    #    de valores per-run (metrics + indicadores + medias químicas + unicidad).
    if len(series) >= 2:
        get_values = _build_series_value_getter(series, indicator_data)

        print("📊 Boxplots multiobjetivo (HV, Espaciamiento, IGD+, ε+, Pareto)...")
        plot_boxplots(series, pop_size, output_dir, get_values, BOXPLOT_MO_CONFIGS,
                      f"boxplots_mo_pop{pop_size}.png",
                      "Distribución de Indicadores Multiobjetivo")
        print("📊 Boxplots químicos (QED, SA, Lipinski, Validez, Unicidad, Novedad)...")
        plot_boxplots(series, pop_size, output_dir, get_values, BOXPLOT_CHEM_CONFIGS,
                      f"boxplots_chemical_pop{pop_size}.png",
                      "Distribución de Indicadores Químicos")

        print("📋 Tabla estadística...")
        generate_statistical_table(series, pop_size, output_dir,
                                   indicator_data=indicator_data)

        print("📋 Tablas LaTeX de comparación...")
        generate_latex_comparison_tables(series, pop_size, output_dir, get_values)

    # 7. Superposición de frentes de Pareto
    if len(series) >= 2:
        print("🔀 Frentes de Pareto superpuestos...")
        plot_pareto_comparison(series, pop_size, output_dir)

    # 8. Frente de Pareto por serie (20 runs combinadas)
    print("🏆 Frente de Pareto por serie (20 runs combinadas)...")
    plot_pareto_per_series(series, pop_size, output_dir)

    # 9. Grid QED vs SA: los N algoritmos separados en una sola imagen
    print("🧩 Grid QED vs SA por algoritmo (una imagen)...")
    plot_pareto_qed_sa_grid(series, pop_size, output_dir)


# ─── Construcción de series por modo ────────────────────────────────────────

def build_algorithm_series(algorithms, pop_size):
    """Modo algoritmos: una serie por algoritmo, leyendo el combo base."""
    series = []
    for alg in algorithms:
        pop_dir = os.path.join(BASELINE_DIR, alg, f"pop{pop_size}")
        series.append(Series(alg, pop_dir, color_key=alg))
    return series


def build_operator_series(alg, pop_size, combos):
    """Modo operadores: para un algoritmo, una serie por combo de operadores.
    Todos los combos viven en results/<combo>/ (sbx_pm es el base)."""
    series = []
    for combo in combos:
        pop_dir = os.path.join(RESULTS_DIR, combo, alg, f"pop{pop_size}")
        if _has_runs(pop_dir):
            label = combo.replace('_', '+')
            series.append(Series(label, pop_dir, color_key=label))
    return series


# ─── Main ────────────────────────────────────────────────────────────────────

def run_algorithm_comparison(algorithms, pop_size):
    """Comparación entre algoritmos (modo default)."""
    if algorithms is None:
        algorithms = discover_algorithms(pop_size)
        if not algorithms:
            print(f"No se encontraron resultados para pop_size={pop_size}")
            print(f"Directorios disponibles en {BASELINE_DIR}:")
            if os.path.exists(BASELINE_DIR):
                for d in os.listdir(BASELINE_DIR):
                    full = os.path.join(BASELINE_DIR, d)
                    if os.path.isdir(full):
                        pops = [p for p in os.listdir(full)
                                if os.path.isdir(os.path.join(full, p))]
                        print(f"  {d}: {pops}")
            return

    print(f"\n{'='*60}")
    print(f"  Comparación entre algoritmos")
    print(f"  Algoritmos: {', '.join(algorithms)}")
    print(f"  pop_size: {pop_size}")
    print(f"{'='*60}")

    series = build_algorithm_series(algorithms, pop_size)
    output_dir = os.path.join(RESULTS_DIR, "comparison", f"pop{pop_size}")
    _generate_report(series, pop_size, output_dir,
                     "Comparación — Todos los Algoritmos")

    print(f"\n{'='*60}")
    print(f"  ✅ Generación completa: {output_dir}")
    print(f"{'='*60}\n")


def run_operator_comparison(algorithms, pop_size):
    """Comparación de combos de operadores, un reporte por algoritmo."""
    combos = discover_combos()
    if not combos:
        print(f"No se encontraron combos de operadores en {RESULTS_DIR}")
        return

    # Algoritmos candidatos: los presentes en cualquier combo.
    if algorithms is None:
        algs = set()
        for combo in combos:
            algs |= set(discover_algorithms(pop_size,
                        base_dir=os.path.join(RESULTS_DIR, combo)))
        algorithms = sorted(algs)

    print(f"\n{'='*60}")
    print(f"  Comparación de operadores (por algoritmo)")
    print(f"  Combos detectados: {', '.join(combos)}")
    print(f"  Algoritmos candidatos: {', '.join(algorithms)}")
    print(f"  pop_size: {pop_size}")
    print(f"{'='*60}")

    generated = []
    for alg in algorithms:
        series = build_operator_series(alg, pop_size, combos)
        if len(series) < 2:
            print(f"\n  ⚠ {alg}: solo {len(series)} combo(s) con datos; "
                  f"se omite (se requieren ≥2 para comparar).")
            continue
        output_dir = os.path.join(RESULTS_DIR, "comparison", "operadores",
                                  alg, f"pop{pop_size}")
        _generate_report(series, pop_size, output_dir,
                         f"Comparación de Operadores — {alg}")
        generated.append((alg, output_dir))

    print(f"\n{'='*60}")
    if generated:
        print(f"  ✅ Reportes de operadores generados:")
        for alg, out in generated:
            print(f"     {alg}: {out}")
    else:
        print(f"  ⚠ No se generó ningún reporte (cada algoritmo necesita "
              f"≥2 combos con datos).")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Gráficas comparativas MOO")
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help="Algoritmos a comparar (auto-detecta si no se especifica)")
    parser.add_argument('--pop_size', type=int, default=200,
                        help="Tamaño de población (default: 200)")
    parser.add_argument('--operadores', action='store_true',
                        help="Compara combos de operadores por algoritmo "
                             "(todos los results/<combo>/)")
    args = parser.parse_args()

    if args.operadores:
        run_operator_comparison(args.algorithms, args.pop_size)
    else:
        run_algorithm_comparison(args.algorithms, args.pop_size)


if __name__ == "__main__":
    main()
