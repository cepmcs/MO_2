"""
Gráficas comparativas entre algoritmos MOO.

Estructura de resultados esperada:
  results/<crossover>_<mutation>/<ALGORITMO>/pop<N>/run_XX/

Dos modos de comparación (mismo suite de gráficas):

  1. Algoritmos (default): superpone los algoritmos entre sí para un combo
     de operadores.  Lee de results/<combo>/<ALGO>/pop{N}/.
     Salida en plots/<combo>/pop{N}/.

  2. Operadores (--operadores): para cada algoritmo, superpone las variantes
     de operadores genéticos.  Lee de results/<combo>/<ALGO>/pop{N}/.
     Salida en plots/operadores/<ALGO>/pop{N}/.

Uso:
    python plot_comparison.py                          # Algoritmos (auto-detecta combo)
    python plot_comparison.py --combo sbx_pm           # Algoritmos con combo específico
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
import matplotlib.patches as mpatches
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
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

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

# Orden en que se presentan los algoritmos en la comparación final.
ALGORITHM_ORDER = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'MOPSO']

# Nombres de presentación para captions.  analisis.py tiene su propio DISPLAY
# con las baselines incluidas; acá alcanza con los algoritmos porque es lo
# único que llega como etiqueta de reporte.
DISPLAY_ALG = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
               'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO'}

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


def _alg_from_output_dir(output_dir):
    """Nombre del algoritmo en un reporte de operadores: el componente que sigue
    a 'operadores' en la ruta de salida.  None si no es un reporte de ese modo."""
    parts = output_dir.replace('\\', '/').rstrip('/').split('/')
    if 'operadores' in parts:
        i = parts.index('operadores')
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


# ─── Carga de datos ──────────────────────────────────────────────────────────

def discover_operator_combos():
    """Lista los combos de operadores bajo results/ (primer nivel de dirs)."""
    combos = []
    if not os.path.isdir(RESULTS_DIR):
        return combos
    skip = {'comparison', 'comparison_operadores'}
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(path) and name not in skip:
            combos.append(name)
    return combos


def discover_algorithms(pop_size, combo=None):
    """Descubre qué algoritmos tienen resultados para un pop_size dado.
    Si combo es None, busca en todos los combos disponibles."""
    if combo:
        base = os.path.join(RESULTS_DIR, combo)
        if not os.path.isdir(base):
            return []
        algorithms = []
        for alg_name in sorted(os.listdir(base)):
            alg_path = os.path.join(base, alg_name)
            if not os.path.isdir(alg_path) or alg_name == "comparison":
                continue
            pop_path = os.path.join(alg_path, f"pop{pop_size}")
            if os.path.isdir(pop_path) and _has_runs(pop_path):
                algorithms.append(alg_name)
        return algorithms
    else:
        algs = set()
        for c in discover_operator_combos():
            algs.update(discover_algorithms(pop_size, combo=c))
        return sorted(algs)


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
    """Carga las métricas por run de una serie.  Acepta tanto un metrics.csv
    agregado a nivel de serie como los metrics.csv individuales de cada run."""
    path = os.path.join(pop_dir, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    all_dfs = []
    for run_dir in sorted(glob.glob(os.path.join(pop_dir, "run_*"))):
        csv_path = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(csv_path):
            all_dfs.append(pd.read_csv(csv_path))
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


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

    # Recolectar handles/labels para leyenda compartida (sin duplicados)
    legend_handles, legend_labels = [], []

    for ax, (ylabel, title, curves) in zip(axes, panels):
        for idx, s in enumerate(series):
            if s.label not in curves:
                continue
            gens, vals = curves[s.label]
            line, = ax.plot(gens, vals, color=get_color(s.color_key, idx),
                            linewidth=1.2, label=s.label, zorder=3)
            if s.label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(s.label)
        ax.set_xlabel('Generación')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f'{suptitle}',
                 fontsize=14, fontweight='bold', y=1.02)
    # Leyenda única compartida al pie de la figura
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center',
                   ncol=len(legend_labels), framealpha=0.9, edgecolor='#cccccc',
                   fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
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
    ('mean_fsp3', 'Fsp3 (↑)',    True),
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

        bp = ax.boxplot(data, tick_labels=['']*len(labels), patch_artist=True,
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

    fig.suptitle(f'{suptitle}',
                 fontsize=14, fontweight='bold', y=1.02)
                 
    legend_handles = []
    legend_labels = []
    # Usamos series original para que la leyenda esté en el orden de definición
    for idx, s in enumerate(series):
        color = get_color(s.color_key, idx)
        patch = mpatches.Patch(color=color, alpha=0.6, label=s.label)
        legend_handles.append(patch)
        legend_labels.append(s.label)
        
    if legend_handles:
        fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center',
                   ncol=len(legend_labels), framealpha=0.9, edgecolor='#cccccc',
                   fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")




def _df_to_F(df):
    """Convierte DataFrame con qed, sa, fsp3 a matriz F de minimización."""
    return np.array([[-r['qed'], r['sa'], -r['fsp3']] for _, r in df.iterrows()])


def _compute_non_dominated(df):
    """Recalcula el frente no-dominado de un DataFrame con qed, sa, fsp3."""
    required = {'qed', 'sa', 'fsp3'}
    if not required.issubset(df.columns) or df.empty:
        return df
    F = _df_to_F(df)
    front_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return df.iloc[front_idx].reset_index(drop=True)


def _front_bounds(pf_F):
    """Ideal (mínimos) y escala (nadir − ideal) del frente de referencia.

    Se usan para normalizar los objetivos a [0,1] antes de IGD+ y ε+, de modo
    que los tres objetivos pesen por igual (sin esto, SA domina por su rango
    mayor).  Las dimensiones constantes (escala 0) se fijan a 1 para no dividir
    por cero: tras normalizar quedan en 0 y no afectan las distancias."""
    ideal = pf_F.min(axis=0)
    nadir = pf_F.max(axis=0)
    scale = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return ideal, scale


def _normalize_F(F, ideal, scale):
    """Normaliza una matriz de objetivos de minimización a [0,1] usando los
    bounds (ideal/nadir) del frente de referencia."""
    return (F - ideal) / scale


def _additive_epsilon(F, pf):
    """Additive Epsilon indicator (manual, pymoo 0.6 no lo incluye).

    ε+ = max_j  min_i  max_k  (F_i_k - PF_j_k)

    Mide el mínimo desplazamiento uniforme necesario para que F domine
    a todo el frente de referencia PF.  Menor es mejor (0 = iguala PF).

    F y pf deben estar ya normalizados con los mismos bounds (ver
    _front_bounds / _normalize_F) para que las coordenadas sean comparables.
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
    # Normalización a [0,1] con los bounds del frente de referencia, idéntica
    # para IGD+ y ε+, de modo que los objetivos pesen por igual.
    ideal, scale = _front_bounds(pf_F)
    pf_n = _normalize_F(pf_F, ideal, scale)
    igd_plus_ind = IGDPlus(pf_n)

    for s in series:
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            required = {'qed', 'sa', 'fsp3'}
            if df.empty or not required.issubset(df.columns):
                continue
            F_run = _normalize_F(_df_to_F(df), ideal, scale)
            rows.append({
                'run': os.path.basename(run_dir),
                'igd_plus': float(igd_plus_ind(F_run)),
                'epsilon': _additive_epsilon(F_run, pf_n),
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


# Los tres planos del espacio de objetivos, en el orden en que se encadenan
# mejor: los dos primeros comparten el eje QED y los dos últimos el eje Fsp3,
# así cada panel comparte un eje con el que tiene al lado.
PARETO_PLANES = [('qed', 'sa'), ('qed', 'fsp3'), ('sa', 'fsp3')]


def _pad_lim(values, frac=0.08):
    """Rango de una serie de valores con un margen, para que los puntos del
    borde no queden pegados al marco."""
    lo, hi = min(values), max(values)
    pad = (hi - lo) * frac if hi > lo else 0.05
    return lo - pad, hi + pad


def _plane_limits(combined_paretos, xcol, ycol):
    """Límites de un plano sobre TODAS las series.  Cuando la figura se parte en
    filas, cada una tiene que dibujarse con estos límites: si cada fila se
    auto-escalara a lo suyo, dos frentes de extensión distinta ocuparían el
    mismo marco y la comparación entre filas sería un espejismo."""
    xs, ys = [], []
    for df in combined_paretos.values():
        if xcol in df.columns and ycol in df.columns:
            xs.extend(df[xcol].values)
            ys.extend(df[ycol].values)
    if not xs:
        return None
    return _pad_lim(xs), _pad_lim(ys)


def _plot_pareto_plane(ax, series_order, combined_paretos, counts, xcol, ycol,
                       lims=None, con_titulo=True):
    """Dibuja un plano del frente: las series superpuestas más un marcador
    'pastel' donde una misma molécula fue hallada por dos o más series.
    Con lims dibuja en esos límites; sin ellos, auto-escala a sus datos.
    Devuelve los handles de leyenda, en el orden de las series."""
    handles = []
    all_x, all_y = [], []
    coord_colors = {}   # (xr, yr) → colores (uno por molécula) que caen ahí
    for idx, s in series_order:
        df = combined_paretos[s.label]
        if xcol not in df.columns or ycol not in df.columns:
            continue
        color = get_color(s.color_key, idx)
        sc = ax.scatter(df[xcol], df[ycol], c=color, marker=PARETO_MARKER,
                        s=45, alpha=1.0,
                        edgecolors='white', linewidths=0.4,
                        label=f'{s.label} ({counts[s.label]})', zorder=3)
        handles.append(sc)
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

    xlabel = OBJECTIVE_LABELS.get(xcol, xcol)
    ylabel = OBJECTIVE_LABELS.get(ycol, ycol)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if con_titulo:
        ax.set_title(f'{xlabel} vs {ylabel}')

    if lims is not None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    elif all_x and all_y:
        ax.set_xlim(*_pad_lim(all_x))
        ax.set_ylim(*_pad_lim(all_y))

    return handles


def _combined_pareto_fronts(series):
    """Frente de Pareto global de cada serie: junta las moléculas de todas sus
    runs, elimina SMILES duplicados y recalcula la no-dominancia sobre el total.
    Devuelve (combined_paretos, series_order, counts)."""
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

    counts = {label: len(df) for label, df in combined_paretos.items()}
    return combined_paretos, series_order, counts


# ─── Contribución al frente no dominado conjunto ─────────────────────────────
#
#   El hipervolumen mide la extensión del frente, no la calidad de lo que hay
#   dentro: un frente puede ganar volumen estirándose hacia un extremo aunque el
#   grueso de sus soluciones esté dominado.  Para separar las dos cosas se junta
#   lo que produjeron todos los combos, se recalcula la no-dominancia global y
#   se mira quién aportó los supervivientes.  Es dominancia de Pareto pura sobre
#   los tres objetivos: no hay umbrales ni ponderaciones de por medio.

# Okabe-Ito: naranja/azul se distinguen bajo los tres tipos de daltonismo.
CRUCE_COLORS = {'PCX': '#D55E00', 'SBX': '#0072B2'}
COMPARTIDA_COLOR = '#7F7F7F'


def _familia(label):
    """Familia de cruce de un combo: 'pcx_gauss' → 'PCX'."""
    return label.split('_')[0].upper()


def _por_serie(label):
    """Agrupación trivial: cada serie es su propio grupo.  Es la que se usa al
    comparar algoritmos, donde no hay familias que agregar."""
    return label


def _grupos_de(series, grupo_de):
    """Nombres de grupo en el orden de las series, sin repetir."""
    return list(dict.fromkeys(grupo_de(s.label) for s in series))


def _etiqueta_compartida(grupos):
    return 'ambas' if len(grupos) == 2 else 'compartida'


def atribuir_frente(series, pf_df, grupo_de=_familia):
    """Marca qué series produjeron cada molécula del frente conjunto.

    build_reference_front deduplica por SMILES quedándose con la primera
    aparición, así que la fila que sobrevive arrastra el orden en que se
    concatenaron las series y no sirve para atribuir.  Hay que volver a mirar
    cada serie: una misma molécula puede haber sido hallada por varias, y
    contarla como exclusiva de una sería inventar una diferencia.

    grupo_de decide sobre qué se agrega: por familia de cruce al comparar
    operadores, por serie al comparar algoritmos.

    Agrega una columna booleana 'en_<label>' por serie, otra por grupo, y
    'origen' con el grupo que la halló en exclusiva o la marca de compartida.
    """
    out = pf_df.copy()
    for s in series:
        df = load_pareto_molecules(s.pop_dir)
        smiles = set(df['smiles']) if not df.empty else set()
        out[f'en_{s.label}'] = out['smiles'].isin(smiles)

    grupos = _grupos_de(series, grupo_de)
    for g in grupos:
        cols = [f'en_{s.label}' for s in series if grupo_de(s.label) == g]
        out[f'en_{g}'] = out[cols].any(axis=1)

    if len(grupos) > 1:
        cuenta = out[[f'en_{g}' for g in grupos]].sum(axis=1)
        # idxmax sobre las booleanas devuelve el primer grupo que la halló; solo
        # se usa cuando hay exactamente uno, así que no hay desempate que hacer.
        unico = out[[f'en_{g}' for g in grupos]].idxmax(axis=1).str.slice(3)
        out['origen'] = np.where(cuenta > 1, _etiqueta_compartida(grupos), unico)
    return out


def _perfil(df):
    """Descriptores del subconjunto que aporta un operador al frente conjunto:
    dónde cae y qué calidad tiene lo que aporta."""
    if df.empty:
        return {'n': 0, 'fsp3': np.nan, 'fsp3_alto': np.nan,
                'qed': np.nan, 'qed_bajo': np.nan, 'sa': np.nan}
    return {'n': len(df),
            'fsp3': float(df.fsp3.mean()),
            'fsp3_alto': float((df.fsp3 > 0.9).mean()),
            'qed': float(df.qed.mean()),
            'qed_bajo': float((df.qed < 0.60).mean()),
            'sa': float(df.sa.mean())}


def contribucion_agregada(series, pf_df, grupo_de=_familia):
    """Cuánto aporta cada serie —y cada grupo, si agrupan varias— al frente
    conjunto, sobre la unión de las 20 semillas.

    'aporta' cuenta toda molécula hallada por esa serie (compartidas incluidas,
    así que las columnas no suman el total) y 'exclusiva' solo las que no
    encontró ninguna otra."""
    at = atribuir_frente(series, pf_df, grupo_de)
    total = len(at)
    filas = []

    def fila(nombre, mask, excl_mask):
        return {'nombre': nombre, 'total': total,
                'aporta': int(mask.sum()),
                'frac': float(mask.mean()) if total else np.nan,
                'exclusiva': int(excl_mask.sum()),
                **_perfil(at[mask])}

    n_series = [f'en_{s.label}' for s in series]
    for s in series:
        col = f'en_{s.label}'
        otras = at[[c for c in n_series if c != col]].any(axis=1)
        filas.append(fila(s.label, at[col], at[col] & ~otras))

    # Las filas de grupo solo agregan información si agrupan más de una serie.
    grupos = _grupos_de(series, grupo_de)
    if 1 < len(grupos) < len(series):
        for g in grupos:
            col = f'en_{g}'
            otras = at[[f'en_{o}' for o in grupos if o != g]].any(axis=1)
            filas.append(fila(g, at[col], at[col] & ~otras))
    return filas, at


def contribucion_por_semilla(series, grupo_de=_familia):
    """Lo mismo pero dentro de cada semilla: los frentes de la misma semilla
    compiten entre sí y se recalcula la no-dominancia ahí.

    Da un valor por semilla y por grupo, o sea bloques que admiten un test de
    rangos con signo o de Friedman.  El agregado mide otra cosa —todo contra
    todo, 20 veces más candidatos— así que los dos porcentajes no tienen por
    qué coincidir.

    Devuelve dict grupo → array con el % de aportes exclusivos por semilla, el
    % de moléculas compartidas, y las semillas usadas.
    """
    grupos = _grupos_de(series, grupo_de)
    compartida = _etiqueta_compartida(grupos)
    por_serie = {}
    for s in series:
        df = load_pareto_molecules(s.pop_dir)
        if not df.empty:
            por_serie[s.label] = df

    runs = sorted(set().union(*(set(d['run']) for d in por_serie.values())))
    acum = {g: [] for g in grupos}
    compartidas = []
    for run in runs:
        trozos = []
        for label, df in por_serie.items():
            t = df[df['run'] == run].copy()
            if t.empty:
                continue
            t['grupo'] = grupo_de(label)
            trozos.append(t)
        if not trozos:
            continue
        junto = pd.concat(trozos, ignore_index=True)
        # Una molécula puede venir de varios grupos: se resuelve por SMILES
        # antes de la no-dominancia para no contarla dos veces.
        marca = junto.groupby('smiles')['grupo'].agg(
            lambda v: compartida if len(set(v)) > 1 else next(iter(set(v))))
        unico = junto.drop_duplicates('smiles').set_index('smiles')
        unico['origen'] = marca
        frente = _compute_non_dominated(unico.reset_index())
        if frente.empty:
            continue
        n = len(frente)
        for g in grupos:
            acum[g].append(100 * (frente['origen'] == g).sum() / n)
        compartidas.append(100 * (frente['origen'] == compartida).sum() / n)

    return ({g: np.array(v) for g, v in acum.items()},
            np.array(compartidas), runs)


def plot_frente_conjunto(series, pop_size, output_dir, pf_df, grupo_de=_familia):
    """El frente no dominado conjunto en los tres planos, cada molécula pintada
    según quién la aportó (familia de cruce al comparar operadores, algoritmo al
    comparar algoritmos).

    Todos los puntos van en un único scatter con un array de colores: si se
    dibujara un grupo después de otro, el último taparía a los anteriores en la
    zona densa y la figura mostraría una diferencia de orden de dibujo en vez de
    una diferencia real."""
    at = atribuir_frente(series, pf_df, grupo_de)
    if 'origen' not in at.columns:
        return
    grupos = _grupos_de(series, grupo_de)
    compartida = _etiqueta_compartida(grupos)
    # Los combos de operadores no están en COLORS y caerían todos al mismo color
    # del ciclo por defecto; los algoritmos sí tienen color propio asignado.
    paleta = ({g: CRUCE_COLORS[g] for g in grupos} if set(grupos) <= set(CRUCE_COLORS)
              else {g: get_color(g, i) for i, g in enumerate(grupos)})
    paleta[compartida] = COMPARTIDA_COLOR

    orden = [g for g in list(grupos) + [compartida] if (at['origen'] == g).any()]
    cuentas = {g: int((at['origen'] == g).sum()) for g in orden}
    colores = at['origen'].map(paleta).values

    n = len(PARETO_PLANES)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.8))
    for ax, (xcol, ycol) in zip(np.atleast_1d(axes), PARETO_PLANES):
        ax.scatter(at[xcol], at[ycol], c=colores, marker=PARETO_MARKER,
                   s=45, edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_xlabel(OBJECTIVE_LABELS.get(xcol, xcol))
        ax.set_ylabel(OBJECTIVE_LABELS.get(ycol, ycol))
        ax.set_title(f'{OBJECTIVE_LABELS.get(xcol, xcol)} vs '
                     f'{OBJECTIVE_LABELS.get(ycol, ycol)}')
        ax.set_xlim(*_pad_lim(at[xcol].values))
        ax.set_ylim(*_pad_lim(at[ycol].values))

    # 'solo X' y no 'X' a secas: estas cuentas son exclusivas, mientras que la
    # columna «Aporta» de la tabla incluye las compartidas.
    handles = [mpatches.Patch(
        facecolor=paleta[g], edgecolor='white',
        label=f'{g if g == compartida else "solo " + str(g)} ({cuentas[g]})')
        for g in orden]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               framealpha=0.9, edgecolor='#cccccc', fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    alg = _alg_from_output_dir(output_dir)
    por = 'familia de cruce' if set(grupos) <= set(CRUCE_COLORS) else 'algoritmo'
    titulo = f'Frente no dominado conjunto por {por}'
    fig.suptitle(titulo + (f' - {alg}' if alg else ''),
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])
    fname = f"frente_conjunto_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# Elevación de la proyección isométrica: arctan(1/√2).  Con proyección
# ortográfica y azimut múltiplo impar de 45° los tres ejes quedan igual de
# escorzados, así que ninguno se lee privilegiado sobre los otros.
ISO_ELEV = math.degrees(math.atan(1 / math.sqrt(2)))

# Azimuts separados 90°: se leen como una rotación del mismo objeto.  Medidos
# sobre el frente conjunto, -45° es el de mayor dispersión proyectada y +135° el
# de menor oclusión entre puntos.
VISTAS_3D = [-45, 45, 135]


def plot_frente_conjunto_3d(series, pop_size, output_dir, pf_df,
                            grupo_de=_familia, vistas=VISTAS_3D):
    """El frente no dominado conjunto en 3D, desde varios azimuts.

    Complementa a los planos 2D: las proyecciones muestran bien los pares de
    objetivos pero aplastan la superficie, y acá lo que interesa es su forma.

    Proyección ortográfica en todas las vistas: con la perspectiva por defecto
    cada panel parece estar a una distancia distinta y los tamaños dejan de ser
    comparables entre sí.
    """
    at = atribuir_frente(series, pf_df, grupo_de)
    if 'origen' not in at.columns:
        return
    grupos = _grupos_de(series, grupo_de)
    compartida = _etiqueta_compartida(grupos)
    paleta = ({g: CRUCE_COLORS[g] for g in grupos} if set(grupos) <= set(CRUCE_COLORS)
              else {g: get_color(g, i) for i, g in enumerate(grupos)})
    paleta[compartida] = COMPARTIDA_COLOR

    orden = [g for g in list(grupos) + [compartida] if (at['origen'] == g).any()]
    cuentas = {g: int((at['origen'] == g).sum()) for g in orden}
    colores = at['origen'].map(paleta).values

    ncols = len(vistas)
    fig = plt.figure(figsize=(6.2 * ncols, 6.0))
    for i, azim in enumerate(vistas, start=1):
        ax = fig.add_subplot(1, ncols, i, projection='3d')
        ax.set_proj_type('ortho')
        # Un único scatter: matplotlib ordena por profundidad dentro de la
        # llamada, así que la superposición refleja la geometría y no el orden
        # en que se dibujaron los grupos.
        ax.scatter(at['qed'], at['sa'], at['fsp3'], c=colores,
                   marker=PARETO_MARKER, s=26, depthshade=False,
                   edgecolors='white', linewidths=0.25)
        ax.view_init(elev=ISO_ELEV, azim=azim)
        # La caja ortográfica deja aire alrededor de la nube, pero el zoom saca
        # las etiquetas del eje z fuera de su panel: por encima de ~1.05 se
        # montan sobre el panel vecino y la del último se sale del lienzo.
        ax.set_box_aspect(None, zoom=1.05)
        ax.set_xlabel(OBJECTIVE_LABELS.get('qed', 'qed'), labelpad=8)
        ax.set_ylabel(OBJECTIVE_LABELS.get('sa', 'sa'), labelpad=8)
        ax.set_zlabel(OBJECTIVE_LABELS.get('fsp3', 'fsp3'), labelpad=8)
        ax.tick_params(labelsize=8.5, pad=1)
        ax.grid(True)

    handles = [mpatches.Patch(
        facecolor=paleta[g], edgecolor='white',
        label=f'{g if g == compartida else "solo " + str(g)} ({cuentas[g]})')
        for g in orden]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               framealpha=0.9, edgecolor='#cccccc', fontsize=11,
               bbox_to_anchor=(0.5, 0.015))

    alg = _alg_from_output_dir(output_dir)
    por = 'familia de cruce' if set(grupos) <= set(CRUCE_COLORS) else 'algoritmo'
    fig.suptitle(f'Frente no dominado conjunto por {por}'
                 + (f' - {alg}' if alg else ''),
                 fontsize=14, fontweight='bold', y=0.98)
    # Separación amplia entre paneles: las etiquetas del eje z de uno se meten
    # sobre el panel vecino si se dejan pegados.
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.11, top=0.92,
                        wspace=1.25 / ncols)
    fname = f"frente_conjunto_3d_pop{pop_size}.png"
    # Sin bbox_inches='tight': los ejes 3D dibujan sus etiquetas por su cuenta y
    # el recorte automático las deja fuera.  Los márgenes se fijan arriba.
    plt.savefig(os.path.join(output_dir, fname), dpi=180)
    plt.close(fig)
    print(f"  ✓ {fname}")


def plot_pareto_comparison(series, pop_size, output_dir, groups=None):
    """Superpone frentes de Pareto combinados (todas las runs) de cada serie,
    un panel por cada plano de objetivos.

    Con groups —lista de (nombre de fila, [labels])— la figura se parte en una
    fila por grupo en vez de apilar todas las series en el mismo panel.  Las
    filas se dibujan con límites comunes para que se puedan comparar entre sí.
    """
    combined_paretos, series_order, counts = _combined_pareto_fronts(series)
    if not combined_paretos:
        print("  ⚠ Sin datos de Pareto para comparación")
        return

    if groups:
        filas = [(nombre, [(idx, s) for idx, s in series_order
                           if s.label in labels])
                 for nombre, labels in groups]
        filas = [(nombre, so) for nombre, so in filas if so]
    else:
        filas = [(None, series_order)]

    # Con una sola fila cada panel se auto-escala a sus datos, como siempre;
    # con varias, todas comparten los límites de su plano.
    lims = ({p: _plane_limits(combined_paretos, *p) for p in PARETO_PLANES}
            if len(filas) > 1 else {})

    n = len(PARETO_PLANES)
    fig, axes = plt.subplots(len(filas), n, squeeze=False,
                             figsize=(6.4 * n, 5.8 * len(filas)))
    handles = []
    for r, (nombre, fila) in enumerate(filas):
        for c, plano in enumerate(PARETO_PLANES):
            h = _plot_pareto_plane(axes[r][c], fila, combined_paretos, counts,
                                   *plano, lims=lims.get(plano),
                                   con_titulo=(r == 0))
            if c == 0:
                handles.extend(h)
        if nombre:
            axes[r][0].annotate(nombre, xy=(0, 0.5), xytext=(-62, 0),
                                xycoords='axes fraction',
                                textcoords='offset points',
                                ha='center', va='center', rotation=90,
                                fontsize=15, fontweight='bold')

    # Una sola leyenda al pie: las series y sus tamaños son los mismos en todos
    # los paneles, lo que cambia es el par de objetivos (y la fila, si se parte).
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   framealpha=0.9, edgecolor='#cccccc', fontsize=11,
                   bbox_to_anchor=(0.5, 0.01))

    title = 'Frentes de Pareto Globales'
    alg = _alg_from_output_dir(output_dir)
    if alg:
        title += f' - {alg}'

    fig.suptitle(title,
                 fontsize=14, fontweight='bold', y=1.0)
    # El aire para leyenda y título es una altura fija, no una fracción: al
    # partir en filas la figura crece y el porcentaje reservaría de más.
    margen = 0.09 / len(filas)
    plt.tight_layout(rect=[0, margen, 1, 1 - margen / 3])
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
            'best_qed', 'best_sa', 'n_pareto', 'time_sec']
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
    """Media por run de QED, SA y Fsp3 sobre el frente de Pareto final
    (molecules.csv).  Retorna dict[label] → DataFrame con columnas
    [run, mean_qed, mean_sa, mean_fsp3].

    Nota: usa la MEDIA del frente final, no el mejor valor individual."""
    results = {}
    for s in series:
        rows = []
        for run_dir in sorted(glob.glob(os.path.join(s.pop_dir, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            if df.empty or not {'qed', 'sa', 'fsp3'}.issubset(df.columns):
                continue
            rows.append({
                'run': os.path.basename(run_dir),
                'mean_qed': df['qed'].mean(),
                'mean_sa': df['sa'].mean(),
                'mean_fsp3': df['fsp3'].mean(),
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
            '^': r'\textasciicircum{}',
            '×': r'$\times$'}   # etiquetas de presupuesto: 400×250
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

    # pop_size llega como etiqueta del reporte (el algoritmo al comparar
    # operadores, 'final' al comparar algoritmos), no siempre como tamaño de
    # población: solo tiene sentido anunciarlo como N si es un número.
    sufijo = f' ($N={pop_size}$)' if str(pop_size).isdigit() else ''

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption}{sufijo}}}',
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
      - mean_qed/mean_sa/mean_fsp3 → medias del frente final (molecules.csv)
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
        if col in ('mean_qed', 'mean_sa', 'mean_fsp3'):
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
      2) indicadores químicos (QED, SA, Fsp3 medios, Validez, Unicidad, Novedad)
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
        ('Tiempo (s)',       'time_sec',    '.1f', False),
    ]
    chem_cfg = [
        ('QED',      'mean_qed',      '.4f', True),
        ('SA',       'mean_sa',       '.2f', False),
        ('Fsp3', 'mean_fsp3', '.4f', True),
        ('Validez',  'validity',      '.4f', True),
        ('Unicidad', 'uniqueness',    '.4f', True),
        ('Novedad',  'novelty',       '.4f', True),
    ]

    # El contexto sale de pop_size, que es la etiqueta del reporte: el nombre
    # del algoritmo al comparar operadores y 'final' al comparar algoritmos.
    # Antes se derivaba del directorio padre, que con la estructura actual da
    # 'plots' y no dice nada.  El \label mantiene la forma anterior para no
    # romper referencias ya escritas.
    ctx = str(pop_size)
    cap_ctx = ('' if ctx == 'final'
               else f' — {_latex_escape(DISPLAY_ALG.get(ctx, ctx))}')
    dir_ctx = os.path.basename(os.path.dirname(output_dir))
    lab_ctx = '' if dir_ctx == 'comparison' else f'_{dir_ctx.lower()}'

    _write_latex_comparison_table(
        series, col_values, multiobj_cfg,
        f'Comparación de indicadores multiobjetivo{cap_ctx}',
        f'tab:comparison_multiobjective{lab_ctx}_pop{pop_size}',
        output_dir, f'comparison_multiobjective_pop{pop_size}.tex', pop_size)
    # El alcance no es el mismo en las seis columnas y conviene decirlo: QED, SA
    # y Fsp3 son del frente acumulado, validez y novedad de todas las
    # evaluaciones, y unicidad solo de la última generación.
    _write_latex_comparison_table(
        series, col_values, chem_cfg,
        f'Comparación de indicadores químicos{cap_ctx}.  QED, SA y Fsp3 son la '
        f'media del frente no dominado acumulado sobre la corrida completa; '
        f'validez y novedad se calculan sobre las evaluaciones de toda la '
        f'corrida; unicidad corresponde a la población de la última generación',
        f'tab:comparison_chemical{lab_ctx}_pop{pop_size}',
        output_dir, f'comparison_chemical_pop{pop_size}.tex', pop_size)




def plot_pareto_qed_sa_grid(series, pop_size, output_dir):
    """Genera UNA imagen con un panel por serie (los 5 separados, no
    superpuestos), mostrando solo el frente de Pareto QED vs SA.
    Cada serie combina las moléculas de sus runs, elimina duplicados de
    SMILES y recalcula el frente no-dominado global.
    Color: plasma según nº de runs en que aparece cada molécula."""
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

    # Normalización global: 1 run → violeta, max_runs → amarillo
    global_max_runs = max(nr for _, _, nr in paretos)

    n_plots = len(paretos)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 5.5 * nrows),
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.flatten()

    norm = mcolors.Normalize(vmin=1, vmax=max(global_max_runs, 2))
    sc = None
    for ax, (s, pareto, n_runs) in zip(axes_flat, paretos):
        qed = pareto['qed'].values
        sa  = pareto['sa'].values
        n_appeared = pareto['n_runs_appeared'].values
        sc = ax.scatter(qed, sa, c=n_appeared, cmap='plasma', norm=norm,
                        s=90, alpha=0.85,
                        edgecolors='#333333', linewidths=0.4, zorder=3)
        ax.set_xlabel('QED (↑)', fontsize=11)
        ax.set_ylabel('SA (↓)', fontsize=11)
        ax.set_title(f'{s.label}  ({n_runs} ejecuciones, {len(pareto)} no-dom.)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25, color='grey')

    for ax in axes_flat[n_plots:]:
        ax.axis('off')

    # Colorbar compartida horizontal abajo
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='horizontal',
                            shrink=0.6, pad=0.06, aspect=35)
        cbar.set_label('Nº de ejecuciones en que aparece', fontsize=11)

    title = 'Frentes de Pareto QED vs SA por algoritmo'
    alg = _alg_from_output_dir(output_dir)
    if alg:
        title = f'Frentes de Pareto QED vs SA por operador - {alg}'

    fig.suptitle(title,
                 fontsize=14, fontweight='bold')
    fname = f"pareto_qed_sa_grid_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# Etiqueta legible por objetivo (con dirección de optimización)
OBJECTIVE_LABELS = {
    'qed':      'QED (↑)',
    'sa':       'SA (↓)',
    'fsp3': 'Fsp3 (↑)',
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
    # Misma normalización a [0,1] que en los indicadores por-run.
    ideal, scale = _front_bounds(pf_F)
    pf_n = _normalize_F(pf_F, ideal, scale)
    igd_plus_ind = IGDPlus(pf_n)

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
                                 usecols=['gen', 'qed', 'sa', 'fsp3', 'valid'])
            except Exception:
                continue
            df = df[df['valid'].astype(bool)].dropna(subset=['qed', 'sa', 'fsp3'])
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
                F_g = _normalize_F(_df_to_F(front), ideal, scale)
                rows.append({
                    'gen': g,
                    'igd_plus': float(igd_plus_ind(F_g)),
                    'epsilon': _additive_epsilon(F_g, pf_n),
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
    #     QED, SA, Fsp3 como promedio de objetivo por generación).
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
            (f"Promedio de {OBJECTIVE_LABELS.get('fsp3', 'Fsp3')}",
             f"Convergencia de {OBJECTIVE_LABELS.get('fsp3', 'Fsp3')}",
             _objective_curves(series, 'fsp3')),
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
        print("📊 Boxplots químicos (QED, SA, Fsp3, Validez, Unicidad, Novedad)...")
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



    # 9. Grid QED vs SA: los N algoritmos separados en una sola imagen
    print("🧩 Grid QED vs SA por algoritmo (una imagen)...")
    plot_pareto_qed_sa_grid(series, pop_size, output_dir)


# ─── Construcción de series por modo ────────────────────────────────────────

def build_algorithm_series(algorithms, pop_size, combo):
    """Modo algoritmos: una serie por algoritmo para un combo de operadores."""
    series = []
    for alg in algorithms:
        pop_dir = os.path.join(RESULTS_DIR, combo, alg, f"pop{pop_size}")
        series.append(Series(alg, pop_dir, color_key=alg))
    return series


def build_operator_series(alg, pop_size, combos):
    """Modo operadores: para un algoritmo, una serie por combo de operadores."""
    series = []
    for combo in combos:
        pop_dir = os.path.join(RESULTS_DIR, combo, alg, f"pop{pop_size}")
        if _has_runs(pop_dir):
            series.append(Series(combo, pop_dir, color_key=combo))
    return series


def build_finalist_series(algorithms, finalistas_dir):
    """Modo algoritmos sobre finalistas/<ALG>/run_XX/: la configuración elegida
    de cada algoritmo tras las etapas 1 y 2."""
    series = []
    for alg in algorithms:
        d = os.path.join(finalistas_dir, alg)
        if _has_runs(d):
            series.append(Series(alg, d, color_key=alg))
    return series


def build_operator_series_winners(alg, winners_dir):
    """Modo operadores sobre winners/<ALG>/<combo>/<config>/: las configuraciones
    que ganaron su bloque en la etapa 1.  Cada combo ganó con hiperparámetros
    distintos, así que el nivel de configuración se resuelve con glob."""
    series = []
    for combo in sorted(os.listdir(os.path.join(winners_dir, alg))):
        combo_dir = os.path.join(winners_dir, alg, combo)
        if not os.path.isdir(combo_dir):
            continue
        cfgs = [d for d in sorted(glob.glob(os.path.join(combo_dir, '*')))
                if _has_runs(d)]
        if cfgs:
            series.append(Series(combo, cfgs[0], color_key=combo))
    return series


# ─── Main ────────────────────────────────────────────────────────────────────

def run_algorithm_comparison(algorithms, pop_size, combo=None, finalistas=None):
    """Comparación entre algoritmos.

    Con finalistas se lee la configuración elegida de cada algoritmo desde
    finalistas/<ALG>/; sin él, el grid original bajo results/<combo>/<ALG>/pop<N>/."""
    if finalistas:
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
        return

    if combo is None:
        combos = discover_operator_combos()
        if not combos:
            print(f"No se encontraron combos de operadores en {RESULTS_DIR}")
            return
        elif len(combos) == 1:
            combo = combos[0]
            print(f"  Auto-detectado combo: {combo}")
        else:
            print(f"Múltiples combos encontrados: {', '.join(combos)}")
            print("Especificá uno con --combo <nombre>")
            return

    if algorithms is None:
        algorithms = discover_algorithms(pop_size, combo=combo)
        if not algorithms:
            print(f"No se encontraron resultados para combo={combo}, "
                  f"pop_size={pop_size}")
            combo_dir = os.path.join(RESULTS_DIR, combo)
            if os.path.isdir(combo_dir):
                print(f"Directorios en {combo_dir}:")
                for d in sorted(os.listdir(combo_dir)):
                    full = os.path.join(combo_dir, d)
                    if os.path.isdir(full):
                        pops = [p for p in os.listdir(full)
                                if os.path.isdir(os.path.join(full, p))]
                        print(f"  {d}: {pops}")
            return

    print(f"\n{'='*60}")
    print("  Comparación entre algoritmos")
    print(f"  Combo: {combo}")
    print(f"  Algoritmos: {', '.join(algorithms)}")
    print(f"  pop_size: {pop_size}")
    print(f"{'='*60}")

    series = build_algorithm_series(algorithms, pop_size, combo)
    output_dir = os.path.join(PLOTS_DIR, combo, f"pop{pop_size}")
    _generate_report(series, pop_size, output_dir,
                     f"Comparación — {combo} — Todos los Algoritmos")

    print(f"\n{'='*60}")
    print(f"  ✅ Generación completa: {output_dir}")
    print(f"{'='*60}\n")


def run_operator_comparison(algorithms, pop_size, winners_dir=None):
    """Comparación de variantes de operadores, un reporte por algoritmo.

    Con winners_dir se leen las configuraciones ganadoras de la etapa 1
    (winners/<ALG>/<combo>/<config>/), cada una con sus propios hiperparámetros;
    sin él, el grid original bajo results/<combo>/<ALG>/pop<N>/."""
    if winners_dir:
        if not os.path.isdir(winners_dir):
            print(f"No existe {winners_dir}")
            return
        algorithms = algorithms or sorted(
            d for d in os.listdir(winners_dir)
            if os.path.isdir(os.path.join(winners_dir, d)))
        tag = "winners"
    else:
        combos = discover_operator_combos()
        if len(combos) < 2:
            print("Se necesitan ≥2 combos de operadores para comparar.")
            print(f"Combos encontrados: {combos if combos else '(ninguno)'}")
            return
        if algorithms is None:
            algorithms = discover_algorithms(pop_size)
            if not algorithms:
                print(f"No se encontraron algoritmos con resultados "
                      f"para pop_size={pop_size}")
                return
        tag = f"pop{pop_size}"

    print(f"\n{'='*60}")
    print("  Comparación de operadores (por algoritmo)")
    print(f"  Origen: {winners_dir or RESULTS_DIR}")
    print(f"  Algoritmos candidatos: {', '.join(algorithms)}")
    print(f"{'='*60}")

    generated = []
    for alg in algorithms:
        series = (build_operator_series_winners(alg, winners_dir) if winners_dir
                  else build_operator_series(alg, pop_size, combos))
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
    parser = argparse.ArgumentParser(description="Gráficas comparativas MOO")
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help="Algoritmos a comparar (auto-detecta si no se especifica)")
    parser.add_argument('--pop_size', type=int, default=200,
                        help="Tamaño de población (default: 200)")
    parser.add_argument('--combo', default=None,
                        help="Combo de operadores (ej: sbx_pm). "
                             "Auto-detecta si solo hay uno.")
    parser.add_argument('--operadores', action='store_true',
                        help="Compara variantes de operadores por algoritmo")
    parser.add_argument('--winners', nargs='?',
                        const=os.path.join(ROOT_DIR, 'resultados', 'winners'),
                        default=None,
                        help="Con --operadores: lee las configuraciones ganadoras "
                             "de la etapa 1 desde resultados/winners/<ALG>/<combo>/<config>/")
    parser.add_argument('--finalistas', nargs='?',
                        const=os.path.join(ROOT_DIR, 'resultados', 'finalistas'),
                        default=None,
                        help="Comparación final: lee la configuración elegida de "
                             "cada algoritmo desde resultados/finalistas/<ALG>/")
    args = parser.parse_args()

    if args.operadores:
        run_operator_comparison(args.algorithms, args.pop_size, args.winners)
    else:
        run_algorithm_comparison(args.algorithms, args.pop_size, args.combo,
                                 args.finalistas)


if __name__ == "__main__":
    main()

