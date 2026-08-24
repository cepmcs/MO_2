"""
Gráficas comparativas entre algoritmos MOO.

Espacio de objetivos: QED (↑) y SA (↓), o sea F = [-QED, SA] en minimización.
Fsp3 NO es objetivo — entra como constraint (Fsp3 ≥ FSP3_MIN), así que el frente
vive en un plano y Fsp3 se reporta como propiedad de lo que el constraint dejó
pasar, nunca como una dimensión más de la dominancia (ver utils_mo).

Los dos modos leen de resultados/, que es lo que baja del cluster (ver la
cabecera de analisis.py):

  1. Algoritmos (default): superpone los cinco algoritmos, cada uno con la
     configuración que quedó elegida tras las etapas 1 y 2.
     Lee resultados/finalistas/<ALG>/run_XX/   →   plots/comparacion_final/.

  2. Operadores (--operadores): para cada algoritmo, superpone las cuatro
     combinaciones de operadores que ganaron su bloque en la etapa 1.
     Lee resultados/winners/<ALG>/<cruce_mutacion>/<config>/run_XX/
        →  plots/operadores/<ALG>/winners/.  Va a un subdirectorio propio para
     no mezclarse con las tablas que analisis.py etapa2 deja un nivel arriba.

Las funciones de carga y de figura de este módulo son además el motor que reusa
analisis.py en sus cuatro etapas.  Una Series es cualquier directorio con
run_XX/{metrics,molecules,convergence}.csv, así que para comparar otras carpetas
—celdas del grid, por ejemplo— alcanza con apuntar --finalistas a un directorio
cuyos hijos sean las series.  Lo que NO se puede es leer el results/ crudo del
cluster: su estructura es <ALG>/<combo>/<config>/run_XX (ver utils_mo.ga_run_dir)
y la copia que llega al PC como resultados/grid/ viene sin convergence.csv.

Uso:
    python plot_comparison.py                            # comparación final
    python plot_comparison.py --algorithms NSGA2 CMOPSO AGEMOEA
    python plot_comparison.py --operadores               # operadores por algoritmo
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
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

# Paleta de colores distinguibles.  Para algoritmos se usa el nombre como clave;
# para operadores (claves no presentes aquí) se cae al ciclo DEFAULT por índice.
COLORS = {
    'NSGA2':   '#000000',   # Negro
    'CMOPSO':  '#FF0000',   # Rojo 100%
    'AGEMOEA': '#008000',   # Verde
    'MOEAD':   '#1F77B4',   # Azul
    'NSGA3':   '#7B1FA2',   # Violeta
}
DEFAULT_COLORS = ['#000000', '#FF0000', '#008000', '#1F77B4', '#7B1FA2', '#8C564B']

# Orden en que se presentan los algoritmos en la comparación final.
ALGORITHM_ORDER = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'CMOPSO']

# Nombres de presentación para captions.  analisis.py tiene su propio DISPLAY
# con las baselines incluidas; acá alcanza con los algoritmos porque es lo
# único que llega como etiqueta de reporte.
DISPLAY_ALG = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
               'AGEMOEA': 'AGE-MOEA', 'CMOPSO': 'CMOPSO'}
_ALG_POR_DISPLAY = {v: k for k, v in DISPLAY_ALG.items()}

# Todas las series usan el mismo marcador (punto): se distinguen por color,
# no por forma.
PARETO_MARKER = 'o'

# Tamaño del marcador en las figuras de frentes 2D.  Lo elige la densidad de
# puntos del panel, que es lo que produce el solape:
#   pareto_comparison superpone las 5 series en un mismo panel → ~58 pts/in²
#   el grid QED-SA y el frente conjunto dibujan un frente      → ~12-15 pts/in²
# El frente conjunto venía con el valor de la figura densa sin tener su densidad,
# y encima es la figura más ancha (3 paneles), así que al escalarla al ancho de
# texto sus puntos quedaban casi la mitad de los del grid.
MARCADOR_DENSO  = 13    # varias series superpuestas en el panel
MARCADOR_NORMAL = 23    # un frente por panel

# Umbral del constraint de saturación, espejo de utils_mo.FSP3_MIN.  Se define
# acá y no se importa para no arrastrar torch y el VAE a un módulo de gráficas;
# cada metrics.csv trae además la columna 'fsp3_min' con el valor con que corrió,
# así que una divergencia es detectable en el dato y no queda muda.
FSP3_MIN = 0.3

# Etiqueta legible por eje.  Solo QED y SA son objetivos; Fsp3 aparece en las
# figuras como propiedad de lo que el constraint dejó pasar, y su etiqueta lo
# dice para que ningún panel se lea como si fuese un tercer objetivo.
OBJECTIVE_LABELS = {
    'qed':  'QED (↑)',
    'sa':   'SA (↓)',
    'fsp3': f'Fsp3 (restricción $\\geq$ {FSP3_MIN:g})',
}

# Los objetivos, en el orden de las columnas de la matriz F.  Es el único lugar
# donde se declara cuáles son: _df_to_F arma F con ellos y las cargas comprueban
# que estén antes de calcular nada.
OBJECTIVES = ['qed', 'sa']


def get_color(key, idx=0):
    """Color de una serie o de un grupo.  Acepta el nombre corto del algoritmo
    (NSGA2) y también el de presentación (NSGA-II): las figuras que agrupan por
    nombre legible —el frente conjunto del pool— tienen que salir con el mismo
    color que el algoritmo lleva en el resto del documento, no con el del ciclo
    por defecto, que le asignaría a NSGA-III el rojo de CMOPSO."""
    if key in COLORS:
        return COLORS[key]
    corto = _ALG_POR_DISPLAY.get(key)
    if corto in COLORS:
        return COLORS[corto]
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


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


# Las curvas se construyen CRUDAS y el suavizado se aplica al dibujar: así el CSV
# de curvas publica el promedio real sobre las runs y no una media móvil, mientras
# las figuras siguen legibles.  La ventana de cada panel va en su especificación.

def _conv_csv_curves(series, metric):
    """Curva de convergencia de una métrica de convergence.csv.
    Devuelve {label: (gens, media sobre runs)}, sin suavizar."""
    curves = {}
    for s in series:
        df = load_convergence_data(s.pop_dir)
        if df.empty or metric not in df.columns:
            print(f"  ⚠ {s.label}: sin datos de '{metric}'")
            continue
        grouped = df.groupby('gen')[metric].mean().reset_index()
        curves[s.label] = (grouped['gen'].values, grouped[metric].values)
    return curves


def _objective_curves(series, objective):
    """Curva de convergencia del promedio de un objetivo (all_molecules.csv.gz).
    Devuelve {label: (gens, media sobre runs)}, sin suavizar."""
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
        curves[s.label] = (mean_over_runs.index.values, mean_over_runs.values)
    return curves


# Paneles de las figuras de convergencia: (columna, etiqueta y, título, ventana
# de suavizado).  La columna es también el nombre en el CSV de curvas, así que la
# figura y el dato salen de la misma fuente y no pueden divergir.  Los indicadores
# vs frente de referencia llevan ventana 5 y no 20 porque vienen submuestreados
# cada 10 generaciones.
PANELES_MO = [
    ('hv',       'Hipervolumen',   'Convergencia de Hipervolumen (↑)', 20),
    ('igd_plus', 'IGD+ (↓)',       'Convergencia IGD+ (↓)',             5),
    ('epsilon',  'ε+ Aditivo (↓)', 'Convergencia ε+ Aditivo (↓)',       5),
]

# Factibilidad va con las químicas y no con los indicadores MO: mide qué
# fracción de lo generado cumple el constraint, que es una propiedad de las
# moléculas, no del frente.  Es la métrica nueva de esta etapa —antes Fsp3 era
# objetivo y no había nada que cumplir— y la que dice si el algoritmo aprendió a
# quedarse del lado admisible o si sigue gastando evaluaciones fuera.
PANELES_QUIM = [
    ('validity',    'Tasa de Validez',       'Convergencia de Validez',       20),
    ('feasibility', 'Tasa de Factibilidad',  'Convergencia de Factibilidad',  20),
    ('uniqueness',  'Tasa de Unicidad',      'Convergencia de Unicidad',      20),
    ('novelty',     'Tasa de Novedad',       'Convergencia de Novedad',       20),
    ('qed',         'Promedio de QED (↑)',   'Convergencia de QED (↑)',       20),
    ('sa',          'Promedio de SA (↓)',    'Convergencia de SA (↓)',        20),
    # Fsp3 ya no es objetivo, pero su curva es justamente lo que hay que mirar:
    # nada la empuja hacia arriba, así que debería caer hasta apoyarse en el
    # umbral y quedarse ahí.  El panel muestra ese descenso, no una mejora.
    ('fsp3',        f'Promedio de Fsp3 (restr. $\\geq$ {FSP3_MIN:g})',
     'Convergencia de Fsp3 (restricción)', 20),
]


def _mapa_evaluaciones(series):
    """{label: Series(gen → evaluaciones acumuladas)}, promediado sobre las runs.

    Sale de la columna n_eval de convergence.csv y no de gen × pop_size: no
    siempre coinciden —CMOPSO evalúa 200 en una generación y 100 en el resto— y
    acá el eje tiene que ser el gasto real."""
    mapas = {}
    for s in series:
        acum = []
        for f in sorted(glob.glob(os.path.join(s.pop_dir, "run_*", "convergence.csv"))):
            c = pd.read_csv(f)
            if {'gen', 'n_eval'}.issubset(c.columns):
                acum.append(pd.Series(c['n_eval'].cumsum().values,
                                      index=c['gen'].values))
        if acum:
            mapas[s.label] = pd.concat(acum, axis=1).mean(axis=1)
    return mapas


def _a_evaluaciones(curvas, mapas, escala=1000.0):
    """Reindexa curvas de generación a evaluaciones acumuladas (en miles).

    Con presupuestos distintos —200×500 y 100×1000 conviven entre los
    finalistas— la generación no es un eje comparable: en la 500 un algoritmo de
    población 200 ya gastó las 100.000 evaluaciones y uno de 100 va por la mitad.
    Sobre el eje de evaluaciones las cinco curvas terminan en el mismo punto y
    cualquier lectura vertical es a igual presupuesto."""
    out = {}
    for label, (gens, vals) in curvas.items():
        m = mapas.get(label)
        if m is None:
            continue
        ev = m.reindex(gens).values
        ok = ~np.isnan(ev)
        out[label] = (ev[ok] / escala, np.asarray(vals)[ok])
    return out


def escribir_curvas_csv(series, curvas, mapas, output_dir, pop_size):
    """CSV con todas las curvas de convergencia en formato largo: una fila por
    serie y generación, con las evaluaciones acumuladas y una columna por métrica.

    Son los valores CRUDOS —promedio sobre las runs, sin la media móvil que llevan
    las figuras—, así que sirven para citar números en el documento.  IGD+ y ε+
    quedan vacíos en las generaciones que no cayeron en el submuestreo con que se
    calculan (cada 10)."""
    cols = [c for c, *_ in PANELES_MO + PANELES_QUIM]
    filas = []
    for s in series:
        m = mapas.get(s.label)
        if m is None:
            continue
        datos = {c: dict(zip(*curvas[c][s.label]))
                 for c in cols if s.label in (curvas.get(c) or {})}
        for gen in m.index:
            fila = {'series': s.label, 'gen': int(gen),
                    'evaluaciones': int(round(m.loc[gen]))}
            fila.update({c: datos[c].get(gen) for c in cols if c in datos})
            filas.append(fila)
    if not filas:
        return
    out = os.path.join(output_dir, f"convergence_curves_pop{pop_size}.csv")
    pd.DataFrame(filas, columns=['series', 'gen', 'evaluaciones'] + cols
                 ).to_csv(out, index=False)
    print(f"  ✓ convergence_curves_pop{pop_size}.csv  ({len(filas)} filas)")


def _plot_convergence_grid(series, output_dir, specs, curvas, mapas, fname,
                           suptitle):
    """Dibuja una grilla de paneles de convergencia (3 por fila).

    specs: lista de (col, ylabel, title, ventana) — ver PANELES_MO / PANELES_QUIM.
    curvas: {col: {label: (gens, vals crudos)}}.
    mapas: mapa de evaluaciones acumuladas por serie (ver _mapa_evaluaciones).

    El eje x son SIEMPRE evaluaciones, nunca generaciones: es el único donde la
    comparación es a igual presupuesto, porque conviven repartos de 200×500 y
    100×1000 y en la generación 500 un algoritmo ya gastó las 100.000
    evaluaciones y el otro va por la mitad."""
    panels = []
    for col, ylabel, title, ventana in specs:
        c = curvas.get(col) or {}
        c = {lab: (x, _smooth(v, ventana)) for lab, (x, v) in c.items()}
        c = _a_evaluaciones(c, mapas)
        if c:
            panels.append((ylabel, title, c))
    if not panels:
        print(f"  ⚠ {fname}: sin datos de convergencia")
        return
    xlabel = 'Evaluaciones (miles)'

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
            x, vals = curves[s.label]
            line, = ax.plot(x, vals, color=get_color(s.color_key, idx),
                            linewidth=1.2, label=s.label, zorder=3)
            if s.label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(s.label)
        ax.set_xlabel(xlabel)
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
# Fsp3 va sin flecha: no se optimiza, se reporta.  Marcarla con (↑) haría leer
# como derrota que un algoritmo se quede cerca del umbral, que es exactamente lo
# que se espera cuando el constraint reemplaza al objetivo.
BOXPLOT_CHEM_CONFIGS = [
    ('mean_qed',    'QED (↑)',              True),
    ('mean_sa',     'SA (↓)',               False),
    ('feasibility', 'Factibilidad (↑)',     True),
    ('mean_fsp3',   f'Fsp3 (restr. $\\geq$ {FSP3_MIN:g})', True),
    ('validity',    'Tasa de Validez',      True),
    ('uniqueness',  'Unicidad (↑)',         True),
    ('novelty',     'Novedad (↑)',          True),
]


def plot_boxplots(series, output_dir, get_values, plot_configs,
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
    """Convierte DataFrame con qed y sa a matriz F de minimización [-QED, SA].

    Fsp3 no entra: es constraint, no objetivo.  Meterlo como tercera columna
    —como hacía la etapa anterior— cambiaría quién domina a quién: una molécula
    peor en los dos objetivos sobreviviría por tener más Fsp3, cuando el
    constraint solo distingue entre admisible y no admisible."""
    return np.column_stack([-df['qed'].to_numpy(dtype=float),
                            df['sa'].to_numpy(dtype=float)])   # ver OBJECTIVES


def _compute_non_dominated(df):
    """Recalcula el frente no-dominado de un DataFrame con qed y sa.

    Las filas que llegan acá ya pasaron el constraint: molecules.csv publica
    únicamente el frente factible (ver utils_mo.build_pareto), así que no hay
    que volver a filtrar por Fsp3.  La excepción es el frente por generación de
    all_molecules.csv.gz, que sí trae infactibles y se filtra en el origen."""
    if not set(OBJECTIVES).issubset(df.columns) or df.empty:
        return df
    F = _df_to_F(df)
    front_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return df.iloc[front_idx].reset_index(drop=True)


def _front_bounds(pf_F):
    """Ideal (mínimos) y escala (nadir − ideal) del frente de referencia.

    Se usan para normalizar los objetivos a [0,1] antes de IGD+ y ε+, de modo
    que los dos objetivos pesen por igual (sin esto, SA domina por su rango
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

def build_reference_front(series):
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


def compute_indicators_per_run(series, pf_F):
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
            if df.empty or not set(OBJECTIVES).issubset(df.columns):
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


# El espacio de objetivos es un plano: con dos objetivos el frente es una curva
# en QED-SA y no hay más proyecciones que mirar.  La etapa anterior dibujaba tres
# paneles porque Fsp3 era el tercer objetivo; ahora es constraint y su lugar está
# en PLANOS_FRENTE, no acá.
PARETO_PLANES = [('qed', 'sa')]

# Los paneles del frente conjunto: el espacio de objetivos más un diagnóstico del
# constraint.  El segundo panel no es una proyección del frente —Fsp3 no ordena
# nada— sino la respuesta a dónde se paró la búsqueda respecto del umbral: como
# ya nada empuja Fsp3 hacia arriba, se espera que las soluciones se estacionen en
# el borde, y conviene verlo en vez de suponerlo.
PLANOS_FRENTE = [('qed', 'sa'), ('qed', 'fsp3')]


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
                        s=MARCADOR_DENSO, alpha=0.55,
                        edgecolors='none', linewidths=0,
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
            _pie_marker(ax, xv, yv, uniq, size=18)

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
#   los dos objetivos: no hay umbrales ni ponderaciones de por medio (el único
#   umbral del experimento, Fsp3, ya filtró antes qué entra a competir).

# Okabe-Ito: naranja/azul se distinguen bajo los tres tipos de daltonismo.
# CMOPSO entra acá porque en el frente conjunto de candidatos convive con las dos
# familias de cruce sin pertenecer a ninguna: no tiene operadores.  Su rojo va
# oscurecido para que no se confunda con el naranja de PCX.
CRUCE_COLORS = {'PCX': '#D55E00', 'SBX': '#0072B2', 'CMOPSO': '#B01818'}
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


# Ancho de la banda que cuenta como «apoyado en el umbral».  Con Fsp3 fuera de
# los objetivos nada la empuja hacia arriba, así que la pregunta útil dejó de ser
# cuántas moléculas llegan alto (con el constraint casi ninguna: el máximo del
# grid ronda 0.64) y pasó a ser cuántas se estacionan justo sobre el borde.
FSP3_BORDE = 0.05


def _perfil(df):
    """Descriptores del subconjunto que aporta un operador al frente conjunto:
    dónde cae y qué calidad tiene lo que aporta."""
    if df.empty:
        return {'n': 0, 'fsp3': np.nan, 'fsp3_borde': np.nan,
                'qed': np.nan, 'qed_bajo': np.nan, 'sa': np.nan}
    return {'n': len(df),
            'fsp3': float(df.fsp3.mean()),
            'fsp3_borde': float((df.fsp3 < FSP3_MIN + FSP3_BORDE).mean()),
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


def _atribucion_por_origen(series, pf_df, grupo_de):
    """Prepara el frente conjunto para dibujarlo: lo atribuye, arma la paleta y
    cuenta cuántas moléculas puso cada grupo en exclusiva.

    La separa de la figura porque la atribución es lo caro —relee molecules.csv
    de todas las series— y porque la paleta tiene que salir de un solo lugar: es
    la misma que usa la tabla de contribución, y armarla dos veces dejaría el
    cuadro y el gráfico del documento contando lo mismo con colores distintos.

    Devuelve None cuando la atribución no aplica (sin la columna 'origen' no hay
    nada que colorear, que es el caso de una sola serie)."""
    at = atribuir_frente(series, pf_df, grupo_de)
    if 'origen' not in at.columns:
        return None
    grupos = _grupos_de(series, grupo_de)
    compartida = _etiqueta_compartida(grupos)
    # Los combos de operadores no están en COLORS y caerían todos al mismo color
    # del ciclo por defecto; los algoritmos sí tienen color propio asignado.
    por_cruce = set(grupos) <= set(CRUCE_COLORS)
    paleta = ({g: CRUCE_COLORS[g] for g in grupos} if por_cruce
              else {g: get_color(g, i) for i, g in enumerate(grupos)})
    paleta[compartida] = COMPARTIDA_COLOR

    orden = [g for g in list(grupos) + [compartida] if (at['origen'] == g).any()]
    return {'at': at, 'paleta': paleta, 'compartida': compartida, 'orden': orden,
            'cuentas': {g: int((at['origen'] == g).sum()) for g in orden},
            'colores': at['origen'].map(paleta).values,
            'por': 'familia de cruce' if por_cruce else 'algoritmo'}


def _leyenda_y_titulo_origen(fig, atr, output_dir, leyenda_y, titulo_y):
    """Leyenda de grupos y título de las figuras del frente conjunto.  Las dos
    coordenadas verticales se pasan por parámetro porque dependen de cuántos
    paneles lleve la figura.

    'solo X' y no 'X' a secas: estas cuentas son exclusivas, mientras que la
    columna «Aporta» de la tabla incluye las compartidas."""
    compartida = atr['compartida']
    handles = [mpatches.Patch(
        facecolor=atr['paleta'][g], edgecolor='white',
        label=f'{g if g == compartida else "solo " + str(g)} ({atr["cuentas"][g]})')
        for g in atr['orden']]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               framealpha=0.9, edgecolor='#cccccc', fontsize=11,
               bbox_to_anchor=(0.5, leyenda_y))

    alg = _alg_from_output_dir(output_dir)
    fig.suptitle(f'Frente no dominado conjunto por {atr["por"]}'
                 + (f' - {alg}' if alg else ''),
                 fontsize=14, fontweight='bold', y=titulo_y)


def _linea_constraint(ax, eje):
    """Marca el umbral del constraint sobre el eje que lleva Fsp3.

    Sin la línea el panel no se puede leer: la nube arranca en 0.3 y parece un
    borde de los datos, cuando es el umbral que la búsqueda tenía prohibido
    cruzar.  Es lo que separa «se paró en el borde» de «no llegó más abajo»."""
    trazo = ax.axvline if eje == 'x' else ax.axhline
    trazo(FSP3_MIN, color='#444444', linestyle=':', linewidth=1.4, zorder=2,
          label=f'Fsp3 = {FSP3_MIN:g}')


def plot_frente_conjunto(series, pop_size, output_dir, pf_df, grupo_de=_familia):
    """El frente no dominado conjunto, cada molécula pintada según quién la
    aportó (familia de cruce al comparar operadores, algoritmo al comparar
    algoritmos).

    Dos paneles: el espacio de objetivos —donde vive el frente— y el diagnóstico
    del constraint, con el umbral dibujado (ver PLANOS_FRENTE).

    Todos los puntos van en un único scatter con un array de colores: si se
    dibujara un grupo después de otro, el último taparía a los anteriores en la
    zona densa y la figura mostraría una diferencia de orden de dibujo en vez de
    una diferencia real."""
    atr = _atribucion_por_origen(series, pf_df, grupo_de)
    if atr is None:
        return
    at, colores = atr['at'], atr['colores']

    planos = [(x, y) for x, y in PLANOS_FRENTE
              if x in at.columns and y in at.columns]
    if not planos:
        return
    n = len(planos)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.8), squeeze=False)
    for ax, (xcol, ycol) in zip(axes[0], planos):
        ax.scatter(at[xcol], at[ycol], c=colores, marker=PARETO_MARKER,
                   s=MARCADOR_NORMAL, alpha=0.55, edgecolors='none',
                   linewidths=0, zorder=3)
        ax.set_xlabel(OBJECTIVE_LABELS.get(xcol, xcol))
        ax.set_ylabel(OBJECTIVE_LABELS.get(ycol, ycol))
        ax.set_title('Espacio de objetivos' if (xcol, ycol) in PARETO_PLANES
                     else 'Restricción de saturación')
        ax.set_xlim(*_pad_lim(at[xcol].values))
        ax.set_ylim(*_pad_lim(at[ycol].values))
        if 'fsp3' in (xcol, ycol):
            _linea_constraint(ax, 'x' if xcol == 'fsp3' else 'y')

    _leyenda_y_titulo_origen(fig, atr, output_dir, 0.01, 1.0)
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])
    fname = f"frente_conjunto_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# La figura 3D del frente conjunto se eliminó al pasar a dos objetivos: existía
# para mostrar la forma de una superficie en QED-SA-Fsp3, y con Fsp3 fuera de la
# dominancia esa superficie no existe — el frente es una curva y proyectarla en
# tres ejes sugeriría un compromiso que ya no se optimiza.


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

    cols = ['hypervolume', 'spacing', 'validity', 'feasibility', 'novelty',
            'igd_plus', 'epsilon',
            'best_qed', 'best_sa', 'mean_fsp3', 'n_pareto', 'time_sec']
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


# Separador decimal de todas las tablas LaTeX, en los dos módulos (analisis.py
# lo toma de acá).  Para coma usar '{,}': las llaves hacen que en modo matemático
# LaTeX la trate como símbolo ordinario y no como puntuación, que llevaría un
# espacio detrás.
SEP_DECIMAL = '.'


def _num_es(x, fmt):
    """Número con el separador decimal del documento."""
    return f'{x:{fmt}}'.replace('.', SEP_DECIMAL)


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

    # Mejor serie por columna (según dirección de optimización).  Con
    # higher_better=None la columna se reporta sin competir —es el caso de Fsp3,
    # que dejó de ser objetivo— así que no lleva ni flecha ni negrita: marcar una
    # ganadora afirmaría un criterio que el experimento no optimiza.
    best = {}
    for _, col, _, hb in cols:
        if hb is None:
            continue
        cand = [(s.label, means[(s.label, col)]) for s in series
                if (s.label, col) in means]
        if cand:
            best[col] = (max if hb else min)(cand, key=lambda t: t[1])[0]

    arrow = lambda hb: '' if hb is None else (r'$\uparrow$' if hb
                                              else r'$\downarrow$')
    col_spec = 'l' + 'c' * len(cols)
    header_cells = ['Algoritmo'] + [f'{h} {arrow(hb)}'.strip()
                                    for h, _, _, hb in cols]

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
            body = f'{_num_es(m, fmt)} \\pm {_num_es(sd, fmt)}'
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
    # Fsp3 entra con higher_better=None: se reporta pero no se compite por ella,
    # así que la tabla no debe poner en negrita a «la mejor».  El resto de las
    # columnas sí tienen dirección y la conservan.
    chem_cfg = [
        ('QED',           'mean_qed',    '.4f', True),
        ('SA',            'mean_sa',     '.2f', False),
        ('Factibilidad',  'feasibility', '.4f', True),
        ('Fsp3',          'mean_fsp3',   '.4f', None),
        ('Validez',       'validity',    '.4f', True),
        ('Unicidad',      'uniqueness',  '.4f', True),
        ('Novedad',       'novelty',     '.4f', True),
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
    # El alcance no es el mismo en todas las columnas y conviene decirlo: QED, SA
    # y Fsp3 son del frente acumulado, validez, factibilidad y novedad de todas
    # las evaluaciones, y unicidad solo de la última generación.
    _write_latex_comparison_table(
        series, col_values, chem_cfg,
        f'Comparación de indicadores químicos{cap_ctx}.  QED, SA y Fsp3 son la '
        f'media del frente no dominado acumulado sobre la corrida completa; '
        f'validez, factibilidad y novedad se calculan sobre las evaluaciones de '
        f'toda la corrida (factibilidad como fracción de las válidas que cumplen '
        f'Fsp3 $\\geq$ {FSP3_MIN:g}); unicidad corresponde a la población de la '
        f'última generación.  Fsp3 se informa sin dirección de optimización: es '
        f'la restricción, no un objetivo, y su valor dice dónde se estacionó la '
        f'búsqueda respecto del umbral',
        f'tab:comparison_chemical{lab_ctx}_pop{pop_size}',
        output_dir, f'comparison_chemical_pop{pop_size}.tex', pop_size)




def _cmap_recortada(nombre, hasta):
    """El tramo [0, hasta] de un colormap, como rampa propia.

    El extremo pálido de plasma —el amarillo— es ilegible sobre el fondo blanco de
    la figura, y ahí caen justamente los valores altos: las moléculas que reaparecen
    en muchas semillas, que son ~10% del frente y lo que interesa ver.  Recortarlo
    las deja en naranja saturado y conserva el tramo oscuro, que es el que lleva el
    grueso de los puntos y le da forma al frente.

    Cambiar a un tono único claro→oscuro, que es la regla habitual para codificar
    magnitud, acá no sirve: tres cuartos de las moléculas están en el valor mínimo,
    así que el extremo claro se queda con el grueso y el frente se desdibuja."""
    base = plt.get_cmap(nombre)
    return mcolors.LinearSegmentedColormap.from_list(
        f'{nombre}_{hasta:g}', base(np.linspace(0.0, hasta, 256)))


# Variantes de color del grid QED vs SA.  Los dos paneles comparten geometría y
# responden preguntas distintas, así que se generan como archivos separados:
#   nruns → ¿el hallazgo se repite entre semillas, o lo vio una sola?
#   fsp3  → ¿qué margen sobre el umbral tiene cada punto del compromiso QED-SA?
# El sufijo va en el nombre del archivo para que no se confundan.
#
# El valor va solo en el color: todos los marcadores miden lo mismo, así que la
# posición de un punto no compite con su tamaño por la atención del lector.
# 'entero' marca las escalas que cuentan cosas: la colorbar lleva solo marcas
# enteras, porque una molécula no aparece en 2.5 ejecuciones.  Fsp3 es una
# fracción continua y ahí las marcas decimales son las correctas.
GRID_COLOR_MODES = {
    'nruns': dict(col='n_runs_appeared', cmap=_cmap_recortada('plasma', 0.72),
                  label='Nº de ejecuciones en que aparece', entero=True),
    'fsp3':  dict(col='fsp3', cmap='viridis',
                  label=f'Fsp3 (restricción $\\geq$ {FSP3_MIN:g})'),
}


def plot_pareto_qed_sa_grid(series, pop_size, output_dir, color_by='nruns'):
    """Genera UNA imagen con un panel por serie (los 5 separados, no
    superpuestos), mostrando solo el frente de Pareto QED vs SA.
    Cada serie combina las moléculas de sus runs, elimina duplicados de
    SMILES y recalcula el frente no-dominado global.

    color_by elige qué se codifica en color (ver GRID_COLOR_MODES)."""
    if color_by not in GRID_COLOR_MODES:
        raise ValueError(f"color_by debe ser uno de {list(GRID_COLOR_MODES)}")
    modo = GRID_COLOR_MODES[color_by]

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
        if modo['col'] not in pareto.columns:
            continue
        # Los valores altos se dibujan últimos para que el grueso del frente no
        # los tape: son justamente los que interesa ver.
        paretos.append((s, pareto.sort_values(modo['col']), n_runs))

    if not paretos:
        print("  ⚠ Sin datos de Pareto para grid QED vs SA")
        return

    if color_by == 'fsp3':
        # Escala absoluta para que los paneles se comparen entre sí, pero
        # arrancando en el umbral y no en 0: el frente publicado es factible por
        # construcción, así que el tramo [0, FSP3_MIN) de la rampa no puede
        # recibir ningún punto y gastarlo aplanaría todo el frente en un mismo
        # tono.  El tope sale de los datos —con Fsp3 fuera de los objetivos nadie
        # llega cerca de 1— acotado por abajo para que la banda nunca degenere.
        fsp3_max = max((p['fsp3'].max() for _, p, _ in paretos), default=1.0)
        norm = mcolors.Normalize(vmin=FSP3_MIN,
                                 vmax=max(float(fsp3_max), FSP3_MIN + 0.1))
    else:
        # 1 run → violeta, max_runs → amarillo
        global_max_runs = max(nr for _, _, nr in paretos)
        norm = mcolors.Normalize(vmin=1, vmax=max(global_max_runs, 2))

    n_plots = len(paretos)
    # Con 4 series (los combos de operadores) una grilla 2×2 queda pareja; con
    # 3 columnas sobraría una celda vacía.  Con 5 (los algoritmos) 3+2 es lo mejor.
    ncols = 2 if n_plots == 4 else min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 5.5 * nrows),
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.flatten()

    sc = None
    for ax, (s, pareto, n_runs) in zip(axes_flat, paretos):
        qed = pareto['qed'].values
        sa  = pareto['sa'].values
        sc = ax.scatter(qed, sa, c=pareto[modo['col']].values,
                        cmap=modo['cmap'], norm=norm,
                        s=MARCADOR_NORMAL, alpha=0.6,
                        edgecolors='none', linewidths=0, zorder=3)
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
        if modo.get('entero'):
            # Marcas enteras con paso redondo, y siempre los dos extremos: 1 (la
            # halló una sola ejecución) y el total, que son los que interpretan la
            # escala.  Con 20 ejecuciones da 1, 5, 10, 15, 20.
            hi = int(round(norm.vmax))
            paso = next(p for p in (1, 2, 5, 10, 25, 50, 100)
                        if (hi - 1) / p <= 5)
            cbar.set_ticks(sorted({1, hi} | set(range(paso, hi, paso)) - {0}))
        cbar.set_label(modo['label'], fontsize=11)

    title = 'Frentes de Pareto QED vs SA por algoritmo'
    alg = _alg_from_output_dir(output_dir)
    if alg:
        title = f'Frentes de Pareto QED vs SA por operador - {alg}'
    # Qué codifica el color no va en el título: lo dice la colorbar, y el detalle
    # va en el pie de figura del documento.

    fig.suptitle(title,
                 fontsize=14, fontweight='bold')
    fname = f"pareto_qed_sa_grid_{color_by}_pop{pop_size}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")




def _indicator_curves(series, pop_size, output_dir, pf_F, gen_stride=10):
    """Curvas de convergencia de IGD+ y ε+ por generación, SIN re-entrenar.

    Reconstruye el frente no-dominado de cada generación a partir de
    all_molecules.csv.gz (que guarda la población completa por gen) y mide
    los indicadores contra el frente de referencia combinado pf_F.
    Promedia sobre todas las runs de cada serie y guarda las curvas en CSV.

    Solo compiten las moléculas FACTIBLES, igual que en el frente final que
    publica utils_mo.build_pareto.  El log crudo sí trae las infactibles, y
    dejarlas entrar mediría la curva contra un frente por generación que incluye
    soluciones que violan el umbral, mientras el frente de referencia se armó
    solo con factibles: los indicadores quedarían comparando dos poblaciones
    distintas y la curva bajaría por admitir lo inadmisible.

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
                df = pd.read_csv(gz_path, usecols=lambda c: c in {
                    'gen', 'qed', 'sa', 'fsp3', 'valid', 'feasible'})
            except Exception:
                continue
            if not {'gen', 'qed', 'sa', 'valid'}.issubset(df.columns):
                continue
            df = df[df['valid'].astype(bool)].dropna(subset=['qed', 'sa'])
            # 'feasible' lo escribe el eval_log de esta etapa; si el log viniera
            # de una corrida sin constraint se cae a Fsp3 ≥ umbral, que es la
            # misma condición calculada desde la propiedad.
            if 'feasible' in df.columns:
                df = df[df['feasible'].astype(bool)]
            elif 'fsp3' in df.columns:
                df = df[df['fsp3'] >= FSP3_MIN]
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

    # Reorganiza a {col: {label: (gens, vals)}}; el suavizado lo aplica el dibujo.
    out = {'igd_plus': {}, 'epsilon': {}}
    for label, curve in series_curves.items():
        for col in out:
            out[col][label] = (curve.index.values, curve[col].values)

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
        pf_F, pf_df = build_reference_front(series)
        if pf_F is not None:
            print(f"   Frente de referencia: {len(pf_F)} soluciones no-dominadas")
            indicator_data = compute_indicators_per_run(series, pf_F)

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

    # 2. Todas las curvas de convergencia, crudas y en un solo diccionario
    #    indexado por el nombre de columna: de acá salen las cuatro figuras y el
    #    CSV, así que el dato publicado y el dibujado no pueden divergir.  Se
    #    calculan una sola vez porque _objective_curves lee los
    #    all_molecules.csv.gz de todas las runs y es la parte cara.
    print("📈 Curvas de convergencia (MO y químicas)...")
    curvas = {
        'hv':          _conv_csv_curves(series, 'hv'),
        'igd_plus':    ind_curves['igd_plus'],
        'epsilon':     ind_curves['epsilon'],
        'validity':    _conv_csv_curves(series, 'validity'),
        'feasibility': _conv_csv_curves(series, 'feasibility'),
        'uniqueness':  _conv_csv_curves(series, 'uniqueness'),
        'novelty':     _conv_csv_curves(series, 'novelty'),
        'qed':         _objective_curves(series, 'qed'),
        'sa':          _objective_curves(series, 'sa'),
        'fsp3':        _objective_curves(series, 'fsp3'),
    }

    # 3. Las curvas van SOLO contra evaluaciones.  La versión por generación se
    #    eliminó: con repartos distintos de pob×gen la generación no es un eje
    #    comparable entre series, así que las dos figuras invitaban a leer a
    #    igual generación una diferencia que era de presupuesto.  El CSV sigue
    #    publicando las dos columnas (gen y evaluaciones) para citar números.
    mapas_eval = _mapa_evaluaciones(series)
    if not mapas_eval:
        print("  ⚠ sin n_eval en convergence.csv: se omiten las curvas de "
              "convergencia (el eje de evaluaciones sale de ahí)")
    else:
        for specs, base, titulo in [
                (PANELES_MO, 'mo', 'Convergencia de Indicadores Multiobjetivo'),
                (PANELES_QUIM, 'chemical', 'Convergencia de Indicadores Químicos')]:
            _plot_convergence_grid(
                series, output_dir, specs, curvas, mapas_eval,
                f"convergence_{base}_evals_pop{pop_size}.png", titulo)
        escribir_curvas_csv(series, curvas, mapas_eval, output_dir, pop_size)

    # 5. Boxplots + tablas (requiere ≥2 series).  Comparten un único getter
    #    de valores per-run (metrics + indicadores + medias químicas + unicidad).
    if len(series) >= 2:
        get_values = _build_series_value_getter(series, indicator_data)

        print("📊 Boxplots multiobjetivo (HV, Espaciamiento, IGD+, ε+, Pareto)...")
        plot_boxplots(series, output_dir, get_values, BOXPLOT_MO_CONFIGS,
                      f"boxplots_mo_pop{pop_size}.png",
                      "Distribución de Indicadores Multiobjetivo")
        print("📊 Boxplots químicos (QED, SA, Factibilidad, Fsp3, Validez, "
              "Unicidad, Novedad)...")
        plot_boxplots(series, output_dir, get_values, BOXPLOT_CHEM_CONFIGS,
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



    # 9. Grid QED vs SA: los N algoritmos separados en una sola imagen, en sus
    #    dos variantes de color (reproducibilidad entre semillas y la restricción).
    print("🧩 Grid QED vs SA por algoritmo (una imagen por variante de color)...")
    for modo in GRID_COLOR_MODES:
        plot_pareto_qed_sa_grid(series, pop_size, output_dir, color_by=modo)


# ─── Construcción de series por modo ────────────────────────────────────────

def build_finalist_series(algorithms, finalistas_dir):
    """Modo algoritmos sobre finalistas/<ALG>/run_XX/: la configuración elegida
    de cada algoritmo tras las etapas 1 y 2."""
    series = []
    for alg in algorithms:
        d = os.path.join(finalistas_dir, alg)
        if _has_runs(d):
            series.append(Series(alg, d, color_key=alg))
    return series


def winner_cfg_dir(winners_dir, alg, combo):
    """Directorio de la configuración con que un combo ganó su bloque en la
    etapa 1.  Cada uno ganó con hiperparámetros distintos, así que el nivel de
    configuración no se puede nombrar: se resuelve con glob y se toma el hijo con
    runs.  None si ese combo no tiene datos."""
    cfgs = [d for d in sorted(glob.glob(os.path.join(winners_dir, alg, combo, '*')))
            if _has_runs(d)]
    return cfgs[0] if cfgs else None


def build_operator_series_winners(alg, winners_dir, combos=None):
    """Modo operadores sobre winners/<ALG>/<combo>/<config>/: las configuraciones
    que ganaron su bloque en la etapa 1.

    combos fija el orden de las series, que es el de las leyendas y el de las
    filas de las tablas; sin él se toman los combos del directorio, alfabéticos."""
    if combos is None:
        base = os.path.join(winners_dir, alg)
        combos = [c for c in sorted(os.listdir(base))
                  if os.path.isdir(os.path.join(base, c))]
    series = []
    for combo in combos:
        cfg_dir = winner_cfg_dir(winners_dir, alg, combo)
        if cfg_dir:
            series.append(Series(combo, cfg_dir, color_key=combo))
    return series


# ─── Main ────────────────────────────────────────────────────────────────────

def run_algorithm_comparison(algorithms, finalistas):
    """Comparación final entre algoritmos: la configuración elegida de cada uno,
    leída de finalistas/<ALG>/ (symlinks a la ganadora dentro de winners/)."""
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


def run_operator_comparison(algorithms, winners_dir):
    """Comparación de variantes de operadores, un reporte por algoritmo, sobre
    las configuraciones que ganaron su bloque en la etapa 1
    (winners/<ALG>/<combo>/<config>/), cada una con sus propios hiperparámetros.

    La salida va a plots/operadores/<ALG>/winners/: un nivel más abajo que las
    tablas de analisis.py etapa2, que escribe en plots/operadores/<ALG>/."""
    if not os.path.isdir(winners_dir):
        print(f"No existe {winners_dir}")
        return
    algorithms = algorithms or sorted(
        d for d in os.listdir(winners_dir)
        if os.path.isdir(os.path.join(winners_dir, d)))
    tag = "winners"

    print(f"\n{'='*60}")
    print("  Comparación de operadores (por algoritmo)")
    print(f"  Origen: {winners_dir}")
    print(f"  Algoritmos candidatos: {', '.join(algorithms)}")
    print(f"{'='*60}")

    generated = []
    for alg in algorithms:
        series = build_operator_series_winners(alg, winners_dir)
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
    parser = argparse.ArgumentParser(
        description="Gráficas comparativas MOO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help="Algoritmos a comparar (auto-detecta si no se especifica)")
    parser.add_argument('--operadores', action='store_true',
                        help="Compara variantes de operadores por algoritmo")
    parser.add_argument('--winners', default=os.path.join(RESULTADOS_DIR, 'winners'),
                        help="Con --operadores: las configuraciones ganadoras de "
                             "la etapa 1, en <winners>/<ALG>/<combo>/<config>/")
    parser.add_argument('--finalistas', default=os.path.join(RESULTADOS_DIR,
                                                             'finalistas'),
                        help="Comparación final: la configuración elegida de cada "
                             "algoritmo, en <finalistas>/<ALG>/")
    args = parser.parse_args()

    if args.operadores:
        run_operator_comparison(args.algorithms, args.winners)
    else:
        run_algorithm_comparison(args.algorithms, args.finalistas)


if __name__ == "__main__":
    main()

