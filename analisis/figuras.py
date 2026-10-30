"""
Figuras: convergencia, boxplots, frentes de Pareto, el grid QED-SA y las
estructuras moleculares.

El eje x de la convergencia son evaluaciones, no generaciones.
"""

import glob
import io
import math
import os

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .comun import (
    CRUCES,
    DOCK_WORST,
    FSP3_MIN,
    MARCADOR_DENSO,
    MARCADOR_NORMAL,
    OBJECTIVE_LABELS,
    PARETO_MARKER,
    _build_series_value_getter,
    generate_latex_comparison_tables,
    generate_statistical_table,
    get_color,
    load_convergence_data,
    load_pareto_molecules,
    unicas,
)
from .indicadores import (
    _atribucion_por_origen,
    _compute_non_dominated,
    _familia,
    _indicator_curves,
    compute_indicators_per_run,
)



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
                if objective == 'dock':
                    # NaN: no se dockeó (inválida o infactible); ≥ DOCK_WORST: falló.
                    df = df[df['dock'] < DOCK_WORST]
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


PANELES_QUIM = [
    ('validity',    'Tasa de Validez',       'Convergencia de Validez',       20),
    ('uniqueness',  'Tasa de Unicidad',      'Convergencia de Unicidad',      20),
    ('novelty',     'Tasa de Novedad',       'Convergencia de Novedad',       20),
    ('qed',         'Promedio de QED (↑)',   'Convergencia de QED (↑)',       20),
    ('sa',          'Promedio de SA (↓)',    'Convergencia de SA (↓)',        20),
    ('dock',        'Promedio de docking (kcal/mol, ↓)',
                    'Convergencia de Docking (↓)',                            20),
]



def _mapa_evaluaciones(series):
    """{label: Series(gen → evaluaciones acumuladas)}, promediado sobre las runs.

    Sale de la columna n_eval de convergence.csv: es el gasto real de cada
    generación."""
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
    """Reindexa curvas de generación a evaluaciones acumuladas (en miles)."""
    out = {}
    for label, (gens, vals) in curvas.items():
        m = mapas.get(label)
        if m is None:
            continue
        ev = m.reindex(gens).values
        ok = ~np.isnan(ev)
        out[label] = (ev[ok] / escala, np.asarray(vals)[ok])
    return out



def escribir_curvas_csv(series, curvas, mapas, output_dir):
    """CSV con las curvas de convergencia en formato largo: una fila por serie y
    generación, con las evaluaciones acumuladas y una columna por métrica.

    Valores CRUDOS (promedio sobre las runs, sin la media móvil de las figuras),
    para citar números en el documento.  IGD+ y ε+ quedan vacíos en las
    generaciones fuera del submuestreo con que se calculan (cada 10)."""
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
    out = os.path.join(output_dir, "curvas_convergencia.csv")
    pd.DataFrame(filas, columns=['series', 'gen', 'evaluaciones'] + cols
                 ).to_csv(out, index=False)
    print(f"  ✓ curvas_convergencia.csv  ({len(filas)} filas)")



def _plot_convergence_grid(series, output_dir, specs, curvas, mapas, fname,
                           suptitle):
    """Dibuja una grilla de paneles de convergencia (3 por fila).

    specs: lista de (col, ylabel, title, ventana) — ver PANELES_MO / PANELES_QUIM.
    curvas: {col: {label: (gens, vals crudos)}}.
    mapas: mapa de evaluaciones acumuladas por serie (ver _mapa_evaluaciones).

    El eje x son SIEMPRE evaluaciones: es el único a igual presupuesto (ver
    _a_evaluaciones)."""
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
                            linestyle=s.linestyle, linewidth=1.4,
                            label=s.label, zorder=3)
            if s.label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(s.label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Las tasas parten de 0; el docking es negativo y se deja autoescalar.
        if all(np.nanmin(v) >= 0 for _, v in curves.values()):
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
    ('best_dock',   'Mejor docking, kcal/mol (↓)', False),
]

# Fsp3 va sin flecha: no se optimiza, se reporta.  Marcarla con (↑) haría leer
# como derrota que un algoritmo se quede cerca del umbral, que es exactamente lo
# que se espera cuando el constraint reemplaza al objetivo.
BOXPLOT_CHEM_CONFIGS = [
    ('mean_qed',    'QED (↑)',              True),
    ('mean_sa',     'SA (↓)',               False),
    ('mean_dock',   'Docking, kcal/mol (↓)', False),
    ('mean_fsp3',  f'Fsp3 (restr. $\\geq$ {FSP3_MIN:g})', True),
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
        data, labels, colors, hatches = [], [], [], []
        for idx, s in enumerate(series):
            vals = get_values(s.label, col)
            if vals is not None and len(vals):
                data.append(vals)
                labels.append(s.label)
                colors.append(get_color(s.color_key, idx))
                hatches.append(s.hatch)

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=['']*len(labels), patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=6))
        for patch, color, hatch in zip(bp['boxes'], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_hatch(hatch)
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
        patch = mpatches.Patch(facecolor=color, edgecolor='black', alpha=0.6,
                               hatch=s.hatch, label=s.label)
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



# Con tres objetivos el frente es una superficie: se mira por sus tres proyecciones.
PARETO_PLANES = [('qed', 'sa'), ('qed', 'dock'), ('sa', 'dock')]


# Mismos planos que PARETO_PLANES: el frente conjunto solo dibuja el espacio de
# objetivos.
PLANOS_FRENTE = PARETO_PLANES



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
        sc = ax.scatter(df[xcol], df[ycol], c=color, marker=s.marker,
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
        # Eliminar SMILES duplicados (dock = mediana, ver unicas)
        df_unique = unicas(df_all)
        # Recalcular frente no-dominado global
        pareto = _compute_non_dominated(df_unique)
        if not pareto.empty:
            combined_paretos[s.label] = pareto
            series_order.append((idx, s))

    counts = {label: len(df) for label, df in combined_paretos.items()}
    return combined_paretos, series_order, counts



def _leyenda_y_titulo_origen(fig, atr, leyenda_y, titulo_y):
    """Leyenda de grupos y título de las figuras del frente conjunto.  Las dos
    coordenadas verticales dependen de cuántos paneles lleve la figura.

    Un patch por grupo con su color y cuánto aportó en total (exclusivas +
    compartidas), igual que las series en plot_pareto_comparison: así todo
    color que aparece en el plano —punto propio o sector de un marcador
    pastel— tiene su clave acá.  Sin entrada para 'compartida': lo compartido
    se ve mezclando los colores ya listados (ver _pie_marker), no como un
    color propio."""
    at, paleta, compartida = atr['at'], atr['paleta'], atr['compartida']
    grupos = [g for g in paleta if g != compartida]
    handles = [mpatches.Patch(
        facecolor=paleta[g], edgecolor='white',
        label=f'{g} ({int(at[f"en_{g}"].sum())})')
        for g in grupos]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   framealpha=0.9, edgecolor='#cccccc', fontsize=11,
                   bbox_to_anchor=(0.5, leyenda_y))

    fig.suptitle(f'Frente no dominado conjunto por {atr["por"]}',
                 fontsize=14, fontweight='bold', y=titulo_y)



def plot_frente_conjunto(series, output_dir, fname, pf_df, grupo_de=_familia):
    """El frente no dominado conjunto en el espacio de objetivos (ver
    PLANOS_FRENTE): un scatter por grupo en su color, con un marcador
    'pastel' donde una misma molécula la aportaron dos o más grupos —igual
    que plot_pareto_comparison (ver _pie_marker)."""
    atr = _atribucion_por_origen(series, pf_df, grupo_de)
    if atr is None:
        return
    at, paleta, compartida = atr['at'], atr['paleta'], atr['compartida']
    grupos = [g for g in paleta if g != compartida]

    planos = [(x, y) for x, y in PLANOS_FRENTE
              if x in at.columns and y in at.columns]
    if not planos:
        return
    n = len(planos)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 5.8), squeeze=False)
    for ax, (xcol, ycol) in zip(axes[0], planos):
        for g in grupos:
            df_g = at[at[f'en_{g}']]
            ax.scatter(df_g[xcol], df_g[ycol], c=paleta[g], marker=PARETO_MARKER,
                       s=MARCADOR_DENSO, alpha=0.55, edgecolors='none',
                       linewidths=0, zorder=3)
        for _, row in at[at['origen'] == compartida].iterrows():
            cols = [paleta[g] for g in grupos if row[f'en_{g}']]
            _pie_marker(ax, row[xcol], row[ycol], cols, size=18)
        ax.set_xlabel(OBJECTIVE_LABELS.get(xcol, xcol))
        ax.set_ylabel(OBJECTIVE_LABELS.get(ycol, ycol))
        ax.set_title('Espacio de objetivos')
        ax.set_xlim(*_pad_lim(at[xcol].values))
        ax.set_ylim(*_pad_lim(at[ycol].values))

    _leyenda_y_titulo_origen(fig, atr, 0.01, 1.0)
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")




def plot_pareto_comparison(series, output_dir, fname, groups=None):
    """Superpone los frentes combinados de cada serie, un panel por plano de
    objetivos.  Con groups —(nombre de fila, [labels])— la figura se parte en una
    fila por grupo, con límites comunes para poder compararlas.
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

    fig.suptitle('Frentes de Pareto Globales',
                 fontsize=14, fontweight='bold', y=1.0)
    # El aire para leyenda y título es una altura fija, no una fracción: al
    # partir en filas la figura crece y el porcentaje reservaría de más.
    margen = 0.09 / len(filas)
    plt.tight_layout(rect=[0, margen, 1, 1 - margen / 3])
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")





# Propiedades de las curvas de densidad: (columna, etiqueta, (mínimo, máximo) del eje;
# None = el de los datos con un margen).  QED llega a 1 y SA no baja de 1.
KDE_PROPIEDADES = [
    ('qed',  'QED (↑)',               (None, 1.0)),
    ('sa',   'SA (↓)',                (1.0, None)),
    ('dock', 'Docking, kcal/mol (↓)', (None, None)),
]



def plot_kde_propiedades(series, output_dir, fname):
    """Curvas de densidad (KDE) de QED, SA y docking en el frente de cada serie:
    cómo se reparten las moléculas del frente, no su calidad."""
    paretos, _, _ = _combined_pareto_fronts(series)
    if not paretos:
        print("  ⚠ Sin datos de Pareto para las curvas de densidad")
        return
    fig, axes = plt.subplots(1, len(KDE_PROPIEDADES),
                             figsize=(6.4 * len(KDE_PROPIEDADES), 5.2))
    for ax, (col, etiqueta, (lo, hi)) in zip(axes, KDE_PROPIEDADES):
        todo = np.concatenate([df[col].values for df in paretos.values()])
        pad = 0.05 * (todo.max() - todo.min())
        x = np.linspace(todo.min() - pad if lo is None else lo,
                        todo.max() + pad if hi is None else hi, 600)
        for idx, s in enumerate(series):
            if s.label not in paretos:
                continue
            valores = paretos[s.label][col].values
            y = gaussian_kde(valores)(x)
            color = get_color(s.color_key, idx)
            ax.plot(x, y, color=color, linestyle=s.linestyle, linewidth=2.2,
                    label=s.label, zorder=3)
            # Solo las discontinuas (SBX) llevan sombra: seis rellenos superpuestos
            # se mezclan en un gris, y así el cruce se distingue también por el
            # relleno.
            if s.linestyle != '-':
                ax.fill_between(x, y, color=color, alpha=0.08, linewidth=0, zorder=2)
            # La mediana, con el marcador del cruce, sobre el eje.
            ax.plot(np.median(valores), 0, marker=s.marker, color=color, markersize=9,
                    markeredgecolor='white', markeredgewidth=0.8, linestyle='none',
                    clip_on=False, zorder=5)
        ax.set_xlabel(etiqueta)
        ax.set_ylabel('Densidad')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-0.025 * ax.get_ylim()[1], None)
    axes[0].legend(frameon=True, fontsize=10)
    fig.suptitle('Distribución de QED, SA y docking en el frente de cada configuración '
                 '(los marcadores sobre el eje son la mediana)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")



def _cmap_recortada(nombre, hasta):
    """El tramo [0, hasta] de un colormap, como rampa propia.

    El amarillo de plasma es ilegible sobre blanco, y ahí caen los valores altos:
    las moléculas que reaparecen en muchas semillas, que son lo que interesa ver.
    Recortarlo las deja en naranja saturado.

    Un tono único claro→oscuro acá no sirve: tres cuartos de las moléculas están
    en el valor mínimo, así que el extremo claro se queda con el grueso."""
    base = plt.get_cmap(nombre)
    return mcolors.LinearSegmentedColormap.from_list(
        f'{nombre}_{hasta:g}', base(np.linspace(0.0, hasta, 256)))



# Variantes de color del grid QED vs SA.  Misma geometría, preguntas distintas,
# así que salen como archivos separados (el sufijo va en el nombre):
#   nruns → ¿el hallazgo se repite entre semillas, o lo vio una sola?
#   fsp3  → ¿qué margen sobre el umbral tiene cada punto?
# El valor va solo en el color; todos los marcadores miden lo mismo.  'entero'
# marca las escalas que cuentan cosas: una molécula no aparece en 2.5 runs.
GRID_COLOR_MODES = {
    'nruns': dict(col='n_runs_appeared', cmap=_cmap_recortada('plasma', 0.72),
                  label='Nº de ejecuciones en que aparece', entero=True),
    'fsp3':  dict(col='fsp3', cmap='viridis',
                  label=f'Fsp3 (restricción $\\geq$ {FSP3_MIN:g})'),
    'dock':  dict(col='dock', cmap='viridis',
                  label='Docking, kcal/mol (más oscuro, mejor afinidad)'),
}



def plot_pareto_qed_sa_grid(series, output_dir, fname, color_by='nruns', ncols=3):
    """Un panel por serie (separados, no superpuestos) con el frente QED vs SA.
    Cada serie combina las moléculas de sus runs, deduplica por SMILES y
    recalcula el frente.  color_by elige qué va en el color (GRID_COLOR_MODES)."""
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
        df_unique = unicas(df_all)
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
        # Escala absoluta para comparar paneles, pero arrancando en el umbral: el
        # tramo [0, FSP3_MIN) no puede recibir puntos y gastarlo aplanaría todo
        # el frente en un mismo tono.  El tope sale de los datos, acotado por
        # abajo para que la banda no degenere.
        fsp3_max = max((p['fsp3'].max() for _, p, _ in paretos), default=1.0)
        norm = mcolors.Normalize(vmin=FSP3_MIN,
                                 vmax=max(float(fsp3_max), FSP3_MIN + 0.1))
    elif color_by == 'dock':
        # Escala común a todos los paneles; el mejor docking (más negativo) cae
        # en el extremo oscuro de viridis.
        norm = mcolors.Normalize(
            vmin=min(float(p['dock'].min()) for _, p, _ in paretos),
            vmax=max(float(p['dock'].max()) for _, p, _ in paretos))
    else:
        # 1 run → violeta, max_runs → amarillo
        global_max_runs = max(nr for _, _, nr in paretos)
        norm = mcolors.Normalize(vmin=1, vmax=max(global_max_runs, 2))

    n_plots = len(paretos)
    ncols = min(ncols, n_plots)
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

    # Mismos límites en todos los paneles: es lo que deja ver qué región cubre cada
    # configuración.
    todos = pd.concat([p for _, p, _ in paretos])
    for ax in axes_flat[:n_plots]:
        ax.set_xlim(*_pad_lim(todos['qed'].values))
        ax.set_ylim(*_pad_lim(todos['sa'].values))
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

    title = 'Frentes de Pareto QED vs SA por configuración'
    # Qué codifica el color no va en el título: lo dice la colorbar, y el detalle
    # va en el pie de figura del documento.

    fig.suptitle(title,
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")



# ─── Estructuras moleculares ────────────────────────────────────────────────

# Lienzo de cada estructura, en px.  Las celdas vacías se dibujan con estos
# mismos límites para que todos los recuadros de la grilla midan igual.
LIENZO_MOL = (420, 320)



def render(smiles, size=LIENZO_MOL):
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



# ─── Generación de gráficas para un grupo de series ─────────────────────────

def _generate_report(series, out_dir, report_label, frente=None, completo=False):
    """Genera las gráficas y tablas de un grupo de series, en out_dir/tablas y
    out_dir/figuras.

    Por defecto es el reporte breve: las tres tablas LaTeX, la convergencia
    multiobjetivo y química, los frentes de cada serie por separado y la
    distribución (KDE) de sus propiedades.
    Además escribe los números de las curvas de convergencia (curvas_convergencia.csv).
    completo=True agrega las cajas, los frentes superpuestos, el grid QED-SA con
    sus colores y los demás CSV intermedios.

    frente es el frente de referencia común, (pf_F, pf_df), contra el que se
    miden IGD+ y ε+; sin él se omiten esos indicadores."""
    tablas = os.path.join(out_dir, "tablas")
    figuras = os.path.join(out_dir, "figuras")
    os.makedirs(tablas, exist_ok=True)
    os.makedirs(figuras, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  {report_label}")
    print(f"  Series: {', '.join(s.label for s in series)}")
    print(f"  Salida: {out_dir}")
    print(f"{'─'*60}\n")

    # 1. Indicadores basados en frente de referencia (IGD+, ε+).
    #    Se computan ANTES de las convergencias/boxplots porque alimentan
    #    tanto la gráfica de convergencia MO como las tablas/boxplots.
    indicator_data = {}
    ind_curves = {'igd_plus': {}, 'epsilon': {}}
    if len(series) >= 2 and frente is not None:
        pf_F, pf_df = frente
        print(f"📐 Frente de referencia común: {len(pf_F)} soluciones no-dominadas")
        indicator_data = compute_indicators_per_run(series, pf_F)

        if completo:
            # Guardar frente de referencia
            pf_df.to_csv(os.path.join(tablas, "frente_referencia.csv"), index=False)
            print("  ✓ frente_referencia.csv")

            # Guardar indicadores por run
            ind_rows = []
            for label, df_ind in indicator_data.items():
                for _, row in df_ind.iterrows():
                    ind_rows.append({'series': label, **row.to_dict()})
            if ind_rows:
                pd.DataFrame(ind_rows).to_csv(
                    os.path.join(tablas, "indicadores_por_corrida.csv"), index=False)
                print("  ✓ indicadores_por_corrida.csv")

        # Curvas de convergencia de indicadores por generación (sin re-entrenar)
        print("📈 Curvas de convergencia de indicadores (IGD+, ε+)...")
        ind_curves = _indicator_curves(series, tablas, pf_F, guardar_csv=completo)

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
        'uniqueness':  _conv_csv_curves(series, 'uniqueness'),
        'novelty':     _conv_csv_curves(series, 'novelty'),
        'qed':         _objective_curves(series, 'qed'),
        'sa':          _objective_curves(series, 'sa'),
        'dock':        _objective_curves(series, 'dock'),
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
        paneles = [(PANELES_MO, '1_convergencia_multiobjetivo.png',
                    'Convergencia de Indicadores Multiobjetivo'),
                   (PANELES_QUIM, '2_convergencia_quimica.png',
                    'Convergencia de Indicadores Químicos')]
        for specs, fname, titulo in paneles:
            _plot_convergence_grid(series, figuras, specs, curvas, mapas_eval,
                                   fname, titulo)
        escribir_curvas_csv(series, curvas, mapas_eval, tablas)

    # 5. Boxplots + tablas (requiere ≥2 series).  Comparten un único getter
    #    de valores per-run (metrics + indicadores + medias químicas + unicidad).
    if len(series) >= 2:
        get_values = _build_series_value_getter(series, indicator_data)

        if completo:
            print("📊 Boxplots multiobjetivo (HV, Espaciamiento, IGD+, ε+, Pareto)...")
            plot_boxplots(series, figuras, get_values, BOXPLOT_MO_CONFIGS,
                          "cajas_multiobjetivo.png",
                          "Distribución de Indicadores Multiobjetivo")
            print("📊 Boxplots químicos (QED, SA, Docking, Fsp3, Validez, "
                  "Unicidad, Novedad)...")
            plot_boxplots(series, figuras, get_values, BOXPLOT_CHEM_CONFIGS,
                          "cajas_quimicas.png",
                          "Distribución de Indicadores Químicos")

            print("📋 Tabla estadística...")
            generate_statistical_table(series, tablas,
                                       indicator_data=indicator_data)

        print("📋 Tablas LaTeX de comparación...")
        generate_latex_comparison_tables(series, tablas, get_values)

    # 7. Los frentes de cada serie por separado: un panel por configuración, en
    #    pares (una fila por algoritmo: PCX a la izquierda, SBX a la derecha), QED
    #    vs SA con el docking en el color.  Con completo, también los otros colores
    #    del grid y las seis superpuestas.
    print("🧩 Frentes por configuración...")
    plot_pareto_qed_sa_grid(series, figuras, "3_frentes_por_configuracion.png",
                            color_by='dock', ncols=len(CRUCES))
    print("📈 Distribución de las propiedades del frente (KDE)...")
    plot_kde_propiedades(series, figuras, "4_distribucion_propiedades.png")
    if completo:
        for modo in GRID_COLOR_MODES:
            if modo != 'dock':
                plot_pareto_qed_sa_grid(series, figuras, f"grid_qed_sa_{modo}.png",
                                        color_by=modo, ncols=len(CRUCES))
        if len(series) >= 2:
            plot_pareto_comparison(series, figuras, "frentes_superpuestos.png")
