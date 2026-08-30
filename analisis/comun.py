"""
Estilo, series, carga de resultados, tests estadísticos y tablas LaTeX.

Lo que usan todas las etapas.  Una Series es cualquier directorio con
run_XX/{metrics,molecules,convergence}.csv.

Los tests toman la semilla como bloque (las 20 están pareadas): Friedman y, si
da significativo, Wilcoxon por pares con Holm, resumido en grupos homogéneos.
"""

import glob
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use('Agg')

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



# El paquete cuelga de la raíz del repo, de ahí el dirname doble.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# Nombres de presentación para los captions de las figuras: solo algoritmos.
# DISPLAY, más abajo, es el de las etapas e incluye las baselines.
DISPLAY_ALG = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
               'AGEMOEA': 'AGE-MOEA', 'CMOPSO': 'CMOPSO'}

_ALG_POR_DISPLAY = {v: k for k, v in DISPLAY_ALG.items()}


# Todas las series usan el mismo marcador (punto): se distinguen por color,
# no por forma.
PARETO_MARKER = 'o'


# Tamaño del marcador en los frentes 2D, según la densidad del panel:
#   pareto_comparison superpone las 5 series      → ~58 pts/in²
#   el grid QED-SA y el frente conjunto, un frente → ~12-15 pts/in²
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
    """Una serie a comparar: una curva/caja/frente en las gráficas.  En el modo
    algoritmos cada algoritmo es una serie; en operadores, cada combo.

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



# ─── Contribución al frente no dominado conjunto ─────────────────────────────
#
#   El hipervolumen mide la extensión del frente, no la calidad de lo que hay
#   dentro: se puede ganar volumen estirándose hacia un extremo con el grueso
#   dominado.  Acá se junta lo de todos los combos, se recalcula la no-dominancia
#   global y se mira quién aportó los supervivientes.

# Okabe-Ito: naranja/azul se distinguen bajo los tres tipos de daltonismo.
# CMOPSO entra acá porque en el frente conjunto de candidatos convive con las dos
# familias de cruce sin pertenecer a ninguna: no tiene operadores.  Su rojo va
# oscurecido para que no se confunda con el naranja de PCX.
CRUCE_COLORS = {'PCX': '#D55E00', 'SBX': '#0072B2', 'CMOPSO': '#B01818'}

COMPARTIDA_COLOR = '#7F7F7F'



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



# Separador decimal de todas las tablas LaTeX.  Para coma usar '{,}': las llaves
# hacen que en modo matemático LaTeX la trate como símbolo ordinario y no como
# puntuación, que llevaría un espacio detrás.
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
    """get(label, col) → array de valores per-run (o None), buscando en la fuente
    que corresponda:
      - mean_qed/mean_sa/mean_fsp3 → medias del frente final (molecules.csv)
      - igd_plus/epsilon           → indicadores vs frente de referencia
      - uniqueness                 → última generación (convergence.csv)
      - el resto                   → metrics.csv
    Calcula las fuentes una sola vez, para tablas y boxplots."""
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


# El cluster baja UN tar (lo arma train.sh) y se extrae en la
# raíz del repo; cada cosa cae ya en su lugar:
#   grid/        SOLO all_metrics.csv, que es lo único que lee la etapa 1.  El
#                árbol del grid (10.260 runs) no lo consume nadie y no viaja.
#   winners/     las 17 configuraciones que ganaron su bloque en la etapa 1, con
#                sus runs completas (incluido all_molecules.csv.gz).
#   finalistas/  symlinks a la ganadora de cada algoritmo en winners/.  Lo armás
#                vos en el PC, después de la etapa 2.
#   baselines/   NO viene del cluster: las baselines se corren y se analizan acá
#                (baselines.py escribe en results_baselines/).
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")

METRICS_CSV    = os.path.join(RESULTADOS_DIR, "grid", "all_metrics.csv")

WINNERS_DIR    = os.path.join(RESULTADOS_DIR, "winners")

FINALISTAS_DIR = os.path.join(RESULTADOS_DIR, "finalistas")

BASELINES_DIR  = os.path.join(RESULTADOS_DIR, "baselines")

OUT_HP         = os.path.join(PLOTS_DIR, "hiperparametros")

OUT_OPERADORES = os.path.join(PLOTS_DIR, "operadores")

OUT_ALGORITMOS = os.path.join(PLOTS_DIR, "comparacion_final")

OUT_BASELINES  = os.path.join(PLOTS_DIR, "baselines")

# El frente conjunto va aparte: no compara algoritmos entre sí como el resto de
# la etapa 3, sino que caracteriza qué moléculas sobreviven al unirlos y de dónde
# salen.  Son preguntas distintas y conviene que no se mezclen en la lectura.
OUT_FRENTE     = os.path.join(PLOTS_DIR, "frente_conjunto")


# Nombres para el documento (los directorios usan la forma corta).
DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'CMOPSO': 'CMOPSO',
           'RANDOM': 'Aleatorio', 'WEIGHTED_GA': 'GA ponderado',
           'SCREENING': 'Cribado MOSES', 'HILL_CLIMBER': 'Escalador'}


# El algoritmo de enjambre de esta etapa.  Se nombra una sola vez: es el que no
# tiene operadores de cruce/mutación y por eso queda fuera de la comparación de
# operadores, del pool por familias y de los factores GA.  CMOPSO reemplazó al
# MOPSO_CD anterior porque maneja el constraint de forma nativa.
PSO_ALG = 'CMOPSO'



def _num(x, dec):
    """Número con el separador decimal del documento (SEP_DECIMAL)."""
    return f'{x:.{dec}f}'.replace('.', SEP_DECIMAL)



def _fmt_p(p):
    if p is None or pd.isna(p):
        return '---'
    return f'$<$0{SEP_DECIMAL}001' if p < 1e-3 else _num(p, 3)



def fmt_groups(groups):
    """'{A, B} $>$ {C}' con los nombres de presentación.

    Escapa los nombres: los combos de operadores llevan guion bajo (pcx\\_pm) y
    sin escapar rompen la compilación."""
    return ' $>$ '.join(
        '\\{' + ', '.join(_latex_escape(DISPLAY.get(x, x)) for x in g) + '\\}'
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



def rank_biserial(x, y):
    """Correlación rango-biserial de pares emparejados (Kerby, 2014).

    Tamaño de efecto que acompaña al Wilcoxon: de -1 a +1, el signo dice quién
    gana y el valor absoluto qué fracción de la evidencia lo respalda.  Satura en
    ±1 cuando todos los pares van igual, así que mide unanimidad, no magnitud.
    Verificado contra pingouin.wilcoxon()['RBC'].
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]                      # los empates exactos no aportan rango
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())



# No se etiqueta la magnitud (pequeño/mediano/grande): los umbrales que circulan
# se derivan de convertir los cortes de Cohen a distintas escalas y no coinciden
# entre fuentes, ninguna pensada para la versión de pares emparejados.  El valor
# se reporta crudo.


def compare_indicator(get_values, labels, col):
    """Friedman sobre los métodos con la semilla como bloque + post-hoc.

    Devuelve dict con el p de Friedman, la mediana de cada método y la lista de
    comparaciones por pares con su p corregido por Holm; None si falta algún
    método o hay menos de 3 semillas."""
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

    pairs, raw = [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            p = float(stats.wilcoxon(M[:, i], M[:, j]).pvalue)
        except ValueError:      # todas las diferencias son cero
            p = 1.0
        pairs.append((labels[i], labels[j]))
        raw.append(p)
    adj = holm(raw) if raw else []

    return {'p_omnibus': p_omni,
            'medians': {lab: float(np.median(cols[lab][:n])) for lab in labels},
            'pairs': [{'a': a, 'b': b, 'p_raw': pr, 'p_holm': pa}
                      for (a, b), pr, pa in zip(pairs, raw, adj)]}



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
#   Etapa 2 — comparación de combinaciones de operadores, por algoritmo
#
#   Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (lo que ganó su bloque
#   en la etapa 1) y compara los 4 combos entre sí, un reporte por algoritmo.
# ═══════════════════════════════════════════════════════════════════════════

GA_ALGS = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA']
