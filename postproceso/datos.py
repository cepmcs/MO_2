"""
Carga de resultados y cálculo de indicadores.

Los experimentos escriben en el cluster bajo results/ y results_baselines/.
Lo que se baja al PC vive en resultados/:

  grid/        copia liviana de results/ (solo metrics.csv y molecules.csv),
               más el all_metrics.csv del grid completo
  winners/     las 17 configuraciones que ganaron su bloque en la etapa 1
  finalistas/  symlink a la ganadora de cada algoritmo dentro de winners/
  baselines/   copia de results_baselines/
"""

import glob
import os

import numpy as np
import pandas as pd
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
METRICS_CSV    = os.path.join(RESULTADOS_DIR, "grid", "all_metrics.csv")
WINNERS_DIR    = os.path.join(RESULTADOS_DIR, "winners")
FINALISTAS_DIR = os.path.join(RESULTADOS_DIR, "finalistas")
BASELINES_DIR  = os.path.join(RESULTADOS_DIR, "baselines")

PLOTS_DIR      = os.path.join(ROOT_DIR, "plots")
OUT_HP         = os.path.join(PLOTS_DIR, "hiperparametros")
OUT_OPERADORES = os.path.join(PLOTS_DIR, "operadores")
OUT_ALGORITMOS = os.path.join(PLOTS_DIR, "comparacion_final")
OUT_BASELINES  = os.path.join(PLOTS_DIR, "baselines")

# Objetivos y su signo al minimizar: pymoo trabaja con [-QED, SA, -Lipinski].
OBJETIVOS = ('qed', 'sa', 'lipinski')
SIGNOS    = np.array([-1.0, 1.0, -1.0])

ALGORITHM_ORDER = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'MOPSO']
COMBO_DIRS      = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']
BASELINE_KEYS   = ['WEIGHTED_GA', 'HILL_CLIMBER', 'RANDOM', 'SCREENING']

# Nombres para el documento; los directorios usan la forma corta.
DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO',
           'RANDOM': 'Aleatorio', 'WEIGHTED_GA': 'GA ponderado',
           'SCREENING': 'Cribado MOSES', 'HILL_CLIMBER': 'Escalador'}


def display(label):
    return DISPLAY.get(label, label)


# ─── Vocabulario de métricas ─────────────────────────────────────────────────

# columna → (etiqueta, mayor_es_mejor)
HP_METRICS = {
    'hypervolume': ('Hipervolumen', True),
    'spacing':     ('Espaciamiento', False),
    'n_pareto':    ('Tamaño de Pareto', True),
    'validity':    ('Validez', True),
    'novelty':     ('Novedad', True),
    'best_sa':     ('Mejor SA', False),
    'time_sec':    ('Tiempo (s)', False),
}

# Indicadores que compara la etapa 2: (columna, etiqueta, mayor_es_mejor).
OP_INDICATORS = [
    ('hypervolume', 'Hipervolumen',     True),
    ('igd_plus',    'IGD$^+$',          False),
    ('epsilon',     r'$\epsilon^+$',    False),
    ('spacing',     'Espaciamiento',    False),
    ('n_pareto',    'Tamaño de Pareto', True),
    ('validity',    'Validez',          True),
    ('uniqueness',  'Unicidad',         True),
]

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

# Combos tal como aparecen en el CSV del grid; COMBO_DIRS es la misma lista como
# nombres de directorio bajo winners/.
HP_COMBOS = ['pcx/pm', 'pcx/gauss', 'sbx/pm', 'sbx/gauss']

OBJECTIVE_LABELS = {'qed': 'QED (↑)', 'sa': 'SA (↓)', 'lipinski': 'Lipinski (↑)'}


# ─── Series ──────────────────────────────────────────────────────────────────

class Series:
    """Una serie a comparar: una curva, una caja y un frente en cada gráfica.

    label      texto en leyendas y títulos
    path       directorio que contiene los run_XX/
    color_key  clave de color (por defecto, el label)
    """

    def __init__(self, label, path, color_key=None):
        self.label = label
        self.path = path
        self.color_key = color_key if color_key is not None else label


def has_runs(path):
    """True si el directorio tiene al menos una run con convergence.csv."""
    return bool(glob.glob(os.path.join(path, "run_*", "convergence.csv")))


def series_operadores(alg, winners_dir):
    """Una serie por combinación de operadores de un algoritmo.  El nivel de
    configuración se resuelve con glob: cada combo ganó con hiperparámetros
    distintos."""
    series = []
    for combo in COMBO_DIRS:
        cfgs = [d for d in sorted(glob.glob(os.path.join(winners_dir, alg, combo, '*')))
                if has_runs(d)]
        if cfgs:
            series.append(Series(combo, cfgs[0], color_key=combo))
    return series


def series_finalistas(finalistas_dir):
    """Una serie por algoritmo, en el orden de presentación."""
    return [Series(alg, os.path.join(finalistas_dir, alg), color_key=alg)
            for alg in ALGORITHM_ORDER
            if has_runs(os.path.join(finalistas_dir, alg))]


def series_baselines(finalistas_dir, baselines_dir):
    """Los algoritmos finalistas seguidos de las baselines."""
    series = series_finalistas(finalistas_dir)
    for m in BASELINE_KEYS:
        # <baselines>/<METODO>/[pesos/]pop{P}_gen{G}/
        for d in sorted(glob.glob(os.path.join(baselines_dir, m, '*', '*')) +
                        glob.glob(os.path.join(baselines_dir, m, '*'))):
            if has_runs(d):
                series.append(Series(m, d, color_key=m))
                break
    return series


# ─── Lectura de CSV ──────────────────────────────────────────────────────────

def _concat_runs(path, fname, add_run_col=True):
    """Concatena el mismo CSV de todas las runs de una serie."""
    dfs = []
    for run_dir in sorted(glob.glob(os.path.join(path, "run_*"))):
        csv_path = os.path.join(run_dir, fname)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if add_run_col:
            df['run'] = os.path.basename(run_dir)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_convergence(path):
    """convergence.csv de todas las runs: gen, hv, validity, uniqueness, novelty."""
    return _concat_runs(path, "convergence.csv")


def load_metrics(path):
    """Métricas por run.  Acepta un metrics.csv agregado a nivel de serie o los
    metrics.csv individuales de cada run."""
    agregado = os.path.join(path, "metrics.csv")
    if os.path.exists(agregado):
        return pd.read_csv(agregado)
    return _concat_runs(path, "metrics.csv", add_run_col=False)


def load_pareto_molecules(path):
    """molecules.csv (frente final) de todas las runs de una serie."""
    return _concat_runs(path, "molecules.csv")


def load_grid(csv_path):
    """all_metrics.csv del grid, con la columna derivada 'budget'."""
    df = pd.read_csv(csv_path)
    df['budget'] = (df['pop_size'].astype(int).astype(str) + '×'
                    + df['n_gen'].astype(int).astype(str))
    return df


# ─── Frentes de Pareto ───────────────────────────────────────────────────────

def df_to_F(df):
    """DataFrame de moléculas → matriz de objetivos de minimización."""
    return df[list(OBJETIVOS)].to_numpy(dtype=float) * SIGNOS


def tiene_objetivos(df):
    return not df.empty and set(OBJETIVOS).issubset(df.columns)


def compute_non_dominated(df):
    """Filtra el frente no dominado de un DataFrame de moléculas."""
    if not tiene_objetivos(df):
        return df
    idx = NonDominatedSorting().do(df_to_F(df), only_non_dominated_front=True)
    return df.iloc[idx].reset_index(drop=True)


def frente_global(path):
    """Frente de una serie sobre todas sus runs: junta las moléculas, deduplica
    por SMILES y recalcula la dominancia."""
    df = load_pareto_molecules(path)
    if df.empty:
        return df
    return compute_non_dominated(df.drop_duplicates(subset='smiles'))


def build_reference_front(series):
    """Frente de referencia combinando todas las runs de todas las series.

    Procedimiento estándar cuando el frente verdadero es desconocido.  Devuelve
    (pf_F, pf_df) o (None, None) si no hay datos."""
    dfs = [df for df in (load_pareto_molecules(s.path) for s in series)
           if not df.empty]
    if not dfs:
        return None, None
    combinado = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='smiles')
    pf_df = compute_non_dominated(combinado)
    if pf_df.empty:
        return None, None
    return df_to_F(pf_df), pf_df


# ─── Indicadores contra el frente de referencia ──────────────────────────────

def _bounds(pf_F):
    """Ideal y escala del frente de referencia.  Normalizar con esto hace que
    los tres objetivos pesen igual en IGD+ y ε+; sin normalizar, SA domina por
    tener el rango más grande."""
    ideal = pf_F.min(axis=0)
    escala = pf_F.max(axis=0) - ideal
    return ideal, np.where(escala > 1e-12, escala, 1.0)


def _normalize(F, ideal, escala):
    return (F - ideal) / escala


def additive_epsilon(F, pf):
    """ε+ = max_j min_i max_k (F_ik − PF_jk): el desplazamiento uniforme mínimo
    para que F cubra todo el frente de referencia.  Menor es mejor.

    pymoo 0.6 no lo trae.  F y pf tienen que venir normalizados con los mismos
    bounds para que las coordenadas sean comparables."""
    return float(max((F - pf[j]).max(axis=1).min() for j in range(len(pf))))


def compute_indicators_per_run(series, pf_F):
    """IGD+ y ε+ de cada run contra el frente de referencia.
    Devuelve {label: DataFrame con [run, igd_plus, epsilon]}."""
    ideal, escala = _bounds(pf_F)
    pf_n = _normalize(pf_F, ideal, escala)
    igd_plus = IGDPlus(pf_n)

    resultados = {}
    for s in series:
        filas = []
        for run_dir in sorted(glob.glob(os.path.join(s.path, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            if not tiene_objetivos(df):
                continue
            F = _normalize(df_to_F(df), ideal, escala)
            filas.append({'run': os.path.basename(run_dir),
                          'igd_plus': float(igd_plus(F)),
                          'epsilon': additive_epsilon(F, pf_n)})
        if filas:
            resultados[s.label] = pd.DataFrame(filas)
    return resultados


# ─── Valores por run para tablas y boxplots ──────────────────────────────────

def _chemical_means(series):
    """Media por run de cada objetivo sobre el frente final (no el mejor valor)."""
    resultados = {}
    for s in series:
        filas = []
        for run_dir in sorted(glob.glob(os.path.join(s.path, "run_*"))):
            mol_path = os.path.join(run_dir, "molecules.csv")
            if not os.path.exists(mol_path):
                continue
            df = pd.read_csv(mol_path)
            if not tiene_objetivos(df):
                continue
            filas.append({'run': os.path.basename(run_dir),
                          **{f'mean_{o}': df[o].mean() for o in OBJETIVOS}})
        if filas:
            resultados[s.label] = pd.DataFrame(filas)
    return resultados


def _uniqueness_final(series):
    """Unicidad de la última generación de cada run (la que trackea
    GenerationTracker), como medida de diversidad de la población final."""
    resultados = {}
    for s in series:
        filas = []
        for run_dir in sorted(glob.glob(os.path.join(s.path, "run_*"))):
            conv_path = os.path.join(run_dir, "convergence.csv")
            if not os.path.exists(conv_path):
                continue
            df = pd.read_csv(conv_path)
            if df.empty or 'uniqueness' not in df.columns:
                continue
            filas.append({'run': os.path.basename(run_dir),
                          'uniqueness': float(df['uniqueness'].iloc[-1])})
        if filas:
            resultados[s.label] = pd.DataFrame(filas)
    return resultados


def value_getter(series, indicadores=None):
    """Devuelve get(label, col) → array de valores por run, o None.

    Cada columna sale de su fuente: mean_* del frente final, igd_plus/epsilon de
    los indicadores, uniqueness de la última generación, y el resto (hypervolume,
    spacing, validity, novelty, n_pareto, time_sec…) de metrics.csv.  Las fuentes
    se leen una sola vez y se reusan en tablas y boxplots."""
    indicadores = indicadores or {}
    metricas = {s.label: load_metrics(s.path) for s in series}
    quimicos = _chemical_means(series)
    unicidad = _uniqueness_final(series)

    def get(label, col):
        if col.startswith('mean_'):
            df = quimicos.get(label)
        elif col in ('igd_plus', 'epsilon'):
            df = indicadores.get(label)
        elif col == 'uniqueness':
            df = unicidad.get(label)
        else:
            df = metricas.get(label)
        if df is None or df.empty or col not in df.columns:
            return None
        vals = df[col].dropna().values
        return vals if len(vals) else None

    return get


# ─── Curvas de convergencia ──────────────────────────────────────────────────

def _smooth(values, window):
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def convergence_curves(series, metric):
    """Curva de una métrica de convergence.csv, promediada sobre las runs.
    Devuelve {label: (gens, valores)}."""
    curvas = {}
    for s in series:
        df = load_convergence(s.path)
        if df.empty or metric not in df.columns:
            print(f"  ⚠ {s.label}: sin datos de '{metric}'")
            continue
        media = df.groupby('gen')[metric].mean()
        curvas[s.label] = (media.index.values, _smooth(media.values, 20))
    return curvas


def objective_curves(series, objetivo):
    """Curva del promedio de un objetivo por generación, leída del log completo
    de evaluaciones (all_molecules.csv.gz)."""
    curvas = {}
    for s in series:
        medias = []
        for run_dir in sorted(glob.glob(os.path.join(s.path, "run_*"))):
            gz = os.path.join(run_dir, "all_molecules.csv.gz")
            if not os.path.exists(gz):
                continue
            try:
                df = pd.read_csv(gz, usecols=['gen', objetivo])
            except (ValueError, OSError):
                continue
            medias.append(df.groupby('gen')[objetivo].mean())
        if not medias:
            print(f"  ⚠ {s.label}: sin datos de '{objetivo}' en all_molecules.csv.gz")
            continue
        media = pd.concat(medias, axis=1).mean(axis=1)
        curvas[s.label] = (media.index.values, _smooth(media.values, 20))
    return curvas


def indicator_curves(series, pf_F, gen_stride=10):
    """IGD+ y ε+ por generación, reconstruyendo el frente de cada generación
    desde all_molecules.csv.gz.  gen_stride submuestrea generaciones (1 = todas).

    Devuelve ({col: {label: (gens, valores)}}, DataFrame largo para el CSV)."""
    ideal, escala = _bounds(pf_F)
    pf_n = _normalize(pf_F, ideal, escala)
    igd_plus = IGDPlus(pf_n)

    por_serie = {}
    for s in series:
        curvas_run = []
        for run_dir in sorted(glob.glob(os.path.join(s.path, "run_*"))):
            gz = os.path.join(run_dir, "all_molecules.csv.gz")
            if not os.path.exists(gz):
                continue
            try:
                df = pd.read_csv(gz, usecols=['gen', 'valid', *OBJETIVOS])
            except (ValueError, OSError):
                continue
            df = df[df['valid'].astype(bool)].dropna(subset=list(OBJETIVOS))
            if df.empty:
                continue
            filas = []
            for g in sorted(df['gen'].unique())[::gen_stride]:
                frente = compute_non_dominated(df[df['gen'] == g])
                if frente.empty:
                    continue
                F = _normalize(df_to_F(frente), ideal, escala)
                filas.append({'gen': g, 'igd_plus': float(igd_plus(F)),
                              'epsilon': additive_epsilon(F, pf_n)})
            if filas:
                curvas_run.append(pd.DataFrame(filas).set_index('gen'))
        if curvas_run:
            por_serie[s.label] = pd.concat(curvas_run).groupby(level=0).mean()

    if not por_serie:
        print("  ⚠ sin all_molecules.csv.gz para la convergencia de indicadores")
        return {'igd_plus': {}, 'epsilon': {}}, pd.DataFrame()

    curvas = {col: {label: (c.index.values, _smooth(c[col].values, 5))
                    for label, c in por_serie.items()}
              for col in ('igd_plus', 'epsilon')}
    largo = pd.concat([c.assign(series=label).reset_index()
                       for label, c in por_serie.items()], ignore_index=True)
    return curvas, largo
