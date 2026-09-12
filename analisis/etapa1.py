"""
Etapa 1 — el análisis en tres pasos.

  paso 1   las tres configuraciones de mutación dentro de 100×1000.
  paso 2   las mismas tres dentro de 200×500.
  paso 3   los dos presupuestos, con la mutación que ganó los dos pasos.

La mutación se compara por separado en cada presupuesto y no agregando sobre
ambos: así no se asume que la mejor sea la misma, se muestra.  Y respeta el
orden que pidió Pablo, primero 100×1000 y después 200×500.

Cada paso lleva el test que ya usaba el proyecto —Friedman con las semillas como
bloques y Wilcoxon por pares con corrección de Holm— y las dos tablas de
métricas.  Todo se corre con IGD+ y con hipervolumen: si las dos coinciden, la
elección no depende de la métrica, que es lo que hay que poder afirmar.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from .comun import SEP_DECIMAL, _latex_escape, _write_tex
from .indicadores import (_additive_epsilon, _compute_non_dominated, _df_to_F,
                          _front_bounds, _normalize_F)


ORDEN_ALG = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA', 'CMOPSO']

DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'CMOPSO': 'CMOPSO'}


# (columna, encabezado, decimales, dirección)
MULTIOBJETIVO = [
    ('hypervolume', 'Hipervolumen',     4, '↑'),
    ('igd_plus',    'IGD+',             4, '↓'),
    ('epsilon',     'ε+',               4, '↓'),
    ('spacing',     'Espaciamiento',    4, '↓'),
    ('n_pareto',    'Tamaño de Pareto', 1, '↑'),
    ('time_sec',    'Tiempo (s)',       1, ''),
]

QUIMICA = [
    ('best_qed',    'QED',          4, '↑'),
    ('best_sa',     'SA',           2, '↓'),
    ('feasibility', 'Factibilidad', 4, '↑'),
    ('mean_fsp3',   'Fsp3',         4, ''),
    ('validity',    'Validez',      4, '↑'),
    ('novelty',     'Novedad',      4, '↑'),
]


# ─── Carga ───────────────────────────────────────────────────────────────────

def cargar(results_dir):
    """Una fila por corrida, con las columnas que agrupan el experimento."""
    archivos = sorted(glob.glob(os.path.join(results_dir, '**', 'run_*',
                                             'metrics.csv'), recursive=True))
    if not archivos:
        raise SystemExit(f"no hay run_*/metrics.csv bajo {results_dir}")
    d = pd.concat([pd.read_csv(f).assign(_dir=os.path.dirname(f))
                   for f in archivos], ignore_index=True)

    d['config'] = np.where(d.mutation.str.endswith('_adapt'), 'adaptativa',
                           d.mut_prob.map('{:g}'.format))
    d['tipo'] = d.mutation.str.replace('_adapt', '', regex=False)
    d['cruce'] = d.crossover.fillna('—').str.upper()
    d['reparto'] = d.pop_size.astype(str) + 'x' + d.n_gen.astype(str)
    d['algoritmo'] = d.algorithm.map(DISPLAY)
    d['_orden'] = d.algorithm.map({a: i for i, a in enumerate(ORDEN_ALG)})
    return d.sort_values(['_orden', 'cruce', 'tipo'])


def agregar_indicadores(d):
    """IGD+ y ε+ de cada corrida contra un único frente de referencia.

    El frente junta todo lo corrido y recalcula la no-dominancia: es el
    procedimiento estándar cuando el frente verdadero se desconoce.  Tiene que
    ser uno solo para todo el experimento, y no uno por reparto, porque si no
    cada reparto se mide contra una vara que arma él mismo y la comparación
    entre repartos no significa nada.  Los dos indicadores se normalizan con los
    mismos bounds para que QED y SA pesen igual."""
    from pymoo.indicators.igd_plus import IGDPlus

    d['igd_plus'] = np.nan
    d['epsilon'] = np.nan
    frentes = {}
    for carpeta in d._dir:
        p = os.path.join(carpeta, 'molecules.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            if not df.empty and {'qed', 'sa'}.issubset(df.columns):
                frentes[carpeta] = df
    if not frentes:
        return d
    juntos = pd.concat(frentes.values(), ignore_index=True)
    pf = _compute_non_dominated(juntos.drop_duplicates(subset='smiles'))
    ideal, escala = _front_bounds(_df_to_F(pf))
    pf_n = _normalize_F(_df_to_F(pf), ideal, escala)
    igd = IGDPlus(pf_n)
    for carpeta, df in frentes.items():
        F = _normalize_F(_df_to_F(df), ideal, escala)
        i = d.index[d._dir == carpeta]
        d.loc[i, 'igd_plus'] = float(igd(F))
        d.loc[i, 'epsilon'] = _additive_epsilon(F, pf_n)
    print(f"    frente de referencia único con {len(pf)} moléculas")
    return d


# ─── Tests ───────────────────────────────────────────────────────────────────

def comparar(sub, niveles, factor, col, mayor, bloques):
    """Friedman sobre los niveles de un factor, con los bloques pareados, y el
    post-hoc de Wilcoxon con Holm resumido en grupos homogéneos."""
    import itertools
    from .comun import holm, homogeneous_groups
    M = sub.pivot_table(index=bloques, columns=factor, values=col).dropna()
    niveles = [n for n in niveles if n in M.columns]
    if len(niveles) < 2 or M.empty:
        return None
    M = M[niveles]
    # Friedman necesita tres o más; con dos niveles el ómnibus es el propio
    # Wilcoxon pareado, así que se usa ese.
    pares = list(itertools.combinations(niveles, 2))
    p = (stats.friedmanchisquare(*[M[c] for c in niveles]).pvalue
         if len(niveles) > 2
         else stats.wilcoxon(M[niveles[0]], M[niveles[1]]).pvalue)
    crudos = [stats.wilcoxon(M[a], M[b]).pvalue for a, b in pares]
    ajust = holm(crudos)
    med = {c: float(M[c].median()) for c in niveles}
    res = {'p_omnibus': p, 'medians': med,
           'pairs': [{'a': a, 'b': b, 'p_raw': r, 'p_holm': h}
                     for (a, b), r, h in zip(pares, crudos, ajust)]}
    res['grupos'] = homogeneous_groups(res, niveles, med, mayor)
    res['n'] = len(M)
    return res


def test_por_algoritmo(d, niveles, factor, col, mayor, texto, sangria='    '):
    """El mismo test dentro de cada algoritmo.

    El test agregado responde si el factor importa en general; este responde si
    algún algoritmo va en contra, que es lo que hay que poder descartar.  CMOPSO
    no tiene cruce ni gaussiana, así que sus bloques son más chicos."""
    texto += [f'{sangria}por algoritmo:']
    for alg in ORDEN_ALG:
        g = d[d.algorithm == alg]
        if g.empty:
            continue
        bloques = ['run', 'config'] if alg == 'CMOPSO' else ['run', 'tipo', 'cruce']
        bloques = [b for b in bloques if b != factor]
        res = comparar(g, niveles, factor, col, mayor, bloques)
        if res is None:
            continue
        gana = res['grupos'][0]
        sig = res['p_omnibus'] < 0.05
        texto += [f'{sangria}  {DISPLAY[alg]:9s} p = {res["p_omnibus"]:.2e}   '
                  + ('gana ' + ', '.join(gana) if sig else 'sin diferencia')]
    texto += ['']


def texto_test(res, titulo):
    if res is None:
        return [f'  {titulo}: sin datos suficientes', '']
    return [
        f'  {titulo}   {"Friedman" if len(res["medians"]) > 2 else "Wilcoxon"} '
        f'p = {res["p_omnibus"]:.2e}   ({res["n"]} bloques)',
        '    medianas: ' + '   '.join(f'{k}={v:.4f}' for k, v in res['medians'].items()),
        '    grupos:   ' + ' > '.join('{' + ', '.join(g) + '}' for g in res['grupos']),
        '    pares:    ' + '   '.join(f'{p["a"]} vs {p["b"]}: p={p["p_holm"]:.4f}'
                                      for p in res['pairs']), '']


# ─── Rango medio ─────────────────────────────────────────────────────────────

def rango_medio(d, factor, dentro, metrica='hypervolume'):
    """Rango medio de cada nivel de un factor (1 = mejor), rankeando dentro del
    resto: cada nivel compite sobre la misma semilla y el mismo algoritmo, así
    que las escalas distintas de cada algoritmo no entran.

    Descarta los niveles ausentes en algún bloque antes que los bloques, si no un
    nivel corrido en un solo caso se lleva puestos casi todos."""
    M = d.pivot_table(index=dentro, columns=factor, values=metrica)
    M = M.drop(columns=list(M.columns[~M.notna().all()])).dropna()
    if M.empty or M.shape[1] < 2:
        return pd.Series(dtype=float), 0
    R = np.apply_along_axis(stats.rankdata, 1, -M.values)
    return pd.Series(R.mean(axis=0), index=M.columns).round(2), len(M)


# ─── Tablas ──────────────────────────────────────────────────────────────────

def tabla(d, filas, columnas, rangos=None):
    """Una fila por combinación de `filas`; media ± desvío de cada métrica.
    Con `rangos` se antepone la columna de rango medio, que da el orden."""
    out = []
    for clave, g in d.groupby(filas, sort=False):
        clave = clave if isinstance(clave, tuple) else (clave,)
        fila = dict(zip(filas, clave))
        if rangos is not None:
            fila['rango'] = rangos.get(clave[-1], np.nan)
        for col, enc, dec, flecha in columnas:
            if col not in g or g[col].isna().all():
                continue
            fila[f'{enc} {flecha}'.strip()] = (
                f'{g[col].mean():.{dec}f} ± {g[col].std():.{dec}f}')
        out.append(fila)
    return pd.DataFrame(out).set_index(filas) if out else pd.DataFrame()


def _tex(t, titulo, etiqueta, ruta):
    idx = list(t.index.names)
    lineas = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{{titulo}}}', f'\\label{{{etiqueta}}}',
        r'\begin{tabular}{' + 'l' * len(idx) + 'c' * len(t.columns) + '}',
        r'\toprule',
        ' & '.join(_latex_escape(c.capitalize()) for c in idx) + ' & ' +
        ' & '.join(_latex_escape(c) for c in t.columns) + r' \\', r'\midrule',
    ]
    previo = None
    for clave, fila in t.iterrows():
        clave = clave if isinstance(clave, tuple) else (clave,)
        if previo is not None and clave[0] != previo:
            lineas.append(r'\midrule')
        celdas = ['' if i == 0 and clave[0] == previo else _latex_escape(v)
                  for i, v in enumerate(clave)]
        for v in fila:
            v = str(v)
            celdas.append(f"${v.replace('±', chr(92)+'pm').replace('.', SEP_DECIMAL)}$"
                          if '±' in v else _latex_escape(v))
        lineas.append(' & '.join(celdas) + r' \\')
        previo = clave[0]
    lineas += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lineas, ruta, msg=os.path.basename(ruta))


def _emitir(t, nombre, titulo, out_dir, texto, cabecera):
    if t.empty:
        return
    t.to_csv(os.path.join(out_dir, f'{nombre}.csv'))
    _tex(t, titulo, f'tab:{nombre}', os.path.join(out_dir, f'{nombre}.tex'))
    texto += ['=' * 110, cabecera, '=' * 110, t.to_string(), '']


# ─── Los tres pasos ──────────────────────────────────────────────────────────

CONFIGS = ['0.05', '0.1', 'adaptativa']

# Las dos métricas con que se corre cada paso: si coinciden, la elección no
# depende de cuál se mire.  IGD+ es la «relación con el frente de Pareto».
DECIDEN = [('igd_plus', 'IGD+', False), ('hypervolume', 'Hipervolumen', True)]


def _tablas_de(d, filas, out_dir, base, titulo, texto):
    for columnas, etiq, nombre in ((MULTIOBJETIVO, 'multiobjetivo', 'MULTIOBJETIVO'),
                                   (QUIMICA, 'quimica', 'QUÍMICA')):
        t = tabla(d, filas, columnas)
        if t.empty:
            continue
        t.to_csv(os.path.join(out_dir, f'{base}_{etiq}.csv'))
        _tex(t, f'{titulo} — indicadores de {etiq}.  Media y desvío sobre las 20 '
                f'semillas.', f'tab:{base}_{etiq}',
             os.path.join(out_dir, f'{base}_{etiq}.tex'))
        texto += ['=' * 110, f'{titulo} — {nombre}', '=' * 110, t.to_string(), '']


def paso_mutacion(d, reparto, out_dir, texto):
    """Las tres configuraciones dentro de un presupuesto."""
    g = d[d.reparto == reparto]
    titulo = f'Mutación en {reparto}'
    texto += ['#' * 110, f'PASO — {titulo.upper()}', '#' * 110, '']
    ganan = []
    for col, nombre, mayor in DECIDEN:
        res = comparar(g[g.algorithm != 'CMOPSO'], CONFIGS, 'config', col, mayor,
                       ['run', 'algorithm', 'tipo', 'cruce'])
        texto += texto_test(res, nombre)
        test_por_algoritmo(g, CONFIGS, 'config', col, mayor, texto)
        if res:
            ganan.append(res['grupos'][0][0] if len(res['grupos'][0]) == 1
                         else min(res['grupos'][0], key=lambda c: res['medians'][c]))
    _tablas_de(g, ['cruce', 'tipo', 'config'], out_dir,
               f'mutacion_{reparto}', titulo, texto)
    return ganan[0] if ganan else None


def paso_presupuesto(d, config, out_dir, texto):
    """Los dos presupuestos, con la mutación ya elegida."""
    g = d[d.config == config]
    titulo = f'Presupuesto con mutación {config}'
    texto += ['#' * 110, f'PASO — {titulo.upper()}', '#' * 110, '']
    ganador = None
    for col, nombre, mayor in DECIDEN:
        repartos = sorted(g.reparto.unique())
        res = comparar(g[g.algorithm != 'CMOPSO'], repartos, 'reparto', col, mayor,
                       ['run', 'algorithm', 'tipo', 'cruce'])
        texto += texto_test(res, nombre)
        test_por_algoritmo(g, repartos, 'reparto', col, mayor, texto)
        if res and col == 'igd_plus':
            ganador = res['grupos'][0][0]
    # El detalle por algoritmo: el test agrega, pero conviene poder mostrar que
    # ningún algoritmo va en contra.
    _tablas_de(g, ['algoritmo', 'reparto'], out_dir, 'presupuesto', titulo, texto)
    return ganador


# ─── Orquestación ────────────────────────────────────────────────────────────

def etapa1(args):
    d = cargar(args.results)
    os.makedirs(args.out, exist_ok=True)
    print(f"\n{'='*70}\n  ANÁLISIS EN TRES PASOS\n"
          f"  {len(d)} corridas   {args.results} → {args.out}\n{'='*70}")
    print("  indicadores contra frente de referencia...")
    d = agregar_indicadores(d)
    texto = []

    elegidas = {}
    for reparto in sorted(d.reparto.unique(), key=lambda r: -int(r.split('x')[1])):
        print(f"  paso: mutación en {reparto}")
        elegidas[reparto] = paso_mutacion(d, reparto, args.out, texto)
        print(f"    → {elegidas[reparto]}")

    distintas = {v for v in elegidas.values() if v}
    if len(distintas) == 1:
        config = args.mutacion or distintas.pop()
        texto += [f'La misma configuración gana en los dos presupuestos: {config}', '']
        print(f"  paso: presupuesto con mutación {config}")
        ganador = paso_presupuesto(d, config, args.out, texto)
        texto += [f'RESULTADO: mutación {config}, presupuesto {ganador}', '']
        print(f"    → {ganador}")
    else:
        texto += [f'Cada presupuesto elige distinto: {elegidas}.  No se encadena '
                  f'el tercer paso.', '']
        print(f"  ⚠ los presupuestos eligen distinta mutación: {elegidas}")

    ruta = os.path.join(args.out, 'analisis.txt')
    with open(ruta, 'w') as fh:
        fh.write('\n'.join(texto))
    print(f"\n  ✓ analisis.txt  y un .csv/.tex por tabla")
