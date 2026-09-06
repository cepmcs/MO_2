"""
Indicadores de calidad del frente y contribución al frente conjunto.

Frente de referencia combinado, IGD+, ε+ y no-dominancia.  Y la atribución: de
las moléculas que sobreviven al juntar todas las series, quién puso cada una.
"""

import glob
import os

import numpy as np
import pandas as pd
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy import stats

from .comun import (
    COMPARTIDA_COLOR,
    CRUCE_COLORS,
    DISPLAY,
    FSP3_MIN,
    OBJECTIVES,
    _fmt_p,
    _latex_escape,
    _num,
    _write_tex,
    compare_indicator,
    fmt_groups,
    get_color,
    homogeneous_groups,
    load_pareto_molecules,
)





def _df_to_F(df):
    """Convierte DataFrame con qed y sa a matriz F de minimización [-QED, SA].

    Fsp3 no entra: es constraint, no objetivo.  Meterlo como tercera columna
    cambiaría quién domina a quién: una molécula peor en los dos objetivos
    sobreviviría por tener más Fsp3."""
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
    """Additive Epsilon (manual: pymoo 0.6 no lo trae).

    ε+ = max_j  min_i  max_k  (F_i_k - PF_j_k)

    Mínimo desplazamiento uniforme para que F domine a todo el frente de
    referencia; menor es mejor.  F y pf deben venir normalizados con los mismos
    bounds (ver _front_bounds).
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
    """Frente de referencia combinando todas las runs de todas las series →
    (pf_F, pf_df).  El procedimiento estándar cuando el frente verdadero se
    desconoce: juntar todo, deduplicar y recalcular la no-dominancia global."""
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

    No sirve mirar build_reference_front: deduplica quedándose con la primera
    aparición, o sea con el orden en que se concatenaron las series.  Hay que
    releer cada una, porque una misma molécula puede haberla hallado varias.

    grupo_de decide sobre qué se agrega: por familia de cruce al comparar
    operadores, por serie al comparar algoritmos.

    Agrega 'en_<label>' por serie, otra por grupo, y 'origen' con el grupo que la
    halló en exclusiva o la marca de compartida.
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
    """Cuánto aporta cada serie al frente conjunto, sobre la unión de las 20
    semillas.  'aporta' incluye las compartidas (así que las columnas no suman el
    total); 'exclusiva' solo las que no encontró ninguna otra."""
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
    """Lo mismo pero dentro de cada semilla: solo compiten los frentes de esa
    semilla.  Da bloques para un test de Friedman o de rangos con signo.  El
    agregado mide otra cosa (todo contra todo), así que los dos porcentajes no
    tienen por qué coincidir.

    Devuelve dict grupo → (% exclusivos por semilla, % compartidas, semillas).
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
    """Prepara el frente conjunto para dibujarlo: lo atribuye y arma la paleta.

    Va aparte de la figura porque la atribución es lo caro (relee molecules.csv de
    todas las series) y porque la paleta la comparte con la tabla de contribución.

    Devuelve None si la atribución no aplica (una sola serie)."""
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

    return {'at': at, 'paleta': paleta, 'compartida': compartida,
            'por': 'familia de cruce' if por_cruce else 'algoritmo'}





def _indicator_curves(series, pop_size, output_dir, pf_F, gen_stride=10):
    """Curvas de IGD+ y ε+ por generación, SIN re-entrenar.

    Mide el frente ACUMULADO hasta cada generación, no el de esa generación
    sola.  El instantáneo medía otra cosa y engañaba: son 4-9 moléculas contra
    las ~35 del frente de referencia, y al converger la población se apiña y deja
    de cubrirlo, así que la curva SUBÍA.  Peor, su último punto no coincidía con
    el IGD+ de la tabla —que sale de molecules.csv, o sea del frente acumulado— y
    ordenaba los algoritmos al revés: NSGA-II es el mejor de la tabla (0.018) y
    salía último en la curva (0.101).  Con el acumulado la curva baja y su último
    punto es exactamente el valor de la tabla.

    Reconstruye desde all_molecules.csv.gz y promedia sobre las runs.

    Solo compiten las FACTIBLES, igual que utils_mo.build_pareto.  El log crudo
    trae infactibles, y dejarlas entrar compararía un frente por generación que
    viola el umbral contra un frente de referencia que no.

    gen_stride submuestrea generaciones (1 = todas).

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

            # El acumulado se construye incremental —no_dom(acumulado previo +
            # evaluaciones de la generación)— y no recalculando sobre todo lo
            # visto: con las ~100k evaluaciones de una corrida el
            # NonDominatedSorting de pymoo arma una matriz n×n y se va a OOM (el
            # mismo motivo por el que utils_mo tiene su propio filtro de Kung).
            # Así nunca hay más de unos cientos de filas en juego.
            ultima = df['gen'].max()
            rows, acum = [], None
            for k, (g, df_g) in enumerate(df.groupby('gen', sort=True)):
                nuevo = (df_g[OBJECTIVES] if acum is None else
                         pd.concat([acum, df_g[OBJECTIVES]], ignore_index=True))
                acum = _compute_non_dominated(nuevo.drop_duplicates())
                # La última generación va siempre: es la que tiene que cerrar con
                # el valor que publica la tabla de indicadores por run.
                if k % gen_stride and g != ultima:
                    continue
                if acum.empty:
                    continue
                F_g = _normalize_F(_df_to_F(acum), ideal, scale)
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



def _test_aporte(por_grupo, runs):
    """Contraste sobre el % aportado por semilla.  Con dos grupos alcanza el
    Wilcoxon pareado; con más hace falta el mismo Friedman + Holm que el resto
    de la comparación, resumido en grupos homogéneos.

    Devuelve (texto para el caption, dict de resumen) o (None, {})."""
    grupos = list(por_grupo)
    if len(grupos) < 2 or len(runs) < 3:
        return None, {}

    if len(grupos) == 2:
        a, b = grupos
        try:
            p = float(stats.wilcoxon(por_grupo[a], por_grupo[b]).pvalue)
        except ValueError:
            p = 1.0
        g = a if por_grupo[a].mean() > por_grupo[b].mean() else b
        otro = b if g == a else a
        txt = (f'  Repitiendo el cálculo dentro de cada semilla, '
               f'{_latex_escape(g)} aporta {_num(por_grupo[g].mean(), 1)}\\% $\\pm$ '
               f'{_num(por_grupo[g].std(ddof=1), 1)} frente a '
               f'{_num(por_grupo[otro].mean(), 1)}\\% $\\pm$ '
               f'{_num(por_grupo[otro].std(ddof=1), 1)}, en '
               f'{sum(por_grupo[g] > por_grupo[otro])} de las {len(runs)} '
               f'semillas (Wilcoxon de rangos con signo, $p$ = {_fmt_p(p)}).')
        return txt, {'aporte_grupo': g,
                     'aporte_pct': round(float(por_grupo[g].mean()), 2),
                     'aporte_p': p}

    res = compare_indicator(lambda l, c: por_grupo[l], grupos, None)
    if res is None:
        return None, {}
    gr = homogeneous_groups(res, grupos, res['medians'], True)
    txt = (f'  Repitiendo el cálculo dentro de cada semilla, el aporte se '
           f'contrasta con Friedman sobre las {len(runs)} semillas como '
           f'bloques ($p$ = {_fmt_p(res["p_omnibus"])}) y comparaciones por '
           f'pares con Wilcoxon corregidas por Holm.  Grupos homogéneos, de '
           f'mayor a menor aporte: {fmt_groups(gr)}.')
    return txt, {'aporte_grupo': ', '.join(gr[0]),
                 'aporte_pct': round(float(np.mean(por_grupo[gr[0][0]])), 2),
                 'aporte_p': res['p_omnibus']}



def _partir_etiqueta(nombre):
    """'NSGA-II (PCX)' → ('NSGA-II', 'PCX').  Sin paréntesis, el segundo campo
    queda vacío: es el caso de CMOPSO, que no tiene operadores."""
    if nombre.endswith(')') and '(' in nombre:
        alg, cruce = nombre.rsplit('(', 1)
        return alg.strip(), cruce[:-1].strip()
    return nombre, '---'



def write_contribucion_table(series, pf_df, nombre, out_dir,
                             grupo_de=None, etiqueta='Operador', nota=''):
    """Tabla LaTeX de la contribución al frente no dominado conjunto.

    Complementa al test sobre el hipervolumen, que mide la extensión del frente y
    no la calidad de lo que contiene: se junta todo, se recalcula la no-dominancia
    global y se mira quién aportó los supervivientes.  Fsp3 no participa de la
    dominancia pero se reporta, para ver cuánto margen le deja al umbral.

    Las etiquetas 'NSGA-II (PCX)' se parten en dos columnas, con el algoritmo en
    \\multirow: son dos ramas de la misma entidad, no dos entidades.

    Devuelve el resumen del contraste para el CSV de la etapa.
    """
    grupo_de = grupo_de or _familia
    filas, _ = contribucion_agregada(series, pf_df, grupo_de)
    por_grupo, compartidas, runs = contribucion_por_semilla(series, grupo_de)
    _, resumen = _test_aporte(por_grupo, runs)

    partidas = [_partir_etiqueta(DISPLAY.get(f['nombre'], f['nombre']))
                for f in filas]
    total = filas[0]['total'] if filas else 0
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Contribución al frente no dominado conjunto en '
        f'{_latex_escape(nombre)}.  Se unen las soluciones de las series '
        f'comparadas sobre las {len(runs)} semillas, se deduplica por SMILES y '
        f'se recalcula la no-dominancia global ({total} soluciones).  «Aporta» '
        f'cuenta toda molécula del frente hallada por esa serie, por lo que las '
        f'compartidas suman en cada fila que las encontró; «exclusivas» solo '
        f'las que no halló ninguna otra.  Las tres últimas columnas son la media '
        f'sobre lo que cada serie aporta, y describen no cuánto sino qué aporta: '
        f'QED y SA son los objetivos, y Fsp3 va sin flecha porque es la '
        f'restricción ($\\geq$ {_num(FSP3_MIN, 2)}) y no algo que se '
        f'optimice.{nota}}}',
        f'\\label{{tab:contribucion_{nombre.lower()}}}',
        r'\begin{tabular}{llrrrrrr}', r'\toprule',
        f'{etiqueta} & Operadores & Aporta & Exclusivas & \\% & QED $\\uparrow$ & '
        f'SA $\\downarrow$ & Fsp3 \\\\',
        r'\midrule',
    ]
    for i, (f, (alg, cruce)) in enumerate(zip(filas, partidas)):
        # Las filas de grupo, si las hay, van separadas: agregan sobre las
        # anteriores y no son sumables con ellas.
        if i == len(series) and len(filas) > len(series):
            lines.append(r'\midrule')
        # El nombre del algoritmo se escribe una sola vez, abarcando sus ramas.
        n_ramas = sum(1 for a, _ in partidas if a == alg)
        primera = i == 0 or partidas[i - 1][0] != alg
        if primera:
            if i:
                lines.append(r'\midrule')
            celda = (r'\multirow{%d}{*}{%s}' % (n_ramas, _latex_escape(alg))
                     if n_ramas > 1 else _latex_escape(alg))
        else:
            celda = ''
        lines.append(
            f"{celda} & {_latex_escape(cruce)} & "
            f"{f['aporta']} & {f['exclusiva']} & {_num(100*f['frac'], 1)} & "
            f"{_num(f['qed'], 3)} & {_num(f['sa'], 2)} & {_num(f['fsp3'], 3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'contribucion_{nombre}.tex'))

    pd.DataFrame(filas).to_csv(
        os.path.join(out_dir, f'contribucion_{nombre}.csv'), index=False)

    resumen_filas = filas[len(series):] or filas
    for f in resumen_filas:
        print(f"  aporte {f['nombre']:>8s}: {f['aporta']:4d}/{total} "
              f"({100*f['frac']:4.1f}%)  excl. {f['exclusiva']:4d}  "
              f"QED<0.60 {100*f['qed_bajo']:4.1f}%  "
              f"Fsp3 en el borde {100*f['fsp3_borde']:4.1f}%")
    if por_grupo:
        detalle_txt = '   '.join(
            f'{g} {v.mean():.1f}%±{v.std(ddof=1):.1f}' for g, v in por_grupo.items())
        print(f"  por semilla: {detalle_txt}   compartidas {compartidas.mean():.1f}%")
    return resumen
