"""
Etapa 3 — comparación entre algoritmos, frente conjunto y moléculas.

Tres cosas que se leen juntas:

  etapa3      contrasta los cinco finalistas entre sí sobre el hipervolumen.
  frente      junta las DOS familias de cruce de cada algoritmo —el material que
              llega a la fase de afinidad— y mira qué sobrevive y de dónde sale.
              Los presupuestos no son comparables: caracteriza el pool, no ordena.
  moleculas   las de mayor QED del frente de cada algoritmo, sobre ese mismo pool.
"""

import io
import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .comun import (
    ALGORITHM_ORDER,
    DISPLAY,
    FSP3_MIN,
    GA_ALGS,
    OUT_FRENTE,
    PSO_ALG,
    Series,
    _build_series_value_getter,
    _fmt_p,
    _has_runs,
    _latex_escape,
    _num,
    _write_tex,
    build_finalist_series,
    build_operator_series_winners,
    compare_indicator,
    fmt_groups,
    homogeneous_groups,
    load_metrics,
    load_pareto_molecules,
    rank_biserial,
    winner_cfg_dir,
)
from .indicadores import (
    _compute_non_dominated,
    _partir_etiqueta,
    _por_serie,
    atribuir_frente,
    build_reference_front,
    compute_indicators_per_run,
    write_contribucion_table,
)
from .figuras import plot_frente_conjunto
from .etapa2 import COMBO_DIRS, MUT_STD



# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 3 — comparación estadística final entre algoritmos
#
#   Lee finalistas/<ALG>/run_XX/ (la configuración elegida de cada algoritmo
#   tras las etapas 1 y 2) y contrasta los cinco entre sí sobre ALG_METRIC.
# ═══════════════════════════════════════════════════════════════════════════

ALG_METRIC = 'hypervolume'

ALG_METRIC_LABEL = 'hipervolumen'

ALG_HIGHER_BETTER = True



def write_groups_table(res, groups, out_dir, get_values, labels):
    """Las 10 comparaciones por pares, con el resultado resumido en grupos.

    Misma estructura que la tabla de operadores: al $p$ lo acompaña el tamaño de
    efecto, y el ganador va en su propia columna.  Donde el post-hoc no separa no
    se declara ganador, aunque las medianas ordenen."""
    vals = {l: np.asarray(get_values(l, ALG_METRIC), dtype=float) for l in labels}
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación entre algoritmos sobre {ALG_METRIC_LABEL}.  '
        f'Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm '
        f'($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de pares emparejados, con signo positivo cuando gana el '
        f'primero del par.  Grupos homogéneos, de mejor a peor: '
        f'{fmt_groups(groups)}.}}',
        r'\label{tab:comparacion_grupos}',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Par & $p$ (Holm) & $r_{rb}$ & Mejor \\', r'\midrule',
    ]
    for p in res['pairs']:
        a, b = p['a'], p['b']
        r = rank_biserial(vals[a], vals[b])
        mejor = (DISPLAY.get(a if res['medians'][a] > res['medians'][b] else b,
                             a if res['medians'][a] > res['medians'][b] else b)
                 if p['p_holm'] < 0.05 else '---')
        lines.append(
            f"{DISPLAY.get(a, a)} vs {DISPLAY.get(b, b)} & "
            f"{_fmt_p(p['p_holm'])} & "
            f"{('$+$' if r >= 0 else '$-$') + _num(abs(r), 3)} & {mejor} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    path = os.path.join(out_dir, 'grupos_homogeneos.tex')
    print()
    _write_tex(lines, path, msg=path)



def etapa3(args):
    algs = [a for a in ALGORITHM_ORDER
            if _has_runs(os.path.join(args.finalistas, a))]
    series = build_finalist_series(algs, args.finalistas)
    if len(series) < 3:
        print(f"Se necesitan ≥3 algoritmos en {args.finalistas}")
        return
    labels = [s.label for s in series]
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ETAPA 3 — COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, pf_df = build_reference_front(series)
    ind = compute_indicators_per_run(series, pf) if pf is not None else {}
    print(f"  frente de referencia: {len(pf) if pf is not None else 0} soluciones\n")
    get = _build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, ALG_METRIC)
    if res is None:
        print(f"  ⚠ sin datos de {ALG_METRIC}")
        return
    groups = homogeneous_groups(res, labels, res['medians'], ALG_HIGHER_BETTER)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}\n")
    for lab in sorted(labels, key=lambda l: -res['medians'][l]):
        print(f"    {DISPLAY.get(lab, lab):10s} mediana = {res['medians'][lab]:.5f}")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_groups_table(res, groups, args.out, get, labels)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print(f"  ✓ {os.path.join(args.out, 'tests_pares.csv')}")

    analisis_frente_conjunto(args)



# ═══════════════════════════════════════════════════════════════════════════
#   Frente conjunto — el pool de candidatos
#
#   Otra pregunta que la etapa 3: acá se juntan las DOS familias de cruce de cada
#   algoritmo —el material que llega a la fase de afinidad— y se mira qué
#   sobrevive y de dónde sale.  Los presupuestos no son comparables (cada AG
#   aporta dos configuraciones y CMOPSO una): caracteriza el pool, no ordena.
# ═══════════════════════════════════════════════════════════════════════════

# Cada familia de cruce aporta una rama al pool.
POOL_FAMILIAS = ['pcx', 'sbx']



def combos_pool(alg, winners_dir, alpha=0.05):
    """Los dos combos con que un AG entra al pool: uno por familia de cruce y,
    dentro de cada familia, la mutación que gana.

    Misma regla que el campeón de la etapa 2 —la ganadora si el post-hoc la
    separa, la estándar si no— pero por familia, y con el mismo $p$ corregido que
    publica esa tabla.  Decide el hipervolumen.

    En la práctica queda pm en todas las ramas salvo la SBX de MOEA/D y AGE-MOEA,
    donde el test sí separa (p = 0.038 y 0.021) y gana la gaussiana.

    Devuelve [(familia, combo)] en el orden de POOL_FAMILIAS."""
    series = build_operator_series_winners(alg, winners_dir, COMBO_DIRS)
    if not series:
        return []          # CMOPSO: sin operadores no hay ramas que elegir
    hv = {s.label: load_metrics(s.pop_dir).sort_values('run')['hypervolume'].values
          for s in series}
    res = compare_indicator(lambda lab, _col: hv[lab], list(hv), None)

    elegidos = []
    for fam in POOL_FAMILIAS:
        pm, gauss = f'{fam}_{MUT_STD}', f'{fam}_gauss'
        if pm not in hv:
            continue
        if gauss not in hv or res is None:
            elegidos.append((fam, pm))
            continue
        p = next((x['p_holm'] for x in res['pairs']
                  if {x['a'], x['b']} == {pm, gauss}), 1.0)
        gana_gauss = p < alpha and res['medians'][gauss] > res['medians'][pm]
        elegidos.append((fam, gauss if gana_gauss else pm))
    return elegidos



def _series_pool(winners_dir, finalistas_dir):
    """Las dos ramas de cruce de cada AG, más CMOPSO, que no tiene operadores."""
    series = []
    for alg in GA_ALGS:
        # El combo va en la etiqueta y no solo la familia: desde que la mutación
        # puede cambiar entre ramas, decir 'SBX' a secas escondería cuál es.
        for _, combo in combos_pool(alg, winners_dir):
            cfg_dir = winner_cfg_dir(winners_dir, alg, combo)
            if cfg_dir:
                series.append(Series(f'{DISPLAY.get(alg, alg)} ({combo})',
                                        cfg_dir))
    d = os.path.join(finalistas_dir, PSO_ALG)
    if _has_runs(d):
        series.append(Series(PSO_ALG, d))
    return series



def _nota_pool(winners_dir):
    """Frase para el caption de la tabla del pool: con qué configuración entró
    cada algoritmo.

    Las excepciones se arman con la misma regla que elige las ramas, así que la
    frase no puede decir una cosa mientras el pool hace otra.  El detalle
    estadístico vive en las tablas de la etapa 2 y se remite por \\ref."""
    excepciones = []
    for alg in GA_ALGS:
        for fam, combo in combos_pool(alg, winners_dir):
            mut = combo.split('_')[1]
            if mut != MUT_STD:
                excepciones.append(
                    f'{_latex_escape(DISPLAY.get(alg, alg))} en {fam.upper()} '
                    f'({"gaussiana" if mut == "gauss" else _latex_escape(mut)}, '
                    f'cuadro~\\ref{{tab:ops_{alg.lower()}}})')

    nota = ('  Cada algoritmo genético entra con dos ramas, una por familia de '
            'cruce, en la configuración que ganó su bloque en la selección de '
            'hiperparámetros.  La mutación es la polinomial')
    if not excepciones:
        return nota + ' en todas las ramas.'
    detalle = (' y '.join(excepciones) if len(excepciones) < 3
               else ', '.join(excepciones[:-1]) + ' y ' + excepciones[-1])
    return (nota + ', salvo en las ramas donde la comparación de operadores '
            f'separó a las dos mutaciones: {detalle}.')



def _algoritmo_pool(label):
    """Agrupa por algoritmo: las dos ramas de cruce de un AG caen en el mismo
    grupo ('NSGA-II (PCX)' → 'NSGA-II'); CMOPSO queda solo.

    Es la agrupación de las figuras del frente conjunto: lo que interesa es qué
    algoritmo puso cada molécula, no de qué operador salió cada región."""
    return _partir_etiqueta(label)[0]



def analisis_frente_conjunto(args):
    series = _series_pool(args.winners, args.finalistas)
    if len(series) < 2:
        print(f"\n  ⚠ sin datos suficientes en {args.winners}; se omite el "
              f"frente conjunto")
        return

    print(f"\n{'='*70}")
    print("  FRENTE CONJUNTO — pool de candidatos")
    print(f"  {len(series)} configuraciones: {', '.join(s.label for s in series)}")
    print(f"{'='*70}\n")

    pf_F, pf_df = build_reference_front(series)
    if pf_df is None:
        print("  ⚠ no se pudo construir el frente conjunto")
        return
    os.makedirs(args.out_frente, exist_ok=True)

    # El frente en sí, con la atribución: son las moléculas candidatas que pasan
    # a la fase de afinidad, así que conviene tenerlas y no solo su resumen.
    at = atribuir_frente(series, pf_df, _por_serie)
    at.to_csv(os.path.join(args.out_frente, 'frente_pool.csv'), index=False)
    print(f"  ✓ frente_pool.csv  ({len(at)} moléculas)")

    # La tabla desglosa por configuración; la figura agrupa por algoritmo, porque
    # nueve colores serían ilegibles.  Ya no hay versión 3D: con Fsp3 como
    # constraint el frente es una curva, y su lugar lo ocupa el segundo panel de
    # plot_frente_conjunto.
    write_contribucion_table(
        series, pf_df, 'pool', args.out_frente,
        grupo_de=_por_serie, etiqueta='Configuración',
        nota=_nota_pool(args.winners))
    plot_frente_conjunto(series, 'pool', args.out_frente, pf_df,
                            grupo_de=_algoritmo_pool)



# ═══════════════════════════════════════════════════════════════════════════
#   Moléculas representativas
#
#   Las moléculas de mayor QED del frente de cada algoritmo.  El frente junta las
#   20 ejecuciones de sus DOS ramas de cruce (las del pool), deduplica por SMILES
#   y recalcula la dominancia.  Va con el frente conjunto porque ilustra el
#   material que llega a la fase de afinidad, no lo que comparó la etapa 3.
# ═══════════════════════════════════════════════════════════════════════════

MOLECULAS_OUT = os.path.join(OUT_FRENTE, "moleculas_representativas.png")


N_MOLECULAS = 5      # por algoritmo


# Ventana de interés farmacológico.  Ya no lleva banda de Fsp3: el constraint la
# garantiza.  Queda el corte de SA, que desempata entre las decenas de moléculas
# empatadas en el QED máximo.
SA_MAX = 3.0



def load_front(alg, winners_dir, finalistas_dir):
    """Frente no dominado de un algoritmo sobre las configuraciones del pool.

    Para los AG son sus dos ramas de cruce juntas; CMOPSO no tiene operadores y
    va con su única configuración."""
    dfs = []
    for _, combo in combos_pool(alg, winners_dir):
        cfg_dir = winner_cfg_dir(winners_dir, alg, combo)
        if cfg_dir:
            dfs.append(load_pareto_molecules(cfg_dir))
    if not dfs:
        d = os.path.join(finalistas_dir, alg)
        if _has_runs(d):
            dfs.append(load_pareto_molecules(d))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return _compute_non_dominated(df.drop_duplicates(subset='smiles'))



def pick(front, n=N_MOLECULAS):
    """Las n moléculas de mayor QED con SA por debajo de SA_MAX.

    Ordenar solo por QED no alcanza: hay decenas empatadas en QED ≈ 0.948, así
    que manda el desempate por SA.  Si el corte deja el frente vacío se cae al
    frente completo.
    """
    dentro = front[front['sa'] < SA_MAX]
    if dentro.empty:
        dentro = front
    return dentro.nlargest(n, 'qed').reset_index(drop=True)



def render(smiles, size=(420, 320)):
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



def moleculas(args):
    algs = [a for a in ALGORITHM_ORDER
            if _has_runs(os.path.join(args.finalistas, a))]
    fronts = {a: load_front(a, args.winners, args.finalistas) for a in algs}
    fronts = {a: f for a, f in fronts.items() if not f.empty}
    if not fronts:
        print(f"No se encontraron frentes en {args.finalistas}")
        return

    seleccion = {a: pick(f) for a, f in fronts.items()}

    nrows, ncols = len(fronts), N_MOLECULAS
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for i, (alg, sel) in enumerate(seleccion.items()):
        for j in range(ncols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
            if j >= len(sel):
                ax.set_visible(False)
                continue

            m = sel.iloc[j]
            img = render(m['smiles'])
            if img is not None:
                ax.imshow(img)
            # Los dos objetivos más Fsp3: no seleccionó nada, pero deja ver con
            # cuánto margen sobre el umbral quedó cada estructura dibujada.
            ax.set_xlabel(f"QED {m['qed']:.3f}  ·  SA {m['sa']:.2f}  "
                          f"·  Fsp3 {m['fsp3']:.2f}",
                          fontsize=9, labelpad=3)
            if j == 0:
                ax.set_ylabel(DISPLAY.get(alg, alg), fontsize=13,
                              fontweight='bold', labelpad=10)

    fig.suptitle(f'Las {N_MOLECULAS} moléculas de mayor QED del frente de cada '
                 f'algoritmo, con SA < {SA_MAX:g} '
                 f'(todas cumplen Fsp3 ≥ {FSP3_MIN:g})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {args.out}")

    # Detalle de las moléculas elegidas
    rows = []
    for alg, sel in seleccion.items():
        for j, m in sel.iterrows():
            rows.append({'algoritmo': DISPLAY.get(alg, alg), 'puesto': j + 1,
                         'qed': round(m['qed'], 4), 'sa': round(m['sa'], 2),
                         'fsp3': m['fsp3'], 'smiles': m['smiles']})
    out = pd.DataFrame(rows)
    csv = os.path.splitext(args.out)[0] + '.csv'
    out.to_csv(csv, index=False)
    print(f"✓ {csv}\n")
    print(out.to_string(index=False))
