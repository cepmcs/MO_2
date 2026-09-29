"""
Comparación entre algoritmos, frente conjunto y moléculas.

Tres cosas que se leen juntas:

  algoritmos  contrasta los tres algoritmos entre sí sobre el hipervolumen, una vez
              por cruce (PCX y SBX).
  frente      junta las seis configuraciones —tres algoritmos por dos cruces— y mira
              qué sobrevive y de dónde sale.  Caracteriza el pool, no ordena.
  moleculas   las de mayor QED del frente de cada algoritmo, con sus dos cruces
              juntos.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .comun import (
    ALGORITHM_ORDER,
    CRUCES,
    DISPLAY,
    FSP3_MIN,
    OUT_FRENTE,
    _build_series_value_getter,
    _fmt_p,
    _num,
    _num_es,
    _write_tex,
    cfg_dir,
    compare_indicator,
    fmt_groups,
    homogeneous_groups,
    load_metrics,
    load_pareto_molecules,
    rank_biserial,
    series_algoritmos,
    series_pool,
)
from .indicadores import (
    _compute_non_dominated,
    _partir_etiqueta,
    _por_serie,
    atribuir_frente,
    build_reference_front,
    write_contribucion_table,
)
from .figuras import SA_MAX, plot_frente_conjunto, render, top_por_qed



# ═══════════════════════════════════════════════════════════════════════════
#   Comparación estadística entre algoritmos
#
#   Lee <results>/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ y contrasta los tres
#   algoritmos entre sí sobre ALG_METRIC, una vez por cruce.
# ═══════════════════════════════════════════════════════════════════════════

ALG_METRIC = 'hypervolume'

ALG_METRIC_LABEL = 'hipervolumen'

ALG_HIGHER_BETTER = True



def write_groups_table(res, groups, out_dir, get_values, labels, cruce):
    """Las comparaciones por pares, con el resultado resumido en grupos.

    Al $p$ lo acompaña el tamaño de efecto, y el ganador va en su propia columna.
    Donde el post-hoc no separa no se declara ganador, aunque las medianas ordenen."""
    vals = {l: np.asarray(get_values(l, ALG_METRIC), dtype=float) for l in labels}
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación entre algoritmos con cruce {cruce.upper()} sobre '
        f'{ALG_METRIC_LABEL}.  Test de Friedman con las '
        f'{min(len(v) for v in vals.values())} semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm '
        f'($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de pares emparejados, con signo positivo cuando gana el '
        f'primero del par.  Grupos homogéneos, de mejor a peor: '
        f'{fmt_groups(groups)}.}}',
        f'\\label{{tab:comparacion_grupos_{cruce}}}',
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



def comparar_algoritmos(cruce, results, out_root):
    series = series_algoritmos(results, cruce)
    if len(series) < 3:
        print(f"\n  ⚠ cruce {cruce.upper()}: {len(series)} algoritmo(s) con datos "
              f"en {results}; se necesitan ≥3")
        return
    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, cruce)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"  Cruce {cruce.upper()}: {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'─'*70}\n")

    get = _build_series_value_getter(series)
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

    write_groups_table(res, groups, out_dir, get, labels, cruce)
    csv = os.path.join(out_dir, 'tests_pares.csv')
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(csv, index=False)
    print(f"  ✓ {csv}")



def algoritmos(args):
    print(f"\n{'='*70}")
    print("  COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  Datos: {args.results}")
    print(f"{'='*70}")

    for cruce in args.crossovers or CRUCES:
        comparar_algoritmos(cruce, args.results, args.out)

    analisis_frente_conjunto(args)



# ═══════════════════════════════════════════════════════════════════════════
#   Frente conjunto — el pool de candidatos
#
#   Otra pregunta que la comparación de algoritmos: acá se juntan las dos familias
#   de cruce de cada algoritmo y se mira qué sobrevive y de dónde sale.  No ordena
#   algoritmos, caracteriza el pool.
# ═══════════════════════════════════════════════════════════════════════════

def _nota_pool(series):
    """Frase para el caption de la tabla del pool: con qué operadores entró cada
    algoritmo, leída de los metrics.csv de las series."""
    m = pd.concat([load_metrics(s.pop_dir) for s in series], ignore_index=True)
    cx, mut = sorted(set(m['cx_prob'])), sorted(set(m['mut_prob']))
    if len(cx) > 1 or len(mut) > 1:
        return ''
    return (f'  Cada algoritmo entra con dos ramas, una por familia de cruce, con '
            f'probabilidad de cruce {_num_es(cx[0], "g")} y mutación polinomial de '
            f'probabilidad {_num_es(mut[0], "g")} por gen.')



def _algoritmo_pool(label):
    """Agrupa por algoritmo: las dos ramas de cruce de un algoritmo caen en el
    mismo grupo ('NSGA-II (PCX)' → 'NSGA-II').

    Es la agrupación de las figuras del frente conjunto: lo que interesa es qué
    algoritmo puso cada molécula, no de qué operador salió cada región."""
    return _partir_etiqueta(label)[0]



def analisis_frente_conjunto(args):
    series = series_pool(args.results)
    if len(series) < 2:
        print(f"\n  ⚠ sin datos suficientes en {args.results}; se omite el "
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

    at = atribuir_frente(series, pf_df, _por_serie)
    at.to_csv(os.path.join(args.out_frente, 'frente_pool.csv'), index=False)
    print(f"  ✓ frente_pool.csv  ({len(at)} moléculas)")

    write_contribucion_table(
        series, pf_df, 'pool', args.out_frente,
        grupo_de=_por_serie, etiqueta='Configuración',
        nota=_nota_pool(series))
    plot_frente_conjunto(series, 'pool', args.out_frente, pf_df,
                            grupo_de=_algoritmo_pool)



# ═══════════════════════════════════════════════════════════════════════════
#   Moléculas representativas
#
#   Las moléculas de mayor QED del frente de cada algoritmo.  El frente junta las
#   ejecuciones de sus dos ramas de cruce, deduplica por SMILES y recalcula la
#   dominancia.
# ═══════════════════════════════════════════════════════════════════════════

MOLECULAS_OUT = os.path.join(OUT_FRENTE, "moleculas_representativas.png")


N_MOLECULAS = 5      # por algoritmo



def load_front(alg, results):
    """Frente no dominado de un algoritmo sobre sus dos cruces juntos."""
    dfs = []
    for cruce in CRUCES:
        d = cfg_dir(results, alg, cruce)
        if d:
            dfs.append(load_pareto_molecules(d))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return _compute_non_dominated(df.drop_duplicates(subset='smiles'))



def pick(front, n=N_MOLECULAS):
    """Las n moléculas de mayor QED con SA por debajo de SA_MAX, desempatando por
    SA entre las que comparten QED.  Mismo criterio que la figura de moléculas
    por operador: ver figuras.top_por_qed."""
    return top_por_qed(front, n).reset_index(drop=True)



def moleculas(args):
    fronts = {a: load_front(a, args.results) for a in ALGORITHM_ORDER}
    fronts = {a: f for a, f in fronts.items() if not f.empty}
    if not fronts:
        print(f"No se encontraron frentes en {args.results}")
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
