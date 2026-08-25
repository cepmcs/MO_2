"""
Post-procesamiento de los experimentos: las cuatro etapas de la comparación más
la figura de moléculas representativas.  Cada etapa lee lo que produjo la
anterior y deja sus tablas y figuras bajo plots/.

  etapa1     resultados/grid/all_metrics.csv         →  plots/hiperparametros/
  etapa2     resultados/winners/                     →  plots/operadores/<ALG>/
  etapa3     resultados/finalistas/                  →  plots/comparacion_final/
  etapa4     resultados/finalistas/ + baselines/     →  plots/baselines/
  moleculas  resultados/finalistas/                  →  plots/comparacion_final/
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import datos, figuras, tablas

# 'budget' agrupa pop_size×n_gen: están acoplados por el presupuesto fijo de
# 100.000 evaluaciones.
FACTORS_GA     = ['budget', 'crossover', 'mutation', 'cx_prob', 'mut_prob']
FACTORS_PSO    = ['budget', 'w', 'c1', 'c2']
COMBO_FACTORS  = ['crossover', 'mutation']
SUB_FACTORS_GA = ['budget', 'cx_prob', 'mut_prob']

GA_ALGS = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA']

ALG_METRIC = 'hypervolume'
ALG_METRIC_LABEL = 'hipervolumen'

MOLECULAS_OUT = os.path.join(datos.OUT_ALGORITMOS, "moleculas_representativas.png")
N_MOLECULAS = 5     # por algoritmo


def factors_for(alg):
    return FACTORS_PSO if alg == 'MOPSO' else FACTORS_GA


# ═══════════════════════════════════════════════════════════════════════════
#   Reporte común de las etapas 2 y 3
# ═══════════════════════════════════════════════════════════════════════════

def reporte(series, out_dir, contexto=None):
    """Frente de referencia, indicadores, convergencias, boxplots, tablas de
    comparación y frentes de Pareto.  Devuelve el getter de valores por run para
    que la etapa haga sus tests.

    contexto: el algoritmo, cuando las series son sus combinaciones de
    operadores (etapa 2); None cuando son los algoritmos (etapa 3)."""
    os.makedirs(out_dir, exist_ok=True)

    pf_F, pf_df = datos.build_reference_front(series)
    indicadores, curvas_ind = {}, {'igd_plus': {}, 'epsilon': {}}
    if pf_F is None:
        print("  ⚠ no se pudo construir el frente de referencia")
    else:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicadores = datos.compute_indicators_per_run(series, pf_F)
        pf_df.to_csv(os.path.join(out_dir, 'reference_front.csv'), index=False)
        if indicadores:
            pd.concat([df.assign(series=label) for label, df in indicadores.items()],
                      ignore_index=True).to_csv(
                os.path.join(out_dir, 'indicators.csv'), index=False)
        curvas_ind, largo = datos.indicator_curves(series, pf_F)
        if not largo.empty:
            largo.to_csv(os.path.join(out_dir, 'convergence_indicators.csv'),
                         index=False)
            print("  ✓ convergence_indicators.csv")

    get = datos.value_getter(series, indicadores)

    figuras.convergence_grid(
        series, out_dir,
        [('Hipervolumen', 'Convergencia de hipervolumen (↑)',
          datos.convergence_curves(series, 'hv')),
         ('IGD+ (↓)', 'Convergencia de IGD+ (↓)', curvas_ind['igd_plus']),
         ('ε+ (↓)', 'Convergencia de ε+ aditivo (↓)', curvas_ind['epsilon'])],
        'convergence_mo.png', 'Convergencia de los indicadores multiobjetivo')

    figuras.convergence_grid(
        series, out_dir,
        [('Tasa de validez', 'Convergencia de validez',
          datos.convergence_curves(series, 'validity')),
         ('Tasa de unicidad', 'Convergencia de unicidad',
          datos.convergence_curves(series, 'uniqueness')),
         ('Tasa de novedad', 'Convergencia de novedad',
          datos.convergence_curves(series, 'novelty'))]
        + [(f'Promedio de {datos.OBJECTIVE_LABELS[o]}',
            f'Convergencia de {datos.OBJECTIVE_LABELS[o]}',
            datos.objective_curves(series, o)) for o in datos.OBJETIVOS],
        'convergence_chemical.png', 'Convergencia de los indicadores químicos')

    figuras.boxplots(series, out_dir, get, figuras.BOXPLOT_MO,
                     'boxplots_mo.png',
                     'Distribución de los indicadores multiobjetivo')
    figuras.boxplots(series, out_dir, get, figuras.BOXPLOT_CHEM,
                     'boxplots_chemical.png',
                     'Distribución de los indicadores químicos')

    tablas.summary_csv(series, out_dir, get)
    tablas.comparison_tables(series, out_dir, get, contexto)

    titulo = 'Frentes de Pareto globales'
    figuras.pareto_comparison(series, out_dir,
                              f'{titulo} — {contexto}' if contexto else titulo)
    figuras.pareto_qed_sa_grid(
        series, out_dir,
        f'Frentes de Pareto QED vs SA por operador — {contexto}' if contexto
        else 'Frentes de Pareto QED vs SA por algoritmo')
    return get


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 1 — selección de hiperparámetros
#
#   De las 513 configuraciones del grid elige 17: la mejor de cada combinación
#   de operadores en los 4 GA, más la mejor global de MOPSO.  Gana la de menor
#   rango medio de hipervolumen, rankeando dentro de cada una de las 20 semillas
#   (que están pareadas: la población inicial se muestrea con random_state=run_id).
#
#   Los operadores no se testean acá: se barren dentro de cada bloque y su
#   comparación es la etapa 2, ya con la configuración de cada combo afinada.
# ═══════════════════════════════════════════════════════════════════════════

def run_matrix(g, factors, metric):
    """Matriz (configuraciones × semillas), sin las configs incompletas."""
    M = g.pivot_table(index=factors, columns='run', values=metric)
    completas = M.notna().all(axis=1)
    if not completas.all():
        print(f"  ⚠ {int((~completas).sum())} config(s) con semillas faltantes; "
              f"se omiten")
    return M[completas]


def select_config(M, higher_better):
    """Configuración de menor rango medio, rankeando dentro de cada semilla
    (rango 1 = mejor de esa semilla).  Devuelve (elegida, rangos)."""
    A = -M.values if higher_better else M.values
    R = np.apply_along_axis(stats.rankdata, 0, A)
    rangos = pd.Series(R.mean(axis=1), index=M.index)
    return rangos.idxmin(), rangos


def config_label(cfg, factors):
    """Etiqueta compacta, p. ej. '400×250 pcx/pm cx=1 mut=0.031'."""
    cfg = cfg if isinstance(cfg, tuple) else (cfg,)
    partes, cruce, mut = [], None, None
    for f, v in zip(factors, cfg):
        if f == 'budget':
            partes.append(str(v))
        elif f == 'crossover':
            cruce = v
        elif f == 'mutation':
            mut = v
        elif f == 'cx_prob':
            partes.append(f'cx={v:g}')
        elif f == 'mut_prob':
            partes.append(f'mut={v:g}')
        else:
            partes.append(f'{f}={v:g}')
    if cruce is not None:
        partes.insert(1, f'{cruce}/{mut}' if mut is not None else cruce)
    return ' '.join(partes)


def analyze_algorithm(g, alg, metric, out_dir):
    """Elige la configuración de cada combinación de operadores y mide la
    sensibilidad a cada hiperparámetro sobre el grid completo."""
    factors = [f for f in factors_for(alg) if g[f].notna().any()]
    _, higher = datos.HP_METRICS[metric]
    por_combo = set(COMBO_FACTORS).issubset(factors)
    sub_factors = SUB_FACTORS_GA if por_combo else factors

    print(f"\n{'─'*66}\n  {alg}\n{'─'*66}")

    if por_combo:
        bloques = [(f'{cx}/{mu}', gg)
                   for (cx, mu), gg in g.groupby(COMBO_FACTORS, observed=True)]
        bloques.sort(key=lambda t: datos.HP_COMBOS.index(t[0])
                     if t[0] in datos.HP_COMBOS else len(datos.HP_COMBOS))
    else:
        bloques = [(None, g)]

    elegidas = {}
    for nombre, gg in bloques:
        M = run_matrix(gg, sub_factors, metric)
        if M.shape[0] < 2:
            continue
        mejor, rangos = select_config(M, higher)
        vals = M.loc[mejor].values
        elegidas[nombre] = {'cfg': mejor,
                            'label': config_label(mejor, sub_factors),
                            'rank': float(rangos.loc[mejor]),
                            'mean': float(np.mean(vals)),
                            'std': float(np.std(vals, ddof=1)),
                            'n_configs': M.shape[0]}
        print(f"  {(nombre or 'todas'):11s} {elegidas[nombre]['label']}")
        print(f"              rango {elegidas[nombre]['rank']:.2f} de "
              f"{M.shape[0]} configs   μ {elegidas[nombre]['mean']:.5f} ± "
              f"{elegidas[nombre]['std']:.5f}")

    if not elegidas:
        print("  ⚠ sin configuraciones completas; se omite")
        return None

    # El resto de los factores se mide dentro de cada combo, no agrupando: pcx y
    # sbx están en regímenes de hipervolumen distintos y al agruparlos la mediana
    # salta entre ellos, generando un orden espurio.
    if por_combo:
        gg = g.copy()
        gg['_combo'] = gg['crossover'].astype(str) + '/' + gg['mutation'].astype(str)
        effects = {f: {c: tablas.friedman_by_factor(sub, f, metric)
                       for c, sub in gg.groupby('_combo', observed=True)}
                   for f in sub_factors}
    else:
        effects = {f: tablas.friedman_by_factor(g, f, metric) for f in sub_factors}

    def max_W(e):
        """Con combos, un factor tiene un W por combo; se resume por el mayor."""
        if isinstance(e, dict):
            vals = [w for _, w, _ in e.values() if not pd.isna(w)]
            return max(vals) if vals else np.nan
        return e[1]

    dominante = max(effects, key=lambda f: np.nan_to_num(max_W(effects[f])))
    print(f"  Hiperparámetro dominante: "
          f"{datos.FACTOR_LABELS.get(dominante, dominante)}  "
          f"(W = {max_W(effects[dominante]):.3f})")

    figuras.main_effects(g, alg, sub_factors, metric, out_dir, por_combo)
    figuras.metric_vs_validity(g, alg, metric, out_dir, elegidas, sub_factors,
                               factors, por_combo)
    tablas.write_effects_table(effects, alg, sub_factors, out_dir, metric)

    return {'blocks': elegidas, 'factors': factors, 'sub_factors': sub_factors,
            'effects': effects, 'por_combo': por_combo}


def etapa1(args):
    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}.\n"
              f"  En el cluster:  python run_experiments.py --summary-only\n"
              f"  Sobre una copia ya bajada:  python -c "
              f"\"from utils_mo import consolidate_all; "
              f"consolidate_all('{os.path.dirname(args.csv)}')\"")
        return

    df = datos.load_grid(args.csv)
    algs = args.algorithms or [a for a in datos.ALGORITHM_ORDER
                               if a in set(df['algorithm'])]
    label, higher = datos.HP_METRICS[args.metric]

    print(f"\n{'='*66}")
    print("  ETAPA 1 — SELECCIÓN DE HIPERPARÁMETROS")
    print(f"  Datos: {args.csv}  ({len(df)} ejecuciones)")
    print(f"  Métrica: {label} ({'↑' if higher else '↓'})")
    print("  Criterio: menor rango medio, rankeando dentro de cada semilla")
    print(f"  Algoritmos: {', '.join(algs)}")
    print(f"{'='*66}")

    os.makedirs(args.out, exist_ok=True)
    per_alg = {}
    for alg in algs:
        g = df[df['algorithm'] == alg].copy()
        if g.empty:
            print(f"\n  ⚠ {alg}: sin datos")
            continue
        out_dir = os.path.join(args.out, alg)
        os.makedirs(out_dir, exist_ok=True)
        r = analyze_algorithm(g, alg, args.metric, out_dir)
        if r:
            per_alg[alg] = r

    if per_alg:
        print(f"\n{'─'*66}\n  Resumen global\n{'─'*66}")
        tablas.write_selection_summary(per_alg, args.metric, args.out)

    print(f"\n{'='*66}\n  ✅ Listo: {args.out}\n{'='*66}\n")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 2 — comparación de combinaciones de operadores, por algoritmo
#
#   Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (lo que ganó su bloque
#   en la etapa 1) y compara los 4 combos entre sí, un reporte por algoritmo.
# ═══════════════════════════════════════════════════════════════════════════

def analyze_operators(alg, winners_dir, out_root, decision_col):
    series = datos.series_operadores(alg, winners_dir)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    get = reporte(series, out_dir, contexto=alg)

    # Solo se testea el indicador de decisión; el resto se reporta de forma
    # descriptiva en las tablas de comparación.
    label, higher = {c: (l, h) for c, l, h in datos.OP_INDICATORS}[decision_col]
    res = tablas.compare_indicator(get, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)
    print(f"  Friedman ({label}): p = {res['p_omnibus']:.4g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}")

    tablas.write_tests_table(res, alg, out_dir, label)
    grupos = tablas.homogeneous_groups(res, labels, res['medians'], higher)
    txt = ' > '.join('{' + ', '.join(g) + '}' for g in grupos)
    print(f"  grupos homogéneos: {txt}")

    return {'algorithm': alg, 'p_friedman': res['p_omnibus'],
            'n_pares_sig': n_sig, 'n_pares': len(res['pairs']),
            'grupos': txt, 'mejor_grupo': ', '.join(grupos[0])}


def etapa2(args):
    algs = args.algorithms or GA_ALGS
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*64}")
    print("  ETAPA 2 — COMPARACIÓN DE OPERADORES")
    print(f"  Datos: {args.winners}")
    print(f"  Decisión por: {args.metric}")
    print(f"{'='*64}")

    filas = [r for r in (analyze_operators(alg, args.winners, args.out,
                                           args.metric) for alg in algs) if r]
    if filas:
        df = pd.DataFrame(filas)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 3 — comparación estadística final entre algoritmos
#
#   Lee finalistas/<ALG>/run_XX/ (la configuración elegida de cada algoritmo
#   tras las etapas 1 y 2) y contrasta los cinco entre sí sobre ALG_METRIC.
# ═══════════════════════════════════════════════════════════════════════════

def etapa3(args):
    series = datos.series_finalistas(args.finalistas)
    if len(series) < 3:
        print(f"Se necesitan ≥3 algoritmos en {args.finalistas}")
        return
    labels = [s.label for s in series]

    print(f"\n{'='*70}")
    print("  ETAPA 3 — COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  {', '.join(datos.display(l) for l in labels)}")
    print(f"{'='*70}\n")

    get = reporte(series, args.out)

    res = tablas.compare_indicator(get, labels, ALG_METRIC)
    if res is None:
        print(f"  ⚠ sin datos de {ALG_METRIC}")
        return
    grupos = tablas.homogeneous_groups(res, labels, res['medians'], True)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)

    print(f"\n  Friedman: p = {res['p_omnibus']:.3g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}\n")
    for lab in sorted(labels, key=lambda l: -res['medians'][l]):
        print(f"    {datos.display(lab):10s} mediana = {res['medians'][lab]:.5f}")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in grupos))

    tablas.write_groups_table(res, grupos, args.out, ALG_METRIC_LABEL)
    tablas.write_pairs_csv(res, args.out)


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 4 — baselines contra los algoritmos multiobjetivo
#
#   Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de
#   suma ponderada) con los cinco MOEAs ya seleccionados, sobre el mismo
#   presupuesto de 100.000 evaluaciones y las mismas 20 semillas.
# ═══════════════════════════════════════════════════════════════════════════

def etapa4(args):
    series = datos.series_baselines(args.finalistas, args.baselines)
    labels = [s.label for s in series]
    faltan = [m for m in datos.BASELINE_KEYS if m not in labels]
    if faltan:
        print(f"  ⚠ sin datos de: {', '.join(faltan)}")
    if len(series) < 3:
        print("Se necesitan más series")
        return
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ETAPA 4 — BASELINES vs ALGORITMOS MULTIOBJETIVO")
    print(f"  {', '.join(datos.display(l) for l in labels)}")
    print(f"{'='*70}\n")

    pf_F, _ = datos.build_reference_front(series)
    indicadores = datos.compute_indicators_per_run(series, pf_F) if pf_F is not None else {}
    get = datos.value_getter(series, indicadores)

    res = tablas.compare_indicator(get, labels, args.metric)
    if res is None:
        print(f"  ⚠ sin datos de {args.metric}")
        return
    desvios = {l: float(np.std(np.asarray(get(l, args.metric), float), ddof=1))
               for l in labels}
    grupos = tablas.homogeneous_groups(res, labels, res['medians'], True)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}   "
          f"({sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)} de "
          f"{len(res['pairs'])} pares significativos)\n")
    for l in sorted(labels, key=lambda x: -res['medians'][x]):
        tipo = 'baseline' if l in datos.BASELINE_KEYS else 'MOEA'
        print(f"    {datos.display(l):14s} {res['medians'][l]:.5f}  ({tipo})")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in grupos))

    tablas.write_baseline_tables(res, grupos, res['medians'], desvios, labels,
                                 args.metric_label, args.out)
    tablas.write_pairs_csv(res, args.out)


# ═══════════════════════════════════════════════════════════════════════════
#   Moléculas representativas
# ═══════════════════════════════════════════════════════════════════════════

def pick(frente, n=N_MOLECULAS):
    """Las n moléculas de mayor QED, desempatando por menor SA."""
    f = frente.assign(_qed=frente['qed'].round(3))
    return (f.sort_values(['_qed', 'sa'], ascending=[False, True])
            .head(n).drop(columns='_qed').reset_index(drop=True))


def moleculas(args):
    frentes = {}
    for s in datos.series_finalistas(args.finalistas):
        frente = datos.frente_global(s.path)
        if not frente.empty:
            frentes[s.label] = frente
    if not frentes:
        print(f"No se encontraron frentes en {args.finalistas}")
        return

    seleccion = {alg: pick(f) for alg, f in frentes.items()}
    figuras.figura_moleculas(seleccion, args.out, N_MOLECULAS)

    filas = [{'algoritmo': datos.display(alg), 'puesto': j + 1,
              'qed': round(m['qed'], 4), 'sa': round(m['sa'], 2),
              'lipinski': m['lipinski'], 'smiles': m['smiles']}
             for alg, sel in seleccion.items() for j, m in sel.iterrows()]
    out = pd.DataFrame(filas)
    csv = os.path.splitext(args.out)[0] + '.csv'
    out.to_csv(csv, index=False)
    print(f"✓ {csv}\n")
    print(out.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        prog='python -m postproceso',
        description="Post-procesamiento de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='etapa', required=True,
                            metavar='etapa1|etapa2|etapa3|etapa4|moleculas')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p1 = sub.add_parser('etapa1', formatter_class=fmt,
                        help="Selección de hiperparámetros por combinación de "
                             "operadores.")
    p1.add_argument('--csv', default=datos.METRICS_CSV,
                    help="CSV consolidado del grid.")
    p1.add_argument('--out', default=datos.OUT_HP, help="Directorio de salida.")
    p1.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a analizar (default: todos).")
    p1.add_argument('--metric', default='hypervolume',
                    choices=list(datos.HP_METRICS), help="Métrica de selección.")
    p1.set_defaults(func=etapa1)

    p2 = sub.add_parser('etapa2', formatter_class=fmt,
                        help="Comparación de operadores por algoritmo.")
    p2.add_argument('--winners', default=datos.WINNERS_DIR)
    p2.add_argument('--out', default=datos.OUT_OPERADORES)
    p2.add_argument('--algorithms', nargs='+', default=None)
    p2.add_argument('--metric', default='hypervolume',
                    choices=[c for c, _, _ in datos.OP_INDICATORS],
                    help="Indicador con el que se elige el combo ganador.")
    p2.set_defaults(func=etapa2)

    p3 = sub.add_parser('etapa3', formatter_class=fmt,
                        help="Comparación estadística entre algoritmos.")
    p3.add_argument('--finalistas', default=datos.FINALISTAS_DIR)
    p3.add_argument('--out', default=datos.OUT_ALGORITMOS)
    p3.set_defaults(func=etapa3)

    p4 = sub.add_parser('etapa4', formatter_class=fmt,
                        help="Baselines vs algoritmos multiobjetivo.")
    p4.add_argument('--finalistas', default=datos.FINALISTAS_DIR)
    p4.add_argument('--baselines', default=datos.BASELINES_DIR)
    p4.add_argument('--out', default=datos.OUT_BASELINES)
    p4.add_argument('--metric', default='hypervolume')
    p4.add_argument('--metric-label', default='hipervolumen')
    p4.set_defaults(func=etapa4)

    pm = sub.add_parser('moleculas', formatter_class=fmt,
                        help="Moléculas representativas del frente de cada "
                             "algoritmo.")
    pm.add_argument('--finalistas', default=datos.FINALISTAS_DIR)
    pm.add_argument('--out', default=MOLECULAS_OUT)
    pm.set_defaults(func=moleculas)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
