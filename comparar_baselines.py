"""
Etapa 4: baselines contra los algoritmos multiobjetivo.

Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de
suma ponderada) con los cinco
MOEAs ya seleccionados, sobre el mismo presupuesto de 100.000 evaluaciones y las
mismas 20 semillas.

Mismo protocolo que las etapas 2 y 3: Friedman con la semilla como bloque y, si
resulta significativo, comparaciones por pares con Wilcoxon de rangos con signo
corregidas por Holm.  El resultado se resume en grupos homogéneos.

Salida (plots/baselines/)
  grupos_homogeneos.tex     los grupos y las comparaciones contra cada baseline
  comparacion.tex           mediana ± desvío de cada método
  tests_pares.csv           todas las comparaciones con su p de Holm

Uso:
    python comparar_baselines.py
    python comparar_baselines.py --metric igd_plus
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

import plot_comparison as pc
from comparar_operadores import compare_indicator, homogeneous_groups, _fmt_p

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALISTAS = os.path.join(ROOT_DIR, "finalistas")
BASELINES = os.path.join(ROOT_DIR, "results_baselines")
OUT_DIR = os.path.join(ROOT_DIR, "plots", "baselines")

DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO',
           'RANDOM': 'Aleatorio', 'LHS': 'LHS', 'WEIGHTED_GA': 'GA ponderado',
           'SCREENING': 'Cribado MOSES', 'HILL_CLIMBER': 'Escalador'}
# Orden de peor a mejor esperado; LHS quedó fuera por ser indistinguible de RANDOM.
BASELINE_KEYS = ['WEIGHTED_GA', 'HILL_CLIMBER', 'RANDOM', 'SCREENING']


def build_series(finalistas, baselines):
    """Los cinco MOEAs y las tres baselines como series comparables."""
    series = []
    for alg in pc.ALGORITHM_ORDER:
        d = os.path.join(finalistas, alg)
        if pc._has_runs(d):
            series.append(pc.Series(alg, d, color_key=alg))
    for m in BASELINE_KEYS:
        # results_baselines/<METHOD>/[tag/]pop{P}_gen{G}/
        for d in sorted(glob.glob(os.path.join(baselines, m, '*', '*')) +
                        glob.glob(os.path.join(baselines, m, '*'))):
            if pc._has_runs(d):
                series.append(pc.Series(m, d, color_key=m))
                break
    return series


def write_tables(res, groups, medians, stds, labels, metric_label, out_dir):
    n_moea = sum(1 for l in labels if l not in BASELINE_KEYS)

    # Tabla 1: descriptiva
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Algoritmos multiobjetivo frente a las baselines sobre '
        f'{metric_label}, con idéntico presupuesto de 100.000 evaluaciones y las '
        f'mismas 20 semillas.  Mediana y desvío estándar entre ejecuciones.}}',
        r'\label{tab:baselines_desc}',
        r'\begin{tabular}{llc}', r'\toprule',
        r'& Método & ' + metric_label.capitalize() + r' \\', r'\midrule',
    ]
    for i, lab in enumerate(labels):
        if i == n_moea:
            lines.append(r'\midrule')
        tipo = ''
        if i == 0:
            tipo = r'\multirow{%d}{*}{MOEA}' % n_moea
        elif i == n_moea:
            tipo = r'\multirow{%d}{*}{Baseline}' % (len(labels) - n_moea)
        lines.append(f"{tipo} & {DISPLAY.get(lab, lab)} & "
                     f"${medians[lab]:.4f} \\pm {stds[lab]:.4f}$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(out_dir, 'comparacion.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print("  ✓ comparacion.tex")

    # Tabla 2: cada MOEA contra cada baseline
    moeas = [l for l in labels if l not in BASELINE_KEYS]
    bases = [l for l in labels if l in BASELINE_KEYS]
    pmap = {}
    for p in res['pairs']:
        pmap[(p['a'], p['b'])] = p['p_holm']
        pmap[(p['b'], p['a'])] = p['p_holm']

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparaciones por pares entre cada algoritmo multiobjetivo y '
        f'cada baseline sobre {metric_label}.  Test de Friedman global '
        f'($p$ = {_fmt_p(res["p_omnibus"])}) y Wilcoxon de rangos con signo por '
        f'pares con corrección de Holm; se indica el $p$ ajustado.  '
        f'Grupos homogéneos: {fmt_groups(groups)}.}}',
        r'\label{tab:baselines_tests}',
        r'\begin{tabular}{l' + 'c' * len(bases) + '}', r'\toprule',
        'Algoritmo & ' + ' & '.join(DISPLAY.get(b, b) for b in bases) + r' \\',
        r'\midrule',
    ]
    for m in moeas:
        cells = [_fmt_p(pmap.get((m, b))) for b in bases]
        lines.append(f'{DISPLAY.get(m, m)} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(out_dir, 'grupos_homogeneos.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print("  ✓ grupos_homogeneos.tex")


def fmt_groups(groups):
    return ' $>$ '.join('\\{' + ', '.join(DISPLAY.get(x, x) for x in g) + '\\}'
                        for g in groups)


def main():
    ap = argparse.ArgumentParser(description="Etapa 4: baselines vs MOEAs.")
    ap.add_argument('--finalistas', default=FINALISTAS)
    ap.add_argument('--baselines', default=BASELINES)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--metric', default='hypervolume')
    ap.add_argument('--metric-label', default='hipervolumen')
    args = ap.parse_args()

    series = build_series(args.finalistas, args.baselines)
    labels = [s.label for s in series]
    faltan = [m for m in BASELINE_KEYS if m not in labels]
    if faltan:
        print(f"  ⚠ sin datos de: {', '.join(faltan)}")
    if len(series) < 3:
        print("Se necesitan más series")
        return
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ETAPA 4 — BASELINES vs ALGORITMOS MULTIOBJETIVO")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = pc.build_reference_front(series, None)
    ind = pc.compute_indicators_per_run(series, None, pf) if pf is not None else {}
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, args.metric)
    if res is None:
        print(f"  ⚠ sin datos de {args.metric}")
        return
    stds = {l: float(np.std(np.asarray(get(l, args.metric), float), ddof=1))
            for l in labels}
    groups = homogeneous_groups(res, labels, res['medians'], True)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}   "
          f"({sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)} de "
          f"{len(res['pairs'])} pares significativos)\n")
    for l in sorted(labels, key=lambda x: -res['medians'][x]):
        tipo = 'baseline' if l in BASELINE_KEYS else 'MOEA'
        print(f"    {DISPLAY.get(l, l):14s} {res['medians'][l]:.5f}  ({tipo})")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_tables(res, groups, res['medians'], stds, labels,
                 args.metric_label, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print("  ✓ tests_pares.csv")


if __name__ == '__main__':
    main()
