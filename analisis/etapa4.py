"""
Etapa 4 — baselines contra los algoritmos multiobjetivo.

Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de suma
ponderada) con los cinco MOEAs seleccionados, sobre el mismo presupuesto de
100.000 evaluaciones y las mismas 20 semillas.
"""

import glob
import os

import numpy as np
import pandas as pd

from .comun import (
    ALGORITHM_ORDER,
    DISPLAY,
    Series,
    _build_series_value_getter,
    _fmt_p,
    _has_runs,
    _num,
    _write_tex,
    compare_indicator,
    fmt_groups,
    homogeneous_groups,
)
from .indicadores import build_reference_front, compute_indicators_per_run



# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 4 — baselines contra los algoritmos multiobjetivo
#
#   Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de
#   suma ponderada) con los cinco MOEAs ya seleccionados, sobre el mismo
#   presupuesto de 100.000 evaluaciones y las mismas 20 semillas.
# ═══════════════════════════════════════════════════════════════════════════

# Orden de peor a mejor esperado.
BASELINE_KEYS = ['WEIGHTED_GA', 'HILL_CLIMBER', 'RANDOM', 'SCREENING']



def _series_baselines(finalistas, baselines):
    """Los cinco MOEAs y las cuatro baselines como series comparables."""
    series = []
    for alg in ALGORITHM_ORDER:
        d = os.path.join(finalistas, alg)
        if _has_runs(d):
            series.append(Series(alg, d, color_key=alg))
    for m in BASELINE_KEYS:
        # <baselines>/<METHOD>/[tag/]pop{P}_gen{G}/
        for d in sorted(glob.glob(os.path.join(baselines, m, '*', '*')) +
                        glob.glob(os.path.join(baselines, m, '*'))):
            if _has_runs(d):
                series.append(Series(m, d, color_key=m))
                break
    return series



def write_baseline_tables(res, groups, medians, stds, labels, metric_label,
                          out_dir):
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
                     f"${_num(medians[lab], 4)} \\pm {_num(stds[lab], 4)}$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'comparacion.tex'))

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
    _write_tex(lines, os.path.join(out_dir, 'grupos_homogeneos.tex'))



def etapa4(args):
    series = _series_baselines(args.finalistas, args.baselines)
    labels = [s.label for s in series]
    faltan = [m for m in BASELINE_KEYS if m not in labels]
    if faltan:
        print(f"  ⚠ sin datos de: {', '.join(faltan)}")
    if len(series) < 3:
        print("Se necesitan más series")
        return
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ETAPA 4 — BASELINES vs ALGORITMOS MULTIOBJETIVO")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = build_reference_front(series)
    ind = compute_indicators_per_run(series, pf) if pf is not None else {}
    get = _build_series_value_getter(series, ind)

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

    write_baseline_tables(res, groups, res['medians'], stds, labels,
                          args.metric_label, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print("  ✓ tests_pares.csv")
