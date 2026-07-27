"""
Etapa 3: comparación estadística final entre algoritmos.

Lee finalistas/<ALG>/run_XX/ (la configuración elegida de cada algoritmo tras
las etapas 1 y 2) y contrasta los cinco entre sí.

Para cada indicador: Friedman con las 20 semillas como bloques y, si resulta
significativo, las 10 comparaciones por pares con Wilcoxon de rangos con signo
corregidas por Holm.  El resultado se resume en grupos homogéneos: dentro de una
llave el post-hoc no separa, entre llaves sí.

Salida (plots/comparacion_final/)
  grupos_homogeneos.tex   una fila por indicador
  tests_pares.csv         las 10 comparaciones de cada indicador, con p de Holm

Uso:
    python comparar_algoritmos.py
    python comparar_algoritmos.py --finalistas otra_carpeta
"""

import os
import argparse

import numpy as np
import pandas as pd

import plot_comparison as pc
from comparar_operadores import compare_indicator, homogeneous_groups, _fmt_p

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALISTAS_DIR = os.path.join(ROOT_DIR, "finalistas")
OUT_DIR = os.path.join(ROOT_DIR, "plots", "comparacion_final")

# Nombres para el documento (los directorios usan la forma corta).
DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO'}

METRIC = 'hypervolume'
METRIC_LABEL = 'hipervolumen'
HIGHER_BETTER = True


def fmt_groups(groups):
    """'{A, B} > {C}' con los nombres de presentación."""
    return ' $>$ '.join('\\{' + ', '.join(DISPLAY.get(x, x) for x in g) + '\\}'
                        for g in groups)


def write_groups_table(res, groups, out_dir):
    """Las 10 comparaciones por pares, con el resultado resumido en grupos."""
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación entre algoritmos sobre {METRIC_LABEL}.  '
        f'Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm.  '
        f'Grupos homogéneos, de mejor a peor: {fmt_groups(groups)}.}}',
        r'\label{tab:comparacion_grupos}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Par & $p$ (Holm) & Significativo \\', r'\midrule',
    ]
    for p in res['pairs']:
        sig = 'sí' if p['p_holm'] < 0.05 else 'no'
        lines.append(f"{DISPLAY.get(p['a'], p['a'])} vs {DISPLAY.get(p['b'], p['b'])} "
                     f"& {_fmt_p(p['p_holm'])} & {sig} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    path = os.path.join(out_dir, 'grupos_homogeneos.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\n  ✓ {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Etapa 3: comparación estadística entre algoritmos.")
    ap.add_argument('--finalistas', default=FINALISTAS_DIR)
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    series = pc.build_finalist_series(algs, args.finalistas)
    if len(series) < 3:
        print(f"Se necesitan ≥3 algoritmos en {args.finalistas}")
        return
    labels = [s.label for s in series]
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ETAPA 3 — COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = pc.build_reference_front(series, None)
    ind = pc.compute_indicators_per_run(series, None, pf) if pf is not None else {}
    print(f"  frente de referencia: {len(pf) if pf is not None else 0} soluciones\n")
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, METRIC)
    if res is None:
        print(f"  ⚠ sin datos de {METRIC}")
        return
    groups = homogeneous_groups(res, labels, res['medians'], HIGHER_BETTER)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}\n")
    for lab in sorted(labels, key=lambda l: -res['medians'][l]):
        print(f"    {DISPLAY.get(lab, lab):10s} mediana = {res['medians'][lab]:.5f}")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_groups_table(res, groups, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print(f"  ✓ {os.path.join(args.out, 'tests_pares.csv')}")


if __name__ == '__main__':
    main()
