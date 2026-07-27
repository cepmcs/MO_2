"""
Etapa 2: comparación de combinaciones de operadores, por algoritmo.

Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (las configuraciones que
ganaron su bloque en la etapa 1) y compara los 4 combos entre sí.

Salida por algoritmo (plots_operadores/<ALG>/)
  comparison_multiobj_*.tex     indicadores multiobjetivo (HV, spacing, IGD+, ε+)
  comparison_chemical_*.tex     indicadores químicos (QED, SA, validez, unicidad…)
  pareto_comparison_*.png       frentes de Pareto superpuestos
  pareto_qed_sa_grid_*.png      frentes QED-SA en paneles
  tests_<ALG>.tex               Friedman + Wilcoxon post-hoc con Holm
  ganadores.csv                 el combo elegido de cada algoritmo

Las 20 semillas están pareadas (mismo run_id → misma población inicial), así que
el test toma la semilla como bloque: Friedman sobre los 4 combos y, si resulta
significativo, las 6 comparaciones por pares con Wilcoxon de rangos con signo,
corrigiendo por Holm dentro de cada indicador.

Uso:
    python comparar_operadores.py
    python comparar_operadores.py --algorithms NSGA2 MOEAD
    python comparar_operadores.py --metric igd_plus
"""

import os
import glob
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import plot_comparison as pc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WINNERS_DIR = os.path.join(ROOT_DIR, "winners")
# Bajo .../operadores/<ALG>/ para que las figuras de frentes titulen con el
# nombre del algoritmo (plot_comparison lo deduce de la ruta de salida).
OUT_DIR = os.path.join(ROOT_DIR, "plots_operadores", "operadores")

GA_ALGS = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA']
COMBO_ORDER = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']

# (columna, etiqueta, mayor_es_mejor)
INDICATORS = [
    ('hypervolume', 'Hipervolumen',      True),
    ('igd_plus',    'IGD$^+$',           False),
    ('epsilon',     r'$\epsilon^+$',     False),
    ('spacing',     'Espaciamiento',     False),
    ('n_pareto',    'Tamaño de Pareto',  True),
    ('validity',    'Validez',           True),
    ('uniqueness',  'Unicidad',          True),
]


# ─── Series desde winners/ ───────────────────────────────────────────────────

def build_series(alg, winners_dir):
    """Una serie por combo de operadores.  El nivel de configuración se resuelve
    con glob porque cada combo ganó con hiperparámetros distintos."""
    series = []
    for combo in COMBO_ORDER:
        matches = sorted(glob.glob(os.path.join(winners_dir, alg, combo, '*')))
        cfg_dirs = [d for d in matches if pc._has_runs(d)]
        if not cfg_dirs:
            continue
        series.append(pc.Series(combo, cfg_dirs[0], color_key=combo))
    return series


# ─── Tests ───────────────────────────────────────────────────────────────────

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


def compare_indicator(get_values, labels, col):
    """Friedman sobre los combos con la semilla como bloque + post-hoc.

    Devuelve dict con el p de Friedman, el tamaño de efecto W de Kendall, y la
    lista de comparaciones por pares con su p corregido por Holm."""
    cols = {}
    for lab in labels:
        v = get_values(lab, col)
        if v is None:
            return None
        cols[lab] = np.asarray(v, dtype=float)

    n = min(len(v) for v in cols.values())
    if n < 3:
        return None
    M = np.column_stack([cols[lab][:n] for lab in labels])   # semillas × combos

    m, k = M.shape
    R = np.apply_along_axis(stats.rankdata, 1, M)
    chi2 = 12 / (m * k * (k + 1)) * np.sum(R.sum(axis=0) ** 2) - 3 * m * (k + 1)
    W = float(max(chi2 / (m * (k - 1)), 0.0))
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

    return {'p_omnibus': p_omni, 'W': W, 'n_blocks': m,
            'medians': {lab: float(np.median(cols[lab][:n])) for lab in labels},
            'pairs': [{'a': a, 'b': b, 'p_raw': pr, 'p_holm': pa}
                      for (a, b), pr, pa in zip(pairs, raw, adj)]}


def rank_combos(get_values, labels, col, higher_better):
    """Rango medio de cada combo, rankeando dentro de cada semilla."""
    cols = []
    for lab in labels:
        v = get_values(lab, col)
        if v is None:
            return None
        cols.append(np.asarray(v, dtype=float))
    n = min(len(v) for v in cols)
    M = np.column_stack([v[:n] for v in cols])
    A = -M if higher_better else M
    R = np.apply_along_axis(stats.rankdata, 1, A)
    return dict(zip(labels, R.mean(axis=0)))


# ─── Tablas ──────────────────────────────────────────────────────────────────

def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '---'
    return r'$<$0.001' if p < 1e-3 else f'{p:.3f}'


def write_tests_table(res, alg, out_dir, label):
    """Tabla LaTeX de las 6 comparaciones por pares sobre el indicador de
    decisión.  La magnitud del efecto no se repite acá: está en la tabla de
    indicadores, como media ± desvío por combo."""
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación de operadores en {pc._latex_escape(alg)} sobre '
        f'{label.lower()}.  Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm.}}',
        f'\\label{{tab:ops_tests_{alg.lower()}}}',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'Par & $p$ (Holm) & Significativo \\', r'\midrule',
    ]
    for pr in res['pairs']:
        sig = 'sí' if pr['p_holm'] < 0.05 else 'no'
        lines.append(f"{pc._latex_escape(pr['a'])} vs {pc._latex_escape(pr['b'])} & "
                     f"{_fmt_p(pr['p_holm'])} & {sig} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    path = os.path.join(out_dir, f'tests_{alg}.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ tests_{alg}.tex")


def homogeneous_groups(res, labels, medians, higher_better):
    """Grupos de combos que el post-hoc no logra separar, ordenados de mejor a
    peor.  Notación estándar para resumir 6 comparaciones en una línea."""
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


# ─── Orquestación ────────────────────────────────────────────────────────────

def analyze(alg, winners_dir, out_root, decision_col):
    series = build_series(alg, winners_dir)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    # Frente de referencia común a los 4 combos → IGD+ y ε+ comparables.
    pf_F, pf_df = pc.build_reference_front(series, None)
    indicator_data = {}
    if pf_F is not None:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicator_data = pc.compute_indicators_per_run(series, None, pf_F)
        pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = pc._build_series_value_getter(series, indicator_data)

    # Las 4 salidas de la sección 3.2: 2 tablas + 2 figuras de frentes.
    pc.generate_latex_comparison_tables(series, alg, out_dir, get_values)
    pc.plot_pareto_comparison(series, alg, out_dir)
    pc.plot_pareto_qed_sa_grid(series, alg, out_dir)

    # Test: solo sobre el indicador de decisión.  Los demás indicadores se
    # reportan de forma descriptiva en las tablas de comparación.
    label, higher = dict((c, (l, h)) for c, l, h in INDICATORS)[decision_col]
    res = compare_indicator(get_values, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)
    print(f"  Friedman ({label}): p = {res['p_omnibus']:.4g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}")

    write_tests_table(res, alg, out_dir, label)

    groups = homogeneous_groups(res, labels, res['medians'], higher)
    txt = ' > '.join('{' + ', '.join(g) + '}' for g in groups)
    print(f"  grupos homogéneos: {txt}")

    return {'algorithm': alg, 'p_friedman': res['p_omnibus'],
            'n_pares_sig': n_sig, 'n_pares': len(res['pairs']),
            'grupos': txt,
            'mejor_grupo': ', '.join(groups[0])}


def main():
    ap = argparse.ArgumentParser(
        description="Etapa 2: comparación de operadores por algoritmo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--winners', default=WINNERS_DIR)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--algorithms', nargs='+', default=None)
    ap.add_argument('--metric', default='hypervolume',
                    choices=[c for c, _, _ in INDICATORS],
                    help="Indicador con el que se elige el combo ganador.")
    args = ap.parse_args()

    algs = args.algorithms or GA_ALGS
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  ETAPA 2 — COMPARACIÓN DE OPERADORES")
    print(f"  Datos: {args.winners}")
    print(f"  Decisión por: {args.metric}")
    print(f"{'='*64}")

    rows = []
    for alg in algs:
        r = analyze(alg, args.winners, args.out, args.metric)
        if r:
            rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")


if __name__ == '__main__':
    main()
