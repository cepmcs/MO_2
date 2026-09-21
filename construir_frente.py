"""
Frente de referencia para IGD+ y ε+: la no-dominancia sobre [-QED, SA] de todas
las corridas de los experimentos que se le pasen.

IGD+ y ε+ no son absolutos: dependen de este frente y de su ideal/nadir.  Todas
las tablas que se comparen entre sí tienen que medirse contra el mismo archivo.

Cada carpeta es un experimento (results/main, results/exp3, ...) y se leen todos
sus run_*/molecules.csv.  Solo entran experimentos con la restricción de Fsp3: si
aparece una molécula por debajo del umbral, aborta.

    python construir_frente.py                        # cada subcarpeta de results/
    python construir_frente.py results/main results/exp3 --out otro.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from utils_mo import FSP3_MIN, RESULTS_DIR, ROOT_DIR, _non_dominated_front

COLS = ['smiles', 'qed', 'sa', 'fsp3']


def leer(experimento):
    """Moléculas de todas las corridas de un experimento, sin repetir SMILES."""
    archivos = sorted({os.path.realpath(p) for p in glob.glob(
        os.path.join(experimento, '**', 'run_*', 'molecules.csv'), recursive=True)})
    dfs = [d for d in (pd.read_csv(p, usecols=COLS) for p in archivos) if not d.empty]
    df = (pd.concat(dfs, ignore_index=True).drop_duplicates('smiles')
          if dfs else pd.DataFrame(columns=COLS))
    return df, len(archivos)


def main():
    ap = argparse.ArgumentParser(description="Frente de referencia para IGD+ y ε+.")
    ap.add_argument('experimentos', nargs='*',
                    help="Carpetas de experimentos (default: cada subcarpeta de results/).")
    ap.add_argument('--out', default=os.path.join(ROOT_DIR, 'frente_referencia.csv'))
    args = ap.parse_args()

    experimentos = args.experimentos or sorted(
        d for d in glob.glob(os.path.join(RESULTS_DIR, '*')) if os.path.isdir(d))
    if not experimentos:
        ap.error(f"no hay experimentos en {RESULTS_DIR}")
    nombres = [os.path.basename(os.path.normpath(e)) for e in experimentos]

    partes = []
    for exp, nombre in zip(experimentos, nombres):
        df, n = leer(exp)
        if n == 0:
            ap.error(f"{exp}: no tiene run_*/molecules.csv")
        malas = int((df['fsp3'] < FSP3_MIN - 1e-9).sum())
        if malas:
            ap.error(f"{exp}: {malas} moléculas con Fsp3 < {FSP3_MIN}.  ¿Es un "
                     f"experimento sin la restricción?  No puede entrar al frente.")
        print(f"  {nombre:12s} {n:6d} corridas  {len(df):7d} moléculas únicas")
        partes.append(df.assign(experimento=nombre))

    todo = pd.concat(partes, ignore_index=True)
    # Qué experimentos halló cada molécula, antes de deduplicar entre ellos.
    origen = todo.groupby('smiles')['experimento'].agg(
        lambda s: ';'.join(sorted(set(s))))
    todo = todo.drop_duplicates('smiles').reset_index(drop=True)
    F = np.column_stack([-todo['qed'].to_numpy(float), todo['sa'].to_numpy(float)])
    frente = todo.iloc[_non_dominated_front(F)][COLS].copy()
    frente['experimentos'] = frente['smiles'].map(origen)
    frente = frente.sort_values('sa').reset_index(drop=True)
    frente.to_csv(args.out, index=False)

    print(f"\n✅ {args.out}: {len(frente)} moléculas   "
          f"QED {frente.qed.min():.3f}-{frente.qed.max():.3f}   "
          f"SA {frente.sa.min():.2f}-{frente.sa.max():.2f}")
    for nombre in nombres:
        n = int(frente['experimentos'].str.split(';').map(lambda l: nombre in l).sum())
        print(f"  las halla {nombre}: {n}")


if __name__ == "__main__":
    main()
