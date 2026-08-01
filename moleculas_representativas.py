"""
Una imagen con tres moléculas representativas del frente de cada algoritmo:
mayor QED, menor SA, y mejor balance entre ambos.

El frente de cada algoritmo se arma juntando las moléculas de sus 20 ejecuciones,
deduplicando por SMILES y recalculando la dominancia global.

"Mejor balance" es el punto más cercano al ideal (QED máximo, SA mínima) en el
espacio normalizado.  La normalización usa los extremos de la unión de los cinco
frentes, de modo que el criterio sea el mismo para todos los algoritmos.

Uso:
    python moleculas_representativas.py
    python moleculas_representativas.py --out figura.png
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import plot_comparison as pc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(ROOT_DIR, "plots", "comparacion_final",
                           "moleculas_representativas.png")

DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'MOPSO': 'MOPSO'}

N_MOLECULAS = 5      # por algoritmo


def load_front(alg, finalistas):
    """Frente no dominado global de un algoritmo, sobre sus 20 ejecuciones."""
    df = pc.load_pareto_molecules(os.path.join(finalistas, alg))
    if df.empty:
        return df
    return pc._compute_non_dominated(df.drop_duplicates(subset='smiles'))


def pick(front, n=N_MOLECULAS):
    """Las n moléculas de mayor QED, desempatando por menor SA."""
    f = front.assign(_qed=front['qed'].round(3))
    return (f.sort_values(['_qed', 'sa'], ascending=[False, True])
            .head(n).drop(columns='_qed').reset_index(drop=True))


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
    import io
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format='png')


def main():
    ap = argparse.ArgumentParser(
        description="Moléculas representativas del frente de cada algoritmo.")
    ap.add_argument('--finalistas', default=os.path.join(ROOT_DIR, "finalistas"))
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    fronts = {a: load_front(a, args.finalistas) for a in algs}
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
            ax.set_xlabel(f"QED {m['qed']:.3f}   ·   SA {m['sa']:.2f}",
                          fontsize=9.5, labelpad=3)
            if j == 0:
                ax.set_ylabel(DISPLAY.get(alg, alg), fontsize=13,
                              fontweight='bold', labelpad=10)

    fig.suptitle(f'Las {N_MOLECULAS} moléculas de mayor QED del frente de cada '
                 f'algoritmo',
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
                         'lipinski': m['lipinski'], 'smiles': m['smiles']})
    out = pd.DataFrame(rows)
    csv = os.path.splitext(args.out)[0] + '.csv'
    out.to_csv(csv, index=False)
    print(f"✓ {csv}\n")
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
