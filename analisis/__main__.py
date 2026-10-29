"""
Línea de comandos del análisis.

Cada comando lee results/exp4/ (--results) y escribe en plots/exp4/.  Las seis
configuraciones (NSGA-II, NSGA-III y AGE-MOEA, cada uno con PCX y con SBX) se
analizan juntas (ver comandos.py):

  comparacion  las seis en una sola comparación (con sus frentes)
  candidatos   frente conjunto de las seis y 15 moléculas
  todo         los dos comandos

Ambos escriben en plots/exp4/resultados/ (--out): tablas/, figuras/ y frentes/ (los
frentes de cada configuración y el general, en CSV).

Uso:
    python -m analisis todo
    python -m analisis candidatos [--csv top15.csv]
"""

import argparse

from .comun import OUT_DIR, RESULTADOS_DIR
from .comandos import candidatos, comparacion


def todo(args):
    comparacion(args)
    candidatos(args)


def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    ap.add_argument('comando', choices=['comparacion', 'candidatos', 'todo'])
    ap.add_argument('--results', default=RESULTADOS_DIR,
                    help="Carpeta con los resultados del experimento.")
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--completo', action='store_true',
                    help="Reporte completo: cajas químicas, frentes superpuestos, grids "
                         "QED–SA y CSV intermedios.")
    ap.add_argument('--csv', default=None,
                    help="CSV con rank,smiles,qed,sa,fsp3,dock para los candidatos "
                         "(default: <results>/top15_docking.csv).")
    args = ap.parse_args()
    {'comparacion': comparacion, 'candidatos': candidatos, 'todo': todo}[args.comando](args)


if __name__ == '__main__':
    main()
