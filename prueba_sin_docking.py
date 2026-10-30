"""
Prueba rápida SIN GPU ni Uni-Dock, para el nodo de login: corre unas pocas generaciones de un experimento
con el docking reemplazado por puntajes falsos (deterministas, por hash del SMILES).  Sirve para
comprobar el entorno (pymoo, scikit-learn, VAE, pChEMBL, restricciones) antes de gastar cola.

Acepta los mismos argumentos que experimento.py; por defecto, --device cpu, --n_gen 2 y --run_id 0.
    python prueba_sin_docking.py --alg NSGA3 --vae mrp1 --pchembl --logp
Escribe en results/prueba_sin_docking y lo borra al terminar.
"""
import hashlib
import os
import runpy
import shutil
import sys

import pandas as pd

import docking

EXP_NAME = 'prueba_sin_docking'


class DockingFalso:
    """Mismo contrato que docking.DockingCache, con puntajes entre -12 y -4 sacados del SMILES."""

    def __init__(self):
        self.puntajes, self.fallos = {}, 0

    @property
    def n_docked(self):
        return len(self.puntajes)

    def __call__(self, smiles):
        for s in smiles:
            if s not in self.puntajes:
                self.puntajes[s] = -12 + 8 * int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return [self.puntajes[s] for s in smiles]


def con_defecto(argv, opcion, valor):
    return argv if opcion in argv else argv + [opcion, valor]


docking.DockingCache = DockingFalso        # antes de importar utils_mo, que lo toma de acá al cargarse
from utils_mo import results_dir  # noqa: E402
argv = sys.argv[1:]
for opcion, valor in (('--device', 'cpu'), ('--n_gen', '2'), ('--run_id', '0'), ('--exp_name', EXP_NAME)):
    argv = con_defecto(argv, opcion, valor)
sys.argv = ['experimento.py'] + argv

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimento.py'), run_name='__main__')

base = results_dir(argv[argv.index('--exp_name') + 1])
molecules = [os.path.join(d, 'molecules.csv') for d, _, fs in os.walk(base) if 'molecules.csv' in fs]
print(f"\nOK: {len(molecules)} corrida(s) completa(s). Columnas de molecules.csv: "
      f"{list(pd.read_csv(molecules[0]).columns)}" if molecules else "\nERROR: no se escribió molecules.csv")
if os.path.basename(base) == EXP_NAME:
    shutil.rmtree(base)
