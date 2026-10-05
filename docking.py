"""
Docking de exp4: puntaje por molécula y caché por run.

El puntaje es la energía de unión (kcal/mol, menor es mejor) que calcula Uni-Dock en
modo balance.  DOCK_BEST y DOCK_WORST son las cotas con que utils_mo normaliza el
hipervolumen.

Uni-Dock y meeko viven en otro entorno conda (UNIDOCK_PREFIX), así que el docking
corre en un proceso aparte: este mismo archivo, ejecutado con el python de ese
entorno, es el trabajador.
"""

import atexit
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

DOCK_BEST  = -15.0
DOCK_WORST = 1.0

UNIDOCK_PREFIX = os.path.expanduser(os.environ.get('UNIDOCK_PREFIX', '~/miniforge3/envs/unidock'))
RECEPTOR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'receptor_8VUX_B.pdbqt')
BOX_CENTER  = (164.6835, 130.2140, 120.1165)
BOX_SIZE    = (30.0, 30.0, 30.0)
SPACING     = 0.375
SEARCH_MODE = 'balance'
SEED        = 42


# ─── Trabajador: corre en el entorno de Uni-Dock ─────────────────────────────

def _dockear(smiles, prep):
    """Puntajes de una lista de SMILES (DOCK_WORST si alguno no se pudo preparar
    o dockear) y cuántos fallaron."""
    from meeko import PDBQTWriterLegacy
    puntajes = [DOCK_WORST] * len(smiles)
    ok = set()
    with tempfile.TemporaryDirectory() as d:
        ligandos = []
        for i, smi in enumerate(smiles):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            if (AllChem.EmbedMolecule(mol, randomSeed=SEED) != 0 and
                    AllChem.EmbedMolecule(mol, randomSeed=SEED, useRandomCoords=True) != 0):
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
            try:
                texto, listo, _ = PDBQTWriterLegacy.write_string(prep.prepare(mol)[0])
            except Exception:
                continue
            if listo:
                ruta = f'{d}/l{i}.pdbqt'
                with open(ruta, 'w') as f:
                    f.write(texto)
                ligandos.append((i, ruta))
        if ligandos:
            salida = f'{d}/out'
            os.makedirs(salida)
            cmd = [f'{UNIDOCK_PREFIX}/bin/unidock', '--receptor', RECEPTOR,
                   '--center_x', str(BOX_CENTER[0]), '--center_y', str(BOX_CENTER[1]),
                   '--center_z', str(BOX_CENTER[2]),
                   '--size_x', str(BOX_SIZE[0]), '--size_y', str(BOX_SIZE[1]),
                   '--size_z', str(BOX_SIZE[2]), '--spacing', str(SPACING),
                   '--search_mode', SEARCH_MODE, '--seed', str(SEED), '--num_modes', '1',
                   '--gpu_batch', *[ruta for _, ruta in ligandos], '--dir', salida]
            proc = subprocess.run(cmd, env={**os.environ, 'OMP_NUM_THREADS': '1'},
                                  capture_output=True, text=True)
            for i, _ in ligandos:
                try:
                    with open(f'{salida}/l{i}_out.pdbqt') as f:
                        puntajes[i] = float(next(l for l in f if 'VINA RESULT' in l).split()[3])
                    ok.add(i)
                except (OSError, StopIteration, ValueError):
                    pass
            if not ok:
                raise RuntimeError("unidock no produjo resultados:\n"
                                   + (proc.stdout + proc.stderr)[-800:])
    return puntajes, len(smiles) - len(ok)


def _trabajador():
    from meeko import MoleculePreparation
    prep = MoleculePreparation(rigid_macrocycles=True)
    for linea in sys.stdin:
        try:
            puntajes, fallos = _dockear(json.loads(linea), prep)
            respuesta = {'scores': puntajes, 'fallos': fallos}
        except Exception as e:
            respuesta = {'error': str(e)}
        sys.stdout.write(json.dumps(respuesta) + '\n')
        sys.stdout.flush()


# ─── Cliente y caché: corren en el entorno del optimizador ───────────────────

class UnidockClient:
    """dock_fn de Uni-Dock: un proceso trabajador que queda vivo toda la run.
    fallos cuenta las moléculas que no se pudieron preparar o dockear."""

    def __init__(self):
        self.proceso = None
        self.fallos = 0

    def __call__(self, smiles):
        if self.proceso is None:
            self.proceso = subprocess.Popen(
                [f'{UNIDOCK_PREFIX}/bin/python', os.path.abspath(__file__)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            atexit.register(self.cerrar)
        self.proceso.stdin.write(json.dumps(list(smiles)) + '\n')
        self.proceso.stdin.flush()
        linea = self.proceso.stdout.readline()
        if not linea:
            raise RuntimeError("el trabajador de docking murió")
        respuesta = json.loads(linea)
        if 'error' in respuesta:
            raise RuntimeError(respuesta['error'])
        self.fallos += respuesta['fallos']
        return respuesta['scores']

    def cerrar(self):
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.stdin.close()
            try:
                self.proceso.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proceso.kill()


class DockingCache:
    """Dockea cada SMILES una sola vez.  Se llama con una lista de SMILES y devuelve
    sus puntajes en el mismo orden."""

    def __init__(self, dock_fn=None):
        self.dock_fn = dock_fn or UnidockClient()
        self.scores = {}
        self.n_docked = 0

    def __call__(self, smiles):
        nuevas = list(dict.fromkeys(s for s in smiles if s not in self.scores))
        if nuevas:
            puntajes = np.asarray(self.dock_fn(nuevas), dtype=float)
            if puntajes.shape != (len(nuevas),) or not np.isfinite(puntajes).all():
                raise ValueError("dock_fn debe devolver un puntaje finito por molécula")
            self.scores.update(zip(nuevas, puntajes.tolist()))
            self.n_docked += len(nuevas)
        return [self.scores[s] for s in smiles]


if __name__ == '__main__':
    if '--check' in sys.argv:
        print(DockingCache()(['CCO', 'c1ccccc1O', 'CC(=O)Nc1ccc(O)cc1']))
    else:
        _trabajador()
