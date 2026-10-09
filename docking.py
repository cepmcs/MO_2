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
CAJA        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'caja.txt')
SEARCH_MODE = 'balance'
SEED        = 999


# ─── Trabajador: corre en el entorno de Uni-Dock ─────────────────────────────

def _molecula_3d(smi):
    """Molécula con hidrógenos y coordenadas 3D, o None si no se pudo generar."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if (AllChem.EmbedMolecule(mol, randomSeed=SEED) != 0 and
            AllChem.EmbedMolecule(mol, randomSeed=SEED, useRandomCoords=True) != 0):
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def _ligando_pdbqt(smi, prep):
    """Texto PDBQT del ligando de un SMILES, o None si no se pudo preparar."""
    from meeko import PDBQTWriterLegacy
    mol = _molecula_3d(smi)
    if mol is None:
        return None
    try:
        texto, listo, _ = PDBQTWriterLegacy.write_string(prep.prepare(mol)[0])
    except Exception:
        return None
    return texto if listo else None


def _leer_puntaje(ruta):
    """Energía de unión de la pose que Uni-Dock escribió en ruta, o None si no está."""
    try:
        with open(ruta) as f:
            return float(next(linea for linea in f if 'VINA RESULT' in linea).split()[3])
    except (OSError, StopIteration, ValueError):
        return None


def _correr_unidock(ligandos):
    """Dockea en un solo lote {posición: texto PDBQT}.  Devuelve {posición: puntaje}
    de los que salieron; si no sale ninguno, falla con el log de unidock."""
    if not ligandos:
        return {}
    with tempfile.TemporaryDirectory() as d:
        salida = f'{d}/out'
        os.makedirs(salida)
        rutas = {i: f'{d}/l{i}.pdbqt' for i in ligandos}
        for i, texto in ligandos.items():
            with open(rutas[i], 'w') as f:
                f.write(texto)
        cmd = [f'{UNIDOCK_PREFIX}/bin/unidock', '--receptor', RECEPTOR, '--config', CAJA,
               '--search_mode', SEARCH_MODE, '--seed', str(SEED), '--num_modes', '1',
               '--gpu_batch', *rutas.values(), '--dir', salida]
        proc = subprocess.run(cmd, env={**os.environ, 'OMP_NUM_THREADS': '1'},
                              capture_output=True, text=True)
        puntajes = {i: _leer_puntaje(f'{salida}/l{i}_out.pdbqt') for i in ligandos}
    puntajes = {i: p for i, p in puntajes.items() if p is not None}
    if not puntajes:
        raise RuntimeError("unidock no produjo resultados:\n"
                           + (proc.stdout + proc.stderr)[-800:])
    return puntajes


def _dockear(smiles, prep):
    """Puntajes de una lista de SMILES (DOCK_WORST si alguno no se pudo preparar
    o dockear) y cuántos fallaron."""
    ligandos = {}                       # posición en smiles → texto PDBQT
    for i, smi in enumerate(smiles):
        texto = _ligando_pdbqt(smi, prep)
        if texto is not None:
            ligandos[i] = texto
    resultados = _correr_unidock(ligandos)
    puntajes = [resultados.get(i, DOCK_WORST) for i in range(len(smiles))]
    return puntajes, len(smiles) - len(resultados)


def _trabajador():
    from meeko import MoleculePreparation
    RDLogger.DisableLog('rdApp.*')
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
