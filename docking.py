"""
Docking de exp4: puntaje por molécula.

El puntaje es la energía de unión (kcal/mol, menor es mejor) que calcula Uni-Dock en
modo balance.  DOCK_BEST y DOCK_WORST son las cotas con que utils_mo normaliza el
hipervolumen.

Todo corre en el entorno del optimizador (RDKit y meeko).  Del entorno de Uni-Dock
(UNIDOCK_PREFIX) solo se usa el ejecutable unidock.
"""

import os
import subprocess
import tempfile

from meeko import MoleculePreparation, PDBQTWriterLegacy
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

PREP = MoleculePreparation(rigid_macrocycles=True)


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


def _ligando_pdbqt(smi):
    """Texto PDBQT del ligando de un SMILES, o None si no se pudo preparar."""
    mol = _molecula_3d(smi)
    if mol is None:
        return None
    try:
        texto, listo, _ = PDBQTWriterLegacy.write_string(PREP.prepare(mol)[0])
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


def dockear(smiles):
    """Puntajes de una lista de SMILES (DOCK_WORST si alguno no se pudo preparar
    o dockear) y cuántos fallaron."""
    ligandos = {}                       # posición en smiles → texto PDBQT
    for i, smi in enumerate(smiles):
        texto = _ligando_pdbqt(smi)
        if texto is not None:
            ligandos[i] = texto
    resultados = _correr_unidock(ligandos)
    puntajes = [resultados.get(i, DOCK_WORST) for i in range(len(smiles))]
    return puntajes, len(smiles) - len(resultados)


class DockingCache:
    """Dockea cada SMILES una sola vez por run.  Se llama con una lista de SMILES y
    devuelve sus puntajes en el mismo orden.  fallos cuenta las moléculas que no se
    pudieron preparar o dockear."""

    def __init__(self):
        self.puntajes = {}
        self.fallos = 0

    @property
    def n_docked(self):
        return len(self.puntajes)

    def __call__(self, smiles):
        nuevas = list(dict.fromkeys(s for s in smiles if s not in self.puntajes))
        if nuevas:
            puntajes, fallos = dockear(nuevas)
            self.puntajes.update(zip(nuevas, puntajes))
            self.fallos += fallos
        return [self.puntajes[s] for s in smiles]


if __name__ == '__main__':
    print(dockear(['CCO', 'c1ccccc1O', 'CC(=O)Nc1ccc(O)cc1'])[0])
