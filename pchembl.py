"""
Predictor de pChEMBL para MRP1: Random Forest sobre huellas de Morgan 

"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'random_forest_ic50.pkl')
PCHEMBL_MIN = 4.0
PCHEMBL_MAX = 8.5

_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class PChEMBLPredictor:
    """Se llama con una lista de SMILES válidos y devuelve sus pChEMBL en el mismo orden.
    Cada SMILES se predice una sola vez: la población converge y los repite."""

    def __init__(self):
        import joblib   # acá y no arriba: solo lo necesitan los experimentos que usan pChEMBL
        self.model = joblib.load(MODEL_PATH)['model']
        self.cache = {}

    def __call__(self, smiles):
        nuevos = list(dict.fromkeys(s for s in smiles if s not in self.cache))
        if nuevos:
            X = np.zeros((len(nuevos), 2048), dtype=np.uint8)
            for i, smi in enumerate(nuevos):
                DataStructs.ConvertToNumpyArray(_FP.GetFingerprint(Chem.MolFromSmiles(smi)), X[i])
            # con los nombres de columna con que se ajustó, para que sklearn no avise
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
            self.cache.update(zip(nuevos, self.model.predict(X)))
        return [float(self.cache[s]) for s in smiles]
