import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

ms = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CCO')]
fps = [FingerprintMols.FingerprintMol(x) for x in ms]
DataStructs.FingerprintSimilarity(fps[0],fps[1])
