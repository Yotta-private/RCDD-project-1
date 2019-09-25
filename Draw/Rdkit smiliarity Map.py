import matplotlib.pyplot as plt
%matplotlib inline
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True

mol = Chem.MolFromSmiles("BrC1=CC2=C(OCO2)C=C1") 
smi = Chem.MolToSmiles(mol) 
smi

#weight = [0, 0.0025, 0.01, 0.038, 0.01, 0,0,0,0,0.002,0,0,0,0,0,0,0.002,0,0]
#weight = [0,0,0,0,0.005,0.12,0.005,0,0,0,0,0]
#weight = [0,0.002,0,0,0.05,0.09,0.002,0]
weight = [0,0.005,0.005,0.005,0.005,0.005,0.005,0.03,0.01,0.03]
mol = Chem.MolFromSmiles("Brc1ccc2c(c1)OCO2") 
#colors = ['PiYG', 'PiYG_r', 'PuOr', 'PuOr_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',  'coolwarm','coolwarm_r','seismic','seismic_r']
fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weight, size=(800,800), colorMap="seismic", 
                                                 contourLines=None, alpha=None, step=0.01, coordScale=1)  #  coordScale=1.2
#plt.show()
plt.savefig('attention.tif', bbox_inches = 'tight')# dpi = 1200
