"************************************************************"
# Basic molecular features from  rdkit.
# Modify on the basis of deepchem
# __author__ = "yuyao yang"
"************************************************************"

from __future__ import division
from __future__ import unicode_literals
import logging
import types
import numpy as np
import multiprocessing
import rdkit
from rdkit.Chem import Descriptors
"-------------------------------------------------------------------"

class Featurizer(object):
    """
    Abstract class for calculating a set of features for a molecule.
    Child classes implement the _featurize method for calculating features for a single molecule.
    """
    
    def featurize(self, mols, verbose=True, log_every_n=1000):
        """
        Calculate features for molecules.
        Parameters
        ----------
        mols : iterable
        RDKit Mol objects.
        """
        mols = list(mols)
        features = []
    
        for i, mol in enumerate(mols):
            if mol is not None:
                features.append(self._featurize(mol))
            else:
                features.append(np.array([]))

        features = np.asarray(features) # 将结构数据转化为ndarray, 区别于np.array不copy对象

        return features


    def _featurize(self, mol):

        """
        Calculate features for a single molecule.
        Parameters
        ----------
        mol : RDKit Mol
        Molecule.
        """
        raise NotImplementedError('Featurizer is not defined.')

    def __call__(self, mols):

        """
        Calculate features for molecules.
        Parameters
        ----------
        mols : iterable
        RDKit Mol objects.
        """
        return self.featurize(mols)
# list all descriptors in rdkit 2D
features_list = [] 
for i in Descriptors.descList:
    features_list.append(i[0])
    
class MolecularWeight(Featurizer):
    """
    Molecular weight.
    """
    name = ['mw', 'molecular_weight']

    def _featurize(self, mol):
        """
        Calculate molecular weight.
        """

        wt = Descriptors.ExactMolWt(mol)
        wt = [wt]
        return wt

class RDKitDescriptors(Featurizer):
    """
    RDKit descriptors.  #list-of-available-descriptors.
    总数量 = 111 个 变更成200个2D descriptors
    """
    name = 'descriptors'
    allowedDescriptors = features_list
  # (ytz): This is done to avoid future compatibility issues like inclusion of
  # the 3D descriptors or changing the feature size.
    """allowedDescriptors = set([
      'MaxAbsPartialCharge', 'MinPartialCharge', 'MinAbsPartialCharge',
      'HeavyAtomMolWt', 'MaxAbsEStateIndex', 'NumRadicalElectrons',
      'NumValenceElectrons', 'MinAbsEStateIndex', 'MaxEStateIndex',
      'MaxPartialCharge', 'MinEStateIndex', 'ExactMolWt', 'MolWt', 'BalabanJ',
      'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
      'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
      'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10',
      'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
      'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
      'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3',
      'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
      'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
      'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
      'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',
      'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
      'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
      'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
      'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
      'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
      'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
      'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
      'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
      'NumRotatableBonds', 'NumSaturatedCarbocycles',
      'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
      'MolMR' ])"""

    def __init__(self):
        self.descriptors = []
        self.descList = []

        for descriptor, function in Descriptors.descList:
            if descriptor in self.allowedDescriptors:
                self.descriptors.append(descriptor)
                self.descList.append((descriptor, function))

    def _featurize(self, mol):
        rval = []
        for desc_name, function in self.descList:
            rval.append(function(mol))
        return rval

### check
from rdkit import Chem
smiles = 'c1ccccc1CC1CC1'
mol = Chem.MolFromSmiles(smiles)
#  Test MolecularWeight
MW = MolecularWeight()
mw = MW([mol])
print(mw)

# TestRDKitDescriptors
DES = RDKitDescriptors()
descriptors = DES([mol])
print(descriptors.shape) 
#print(descriptors)
