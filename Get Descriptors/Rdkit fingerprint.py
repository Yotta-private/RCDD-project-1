"************************************************************"
# molecular Topological fingerprints from  rdkit.
# Modify on the basis of deepchem
"************************************************************"

from __future__ import division
from __future__ import unicode_literals
"-------------------------------------------------------------------"
import logging
import types
import numpy as np
import multiprocessing
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
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
    

class CircularFingerprint(Featurizer):

    """
    Circular (Morgan) fingerprints.
    Parameters
    ----------
    radius : int, optional (default 2) Fingerprint radius.
    size : int, optional (default 2048) Length of generated bit vector.
    chiral : bool, optional (default False) Whether to consider chirality in fingerprint generation.
    bonds : bool, optional (default True) Whether to consider bond order in fingerprint generation.
    features : bool, optional (default False) Whether to use feature information instead of atom information;
    sparse : bool, optional (default False) Whether to return a dict for each molecule containing the sparse fingerprint.
    smiles : bool, optional (default False) Whether to calculate SMILES strings for fragment IDs 
    (only applicable when calculating sparse fingerprints).
    """

    name = 'circular'
    
    def __init__(self, radius=2, size=2048,chiral=False,bonds=True,features=False,sparse=False,smiles=False):
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features
        self.sparse = sparse
        self.smiles = smiles

    def _featurize(self, mol):
        if self.sparse:
            info = {}
            fp = rdMolDescriptors.GetMorganFingerprint(mol,self.radius,useChirality=self.chiral,
                                                       useBondTypes=self.bonds,useFeatures=self.features,bitInfo=info)
            fp = fp.GetNonzeroElements()  # convert to a dict

      # generate SMILES for fragments

            if self.smiles:
                fp_smiles = {}
                for fragment_id, count in fp.items():
                
                    root, radius = info[fragment_id][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root)
                    frag = Chem.PathToSubmol(mol, env)
                    smiles = Chem.MolToSmiles(frag)
                    fp_smiles[fragment_id] = {'smiles': smiles, 'count': count}
                
                fp = fp_smiles
        
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,self.radius, nBits=self.size,useChirality=self.chiral,
                                                                useBondTypes=self.bonds,useFeatures=self.features)

        return fp

    def __hash__(self):
        return hash((self.radius, self.size, self.chiral, self.bonds, self.features,self.sparse, self.smiles))

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False

        return self.radius == other.radius and \
                   self.size == other.size and \
                   self.chiral == other.chiral and \
                   self.bonds == other.bonds and \
                   self.features == other.features and \
                   self.sparse == other.sparse and \
                   self.smiles == other.smiles

# TestCircularFingerprint
#smiles = 'c1ccccc1'
#mol =Chem.MolFromSmiles(smiles)
#FP = CircularFingerprint(sparse=True,smiles=True, radius=5)
# FP = CircularFingerprint(sparse=False)  #bit
#fp = FP([mol])
#print(fp)
# fragment_id: count
#[{98513984: 6, 2763854213: 6, 3218693969: 6}] sparse=True,smiles=false
#[{98513984: {'smiles': 'ccc', 'count': 6}, 2763854213: {'smiles': 'ccccc', 'count': 6}, 3218693969: {'smiles': '', 'count': 6}}] sparse=True,smiles=True
