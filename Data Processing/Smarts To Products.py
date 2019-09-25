import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

supp2 = Chem.SDMolSupplier('E:/jupyter/cc/20181204-products/e.sdf')
supp3 = Chem.SDMolSupplier('E:/jupyter/cc/20181204-products/a-br.sdf')
mol2=[]
mol3=[]
for mol_2 in supp2:
    mol2.append(mol_2)
for mol_3 in supp3:
    mol3.append(mol_3)
print (len(mol2))
print (len(mol3))
print (Chem.MolToSmiles(mol2[2]))
print (Chem.MolToSmiles(mol3[0]))

# rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger.uncharge 去电荷
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.six import StringIO
sio = StringIO()
w = Chem.SDWriter(sio)

rxn = AllChem.ReactionFromSmarts('[N:2]1(:[C]:[C]([C]):[N:4]:[N:3]:1)[C].[*:8][C:5]#[C:6].[*:7][Br]>>[N:2]1([C:6]=[C:5]([*:8])[N:4]=[N:3]1)[*:7]')
rxn.GetNumReactantTemplates()
x1 = Chem.MolFromSmarts('[N]1(:[C]:[C]([C]):[N]:[N]:1)[C]')
#print (len(x))
name = []
product_smiles = []
for i in range(len(mol2)):
    x2 = mol2[i]
    for j in range(len(mol3)):
        x3 = mol3[j]
        x = rxn.RunReactants((x1,x2,x3))
        #print (len(x))
        c = x[0][0]
        #w.write(c)
        smiles = Chem.MolToSmiles(c)
        product_smiles.append(smiles)
        if i < 9:
            if j < 9:
                name_product = 'RACa00' + str(j+1)+ 'e00' + str(i+1)
            else:
                name_product = 'RACa0' + str(j+1)+ 'e00' + str(i+1)
        if i >= 9:
            if j < 9:
                name_product = 'RACa00' + str(j+1)+ 'e0' + str(i+1)
            else:
                name_product = 'RACa0' + str(j+1)+ 'e0' + str(i+1)
                
        name.append(name_product)
        
        
# print (product_smiles)
print (name[0])
#w.flush()
#print(sio.getvalue())
# sio.getvalue()
#print (product_smiles)  
        #print (Chem.MolToMolBlock(c))  

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem

import re

# file_object=open("decoys.smi")
w=Chem.SDWriter("E:/jupyter/cc/20181204-products/products_20190307_a58e37.sdf")
#for line in file_object.readlines() :
    #[smi,zid,pid]=line.split()
for i in range(len(product_smiles)):
    m=Chem.MolFromSmiles(product_smiles[i])
    #print(Chem.MolToMolBlock(m)) 
    m.SetProp("_Name",name[i])
    AllChem.Compute2DCoords(m)
   
    w.write(m)
w.close()
    

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem

import re
for i in range(len(product_smiles)):
# file_object=open("decoys.smi")
    w=Chem.SDWriter("E:/jupyter/cc/20181204-products/products_spilt/" + name[i] + ".sdf")
#for line in file_object.readlines() :
    #[smi,zid,pid]=line.split()
#for i in range(len(product_smiles)):
    m=Chem.MolFromSmiles(product_smiles[i])
    #print(Chem.MolToMolBlock(m)) 
    m.SetProp("_Name",name[i])
    AllChem.Compute2DCoords(m)
   
    w.write(m)
w.close()
    
