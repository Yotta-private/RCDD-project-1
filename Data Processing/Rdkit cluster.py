# generate fingerprints
fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,1024) for m in reaxys]   
print (len(fps))
clusters=ClusterFps(fps,cutoff=  0.78)
print(len(clusters))


from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
import pandas as pd


#载入数据
#获取数据集中分子
reaxys = [mol for mol in Chem.SDMolSupplier('E:\\jupyter\\cc\\cluster_20190117\\reaxys_all_select3533.sdf') if mol is not None]
### check
print (reaxys[0])
name_check = reaxys[0].GetProp('IDE.RN')
print (name_check)
print (len(reaxys))


#输出数据集中分子包含的属性
properties = reaxys[0].GetPropNames()
for prop in properties:
    print(prop)

### cluster function
def ClusterFps(fps,cutoff=0.2):
    
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
    
    # first generate the distance matrix:
    dists = []   
    nfps = len(fps)   
    for i in range(1,nfps):       
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])       
        dists.extend([1-x for x in sims])
        
    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs

# show one of the clusters
print(clusters[0][0])
cluster_result = []
for i in range(0, len(clusters)):
    cluster_number = clusters[i][0]
    mol = reaxys[cluster_number]
    cluster_result.append(mol)
print (len(cluster_result))


### 将不同聚类分别汇总
for i in range(0, len(clusters)):

    cluster_i= []
    for n in range(0, len(clusters[i])):
        mol = reaxys[n]
        cluster_i.append(mol)
    print (len(cluster_i))
     

# print (clusters[34])

# cluster example
cluster_i= []
for n in clusters[6]:
    mol = reaxys[n]
    cluster_i.append(mol)
print (len(clusters[6]))


legends = []
for mol in cluster_i:
    # concat string for neat printout
    s =  "{0}".format(mol.GetProp('IDE.MF'))
    legends.append(s)
Draw.MolsToGridImage(cluster_i, molsPerRow=4, legends=legends)

#输出cluster数据集中分子包含的属性
properties = cluster_result[0].GetPropNames()
for prop in properties:
    print(prop)

### 输出为sdf文件
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem

import re

w=Chem.SDWriter('E:\\jupyter\\cc\\cluster_20190117\\cluster_result.sdf')

for i in range(len(cluster_result)):
    m = cluster_result[i]
    #print(Chem.MolToMolBlock(m)) 
    # name_m = m.GetProp('IDE.RN')
    # m.SetProp("_Name",name[i])
    AllChem.Compute2DCoords(m)
   
    w.write(m)
w.close()
    

"""# 把环数控制在2个及以下
from rdkit import rdkit.Chem.Lipinski.RingCount(x)"""

legends = []
for mol in cluster_result:
    # concat string for neat printout
    s =  "{0}".format(mol.GetProp('IDE.MF'))
    legends.append(s)
    
Draw.MolsToGridImage(cluster_result, molsPerRow=6, legends=legends)
