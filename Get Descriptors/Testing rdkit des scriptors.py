#"************************************************************"
# Get_descriptors from yield data.------just a test code
# __author__ = "yuyao yang"
#"************************************************************"

from Rdkit fingerprint import *
from Rdkit 2D DES import *
import numpy as np
import pandas as pd
import csv
import rdkit
from rdkit.Chem import Descriptors

data = pd.read_csv("RAC_data.csv")
#print(data.head(2))
smiles_a = list(data.name_a)
smiles_e = list(data.name_e)
smiles_p = list(data.name_p)

mol_a = [Chem.MolFromSmiles(smi) for smi in smiles_a]
# print(len(mol_a))
mol_e = [Chem.MolFromSmiles(smi) for smi in smiles_e]
mol_p = [Chem.MolFromSmiles(smi) for smi in smiles_p]

# des from rdkit 2D
MW = MolecularWeight()
DES = RDKitDescriptors()

# des from ecfp (hash)
FP_sparse = CircularFingerprint(sparse=False,radius=5, size=1024) # ecfp 10 vector bit = 2048 
FP_share = CircularFingerprint(sparse=True,smiles=False, radius=5) # ecfp 10

#***********************************************************************
# get 2d des
mol_weight_a = MW(mol_a) # (702, 1)
des_a = DES(mol_a) # shape (702, 111)
# concatenate
a_des =  np.concatenate((mol_weight_a,des_a),axis=1)
#print(a_des.shape) # (702, 112)

mol_weight_e = MW(mol_e) # (702, 1)
des_e = DES(mol_e) # shape (702, 111)
# concatenate
e_des =  np.concatenate((mol_weight_e,des_e),axis=1)
#print(a_des.shape) # (702, 112)

mol_weight_p = MW(mol_p) # (702, 1)
des_p = DES(mol_p) # shape (702, 111)
# concatenate
p_des =  np.concatenate((mol_weight_p,des_p),axis=1)
#print(a_des.shape) # (702, 112)

des_second = np.concatenate((a_des,e_des),axis=1)
des_final = np.concatenate((des_second,p_des),axis=1)

# print(des_final.shape) shape------(702, 336)
#***********************************************************************

# write to csv files
np.savetxt('RAC_rdkit_2D_des.csv', des_final, delimiter=',')

#***********************************************************************
# get  sparse ecfp10
fp_a = FP_sparse(mol_a)
#print(fp_a)
fp_e = FP_sparse(mol_e)
fp_p = FP_sparse(mol_p)
fp_sencond = np.concatenate((fp_a,fp_e),axis=1)
fp_sparse = np.concatenate((fp_sencond ,fp_p),axis=1)
# print(fp_sparse.shape) shape (702, 3072)
np.savetxt('RAC_ecfp10_sparse.csv', fp_sparse, delimiter=',')
#***********************************************************************

#***********************************************************************
# get  dense ecfp10
# 保留 count信息和fragment信息
from collections import Counter
fp_a = FP_share(mol_a)
fp_e = FP_share(mol_e)
fp_p = FP_share(mol_p)
fp_share = [] # 702
for i in range(702):
    X,Y = Counter(fp_a[i]), Counter(fp_e[i])
    z = dict(X+Y)
    X,Y = Counter(z), Counter(fp_p[i])
    fp_sample = dict(X+Y)
    fp_share .append(fp_sample) 

# print( FP_dense) 返回共有的key值，以count表示value值
ID = []
for i in range(702):
    for fragment, counts in fp_share [i].items():
        ID.append(fragment)
    #print(v)
ID = list(set(ID))
#print(len(ID)) # 所有切割出来的自片段总共有个 （其实后续是可以返回去看看到底是什结构类型）

# print(fp_share[0][19852674]) 按照KEY值去查找结构片段counts值
fp = []
for i in range(702):
    fp_counts = []
    for ids in ID:
        try:
            count = fp_share[i][ids]
        except KeyError:
            count = 0
        fp_counts.append(count)
    fp.append(fp_counts)

#print(fp[0])
fp = np.array(fp)
# print(fp.shape) # (702, 6178)
np.savetxt('RAC_ecfp10_share.csv', fp, delimiter=',')
#print(fp.shape)

## get 6000多个分子片段的smi 返回txt
FP_check = CircularFingerprint(sparse=True,smiles=True, radius=5) 
from collections import Counter
fp_a = FP_check(mol_a)
fp_e = FP_check(mol_e)
fp_p = FP_check(mol_p)

# 为手动创建的6178个smi和对应hash值做一个查询表
directory = []
for i in range(702):
    for k, v in fp_a[i].items():
        # 有些地方的'smiles': ''， 应该就是该分子中对应位置的原子本身,这里先不做处理
        List = [(k, v['smiles'])]
        dic = dict(List)
        #print(dic)
        directory.append(dic)
        
for i in range(702):
    for k, v in fp_e[i].items():
        # 有些地方的'smiles': ''， 应该就是该分子中对应位置的原子本身,这里先不做处理
        List = [(k, v['smiles'])]
        dic = dict(List)
        #print(dic)
        directory.append(dic)
        
for i in range(702):
    for k, v in fp_p[i].items():
        # 有些地方的'smiles': ''， 应该就是该分子中对应位置的原子本身,这里先不做处理
        List = [(k, v['smiles'])]
        dic = dict(List)
        #print(dic)
        directory.append(dic)

# print(len(directory)) # 17458
# 去重
from functools import reduce
def list_dict_duplicate_removal(data_list):
    #data_list = [{"a": "123", "b": "321"}, {"a": "124", "b": "321"}, {"b": "321", "a": "123"}]
    run_function = lambda x, y: x if y in x else x + [y]
    return reduce(run_function, [[], ] + data_list)

lookup_table = list_dict_duplicate_removal(directory)

print(len(lookup_table)) # 6232 原因是有对称情况存在比如，Cc和cC的hash一样，所以正确的应该是上面的6178种
# print(lookup_table)

# write into txt
path = r'RAC_ecfp10_lookup_table.txt'
f = open(path,'w',encoding='utf-8')
for m in lookup_table:
    for k,v in m.items():
        ids = str(k)
        f.write(ids)
        f.write("      ")
        f.write(v+'\n')
f.close()
