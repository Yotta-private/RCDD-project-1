import csv
import pandas as pd
import numpy as np

*****************************************************************************

data = pd.read_csv('E:/jupyter/workspace/RAC/190508_data/yield_data.csv')
name_ids = list(data.name)
#print(name_ids)
name_id = []
for ids in name_ids:
    ids = 'rac' + ids[3:]
    #print(ids)
    name_id.append(ids)
    
print(len(name_id))

#%chk=a001.chk
#%mem=4GB
#%nprocshared=10
#b3lyp/6-31++g(d,p) pop=nbo opt freq

#name_ids = [] # 批量名称

    
    
def FormatConversion(path, ids):
    
    with open(path + str(ids) + ".gjf", 'r') as f: #str(Index.index)
        lines = [line for line in f.readlines()]
        #print(lines)
    f.close()
    
    for line in list(reversed(range(len(lines)))):
        #print(line)
        Line = lines[line].strip()
    
        #print(line.split(' ')[0])
        if Line.split(' ')[0].isalpha():
            k = line
            break
    #print (k)
        
    w=open(str(ids) + '.gjf','w+')
    w.write('%chk='+ str(ids) + '.chk              ')
    w.write('\n')
    w.write('%mem=15GB              ')
    w.write('\n')
    w.write('%nprocshared=4              ')
    w.write('\n')
    w.write('# b3lyp/6-31++g(d,p) pop=nbo opt              ')        
    w.write('\n')
    w.write('\n')
    w.write(str(ids) + '.sdf             ') 
    w.write('\n')
    w.write('\n')
    #w.write('\n')
    for i in range(9, k+1):
        w.write(lines[i])
            #w.write('\n')
    w.write('\n')
    w.write('\n')
    w.write('\n')
    w.close()

#FormatConversion('E:/jupyter/workspace/RAC/190518_read_log/e_gjf/', 'e001.sdf')

for ids in name_id:
    FormatConversion('E:/jupyter/workspace/RAC/190518_read_log/', ids)
