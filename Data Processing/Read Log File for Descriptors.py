filename_list = []
for i in range(1,70):
    if i <= 9:
        name = 'e00' + str(i) + '.log'
    else:
        name = 'e0' + str(i) + '.log'
    filename_list.append(name)
#filename_list.remove('e021.log') # 21还没有算完
print(len(filename_list))

def read_log_alkene(filename):
    print('read' + filename + '------starting' )

    f=open(filename,'r')
    templine=f.readlines()
    f.close()

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][1:24] == 'Alpha  occ. eigenvalues':
            k_homo = line
            break
    #print(k_homo)
    homo = templine[k_homo].strip()[-8:]
    lumo = templine[k_homo + 1].strip()[29:37]
    #print(homo)
    #print(lumo)

    for line in list(range(len(templine))):
        c = templine[line]
        if templine[line][1:45] == 'Sum of electronic and thermal Free Energies=':
            k_EE = line
            break
    EE = templine[k_EE][53:65] # 存在多位数
    #print(EE)

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][56:70] == 'Cal/Mol-Kelvin':
            k_entropy = line + 1
            break
    entropy = templine[k_entropy][62:69] # 可能三位数
    #print(entropy)

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][80:84] == 'Tot=':
            k_dipole = line
            break
    dipole = templine[k_dipole][98:104] # 可能三位数
    #print(dipole)

    ### 键长约在1.21左右，用来定位

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][31:50] == 'Optimized Parameter':
            k_length = line + 3
            break

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][1:5] == 'Grad':
            k_length_end = line
            break

    #print(k_length)
    #print(k_length_end)
    for i in range(k_length, k_length_end):
        length = templine[i][33:39]
    #print(length)
        if length[0:4] == '1.21':#三键的长度
            length_C_C = length
            line_R = i
            break
    #print(line_R)


    R = templine[line_R][10:17]
    R = R.strip(' ')
    r = R.split(',')
    r1 = r[0].split('(')[1]
    r2 = r[1].split(')')[0]
    #print(r2)
    r_index = [int(r1), int(r2)]
    #print(r_index)

    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][1:40] == 'Summary of Natural Population Analysis:':
            k_charge_C = line + 6
            break
    for line in list(reversed(range(len(templine)))):
        c = templine[line]
        if templine[line][3:24] == 'Natural Rydberg Basis':
            k_charge_C_end = line
            break

    #print( k_charge_C)
    #print( k_charge_C_end)
    #print(r_index[0])
    for i in range(k_charge_C, k_charge_C_end):

        if templine[i][10:12].strip(' ') == str(r_index[0]):
            #print(i)

            charge_1 = templine[i][15:23]
            #print(charge_1)
            #print(r_index[1])
            
    for i in range(k_charge_C, k_charge_C_end):

        if templine[i][10:12].strip(' ') == str(r_index[1]):
            
            charge_2 = templine[i][15:23]
            #print(charge_2)

    
    if float(charge_2) < float(charge_1):
        charge_x = charge_2
        charge_2 = charge_1
        charge_1 = charge_x
            
        #charge_cc = [charge_1, charge_2]    

    #print(charge_1)
    #print(charge_cc)  
    
    des = [homo, lumo, EE,dipole,entropy,charge_1,charge_2,length]

    des = results = list(map(float, des))
    
    return des

import pandas as pd
DES = []
for name in filename_list:
    try:
        des = read_log_alkene(name)
        DES.append(des)
    except:
        print(name)
    continue
    #DES.append(des)
# write
name_des = ['homo', 'lumo', 'EE','dipole','entropy','charge_1','charge_2','length']
data = pd.DataFrame(columns=name_des,data=DES)
data.to_csv('alkene-log-des.csv', index = None)
