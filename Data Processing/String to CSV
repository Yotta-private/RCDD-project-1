import numpy as np
import pandas as pd
 
**********************************************************

def read_file_list(FILE_PATH):
    """
    load file
    :return: a list
    """
    with open(FILE_PATH, 'r') as f:
        string_list = f.read().strip().split('\n')
    return string_list


reaction_test = read_file_list('../data/RAC_teststr')
txtDF = pd.DataFrame(reaction_test)
txtDF.to_csv('RAC_reaction_test.csv',index=False)


reaction_train = read_file_list('../data/RAC_trainstr')
txtDF = pd.DataFrame(reaction_train)
txtDF.to_csv('RAC_reaction_train.csv',index=False)
