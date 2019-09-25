
# coding: utf-8

# In[2]:


from neural_fp import *
import rdkit.Chem as Chem
import numpy as np
import os
import csv


# In[31]:


def get_data(data_label = '', shuffle_seed = None, batch_size = 1, data_split = 'cv', cv_folds = '1/1',
                 truncate_to = None, training_ratio = 0.9,molecular_attributes = True, use_fp = None, padding = True):
    
    '''This is a helper script to read the data file and return the training and test data sets separately. This is to allow for an
    already-trained model to be evaluated using the test data (i.e., which we know it hasn't seen before)'''

    # Roots
    #data_label = data_label.lower()
    #data_froot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    print('reading data...')
    data = []
    with open("RAC_data.csv", 'r') as data_fid:
        reader = csv.reader(data_fid, quotechar = '"')
        for row in reader:
            if row[0] != 'smiles_a':
                data.append(row)
    print('done')

    # Truncate if necessary
    if truncate_to is not None:
        data = data[:truncate_to]
        print('truncated data to first {} samples'.format(truncate_to))

    # Get new shuffle seed if possible
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)


    smiles = []
    mols = []
    y = []
    y_label = 'yeild_classfication'
    print('processing data...')
    # Randomize
    np.random.shuffle(data)
    for i, row in enumerate(data):
        try:
            # Molecule first (most likely to fail)
            mol = Chem.MolFromSmiles(row[0], sanitize = False)
            Chem.SanitizeMol(mol)
            mol_2 = Chem.MolFromSmiles(row[1], sanitize = False)
            Chem.SanitizeMol(mol_2)

            input_features = molToGraph(mol, molecular_attributes = molecular_attributes).dump_as_tensor()
            input_features = padGraphTensor(input_features, 30)
            ecfps = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_2,3,nBits=512,useFeatures=True))

            # Are we trying to use Morgan FPs?
            if use_fp == 'Morgan':
                mat_features = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=512,useFeatures=True))
                #print(mol_tensor)
            elif use_fp == 'Morgan2':
                mat_features = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512,useFeatures=True))
            elif type(use_fp) != type(None):
                print('Unrecognised use_FP option {}'.format(use_fp))



            mols.append((input_features,ecfps))

            y.append(row[3]) # Measured log(solubility M/L)
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles = True)) # Smiles

        except Exception as e:
            print('Failed to generate graph for {}, y: {}'.format(row[smiles_index], row[y_index]))
            print(e)


    if 'ratio' in data_split: # split train/notrain
        print('Using first fraction ({}) as training'.format(training_ratio))
        # Create training/development split
        division = int(len(mols) * training_ratio)
        mols_train = mols[:division]
        mols_notrain  = mols[division:]
        y_train = y[:division]
        y_notrain  = y[division:]
        smiles_train = smiles[:division]
        smiles_notrain = smiles[division:]

        # Split notrain up
        cut_index = int(len(mols_notrain) / 2)
                        
        mols_val = mols_notrain[:cut_index] # first half
        y_val = y_notrain[:cut_index] # first half
        smiles_val = smiles_notrain[:cut_index] # first half
        mols_test = mols_notrain[cut_index:] # second half
        y_test = y_notrain[cut_index:] # second half
        smiles_test = smiles_notrain[cut_index:] # second half
        print('Training size: {}'.format(len(mols_train)))
        print('Validation size: {}'.format(len(mols_val)))
        print('Testing size: {}'.format(len(mols_test)))

    elif 'all_train' in data_split: # put everything in train 
        print('Using ALL as training')
        # Create training/development split
        mols_train = mols
        y_train = y
        smiles_train = smiles
        mols_val    = []
        y_val       = []
        smiles_val  = []
        mols_test   = []
        y_test      = []
        smiles_test = []
        print('Training size: {}'.format(len(mols_train)))
        print('Validation size: {}'.format(len(mols_val)))
        print('Testing size: {}'.format(len(mols_test)))

    elif 'cv' in data_split: # cross-validation
        # Default to first fold of 5-fold cross-validation
        folds = 5
        this_fold = 0

        # Read fold information
        try:
            folds = int(cv_folds.split('/')[1])
            this_fold = int(cv_folds.split('/')[0]) - 1
        except:
            pass

        # Get target size of each fold
        N = len(mols)
        print('Total of {} mols'.format(N))
        target_fold_size = int(np.ceil(float(N) / folds))
        # Split up data
        folded_mols = [mols[x:x+target_fold_size]   for x in range(0, N, target_fold_size)]
        folded_y = [y[x:x+target_fold_size]      for x in range(0, N, target_fold_size)]
        folded_smiles = [smiles[x:x+target_fold_size] for x in range(0, N, target_fold_size)]
        print('Split data into {} folds'.format(folds))
        print('...using fold {}'.format(this_fold + 1))

        # Recombine into training and testing
        mols_train   = [x for fold in (folded_mols[:this_fold] + folded_mols[(this_fold + 1):])     for x in fold]
        y_train      = [x for fold in (folded_y[:this_fold] + folded_y[(this_fold + 1):])           for x in fold]
        smiles_train = [x for fold in (folded_smiles[:this_fold] + folded_smiles[(this_fold + 1):]) for x in fold]
        # Test is this_fold
        mols_test    = folded_mols[this_fold]
        y_test       = folded_y[this_fold]
        smiles_test  = folded_smiles[this_fold]

        # Define validation set as random 10% of training
        training_indices = list(range(len(mols_train)))
        np.random.shuffle(training_indices)
        split = int(len(training_indices) * training_ratio)
        mols_train,   mols_val    = [mols_train[i] for i in training_indices[:split]],   [mols_train[i] for i in training_indices[split:]]
        y_train,      y_val       = [y_train[i] for i in training_indices[:split]],      [y_train[i] for i in training_indices[split:]]
        smiles_train, smiles_val  = [smiles_train[i] for i in training_indices[:split]], [smiles_train[i] for i in training_indices[split:]]

        print('Total training: {}'.format(len(mols_train)))
        print('Total validation: {}'.format(len(mols_val)))
        print('Total testing: {}'.format(len(mols_test)))

    else:
        print('Must specify a data_split type of "ratio" or "cv"')
        quit(1)


    if 'cv_full' in data_split: # cross-validation, but use 'test' as validation
        train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
        val   = {}; val['mols']   = mols_test;   val['y']   = y_test;   val['smiles']   = smiles_test;   val['y_label']   = y_label
        test  = {}; test['mols']  = [];  test['y']  = [];  test['smiles']  = []; test['y_label']  = []

    else:

        train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
        val   = {}; val['mols']   = mols_val;   val['y']   = y_val;   val['smiles']   = smiles_val;   val['y_label']   = y_label
        test  = {}; test['mols']  = mols_test;  test['y']  = y_test;  test['smiles']  = smiles_test; test['y_label']  = y_label

    return (train, val, test)


# In[35]:


"""data = get_data(data_label = '', shuffle_seed = None, batch_size = 1, data_split = 'cv', cv_folds = '1/5',
                 truncate_to = None, training_ratio = 0.9,molecular_attributes = True, use_fp = None)"""


# In[36]:


"""train, val, test = data
print(train['mols'][0][0])"""

