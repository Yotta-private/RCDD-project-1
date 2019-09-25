
# coding: utf-8

# In[8]:


from __future__ import print_function
from saving import save_model_history, save_model_history_manual
from neural_fp import sizeAttributeVectors, sizeAttributeVector
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, merge
from keras.layers.core import Flatten, Permute, Reshape, Dropout, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import *
# from keras.utils.visualize_util import plot
import numpy as np
import datetime
import json
import sys
import os
from tqdm import tqdm
import keras.backend as K
import theano.tensor as T
from GraphFP_embedding import *


# In[15]:


K.is_nan = T.isnan 
K.logical_not = lambda x: 1 - x


def mse_no_NaN(y_true, y_pred):
    '''For each sample, sum squared error ignoring NaN values'''
    return K.sum(K.square(K.switch(K.logical_not(K.is_nan(y_true)), y_true, y_pred) - y_pred), axis = -1)

def binary_crossnetropy_no_NaN(y_true, y_pred):
    return K.sum(K.binary_crossentropy(K.switch(K.is_nan(y_true), y_pred, y_true), y_pred), axis = -1)

def build_model(embedding_size = 512, lr = 0.01, optimizer = 'adam', depth = 2, scale_output = 0.05, 
                padding = False, hidden = 0, hidden2 = 0, loss = 'mse', hidden_activation = 'tanh',output_activation = 'softmax', 
                dr1 = 0.0, dr2 = 0.0, output_size = 1, sum_after = False,molecular_attributes = True, use_fp = None, inner_rep = 32):
    
    '''Generates simple embedding model to use molecular tensor as input in order to predict a single-valued output (i.e., yield)
    inputs:
        embedding_size - size of fingerprint for GraphFP layer
        lr - learning rate to use (train_model overwrites this value)
        optimizer - optimization function to use
        depth - depth of the neural fingerprint (i.e., radius)
        scale_output - initial scale for output weights in GraphFP
        padding - whether or not molecular tensors will be padded (i.e., batch_size > 1)
        hidden - number of hidden tanh nodes after FP (0 is linear)
        hidden2 - number of hidden nodes after "hidden" layer
        hidden_activation - activation function used in hidden layers
        output_activation - activation function for final output nodes
        dr1 - dropout rate after embedding
        dr2 - dropout rate after hidden
        loss - loss function as a string (e.g., 'mse')
        sum_after - whether to sum neighbor contributions after passing
                    through a single network layer, or to do so before
                    passing them from the network layer (during updates)
        molecular_attributes - whether to include additional molecular 
                    attributes in the atom-level features (recommended)
        use_fp - whether the representation used is actually a fingerprint
                    and not a convolutional network (for benchmarking)
    outputs:
        model - a Keras model'''



    # Base model
    if type(use_fp) == type(None):
        
        #F = 40
        
        input_features = Input(shape = (None,None, 40,), name = "feature graph")
        ecfps = Input(shape = (512,), name = "ecfps")
        
        FPS = []
        
        
        nn_fps = GraphFP(output_dim = embedding_size, inner_dim = 39, 
                         depth = depth,scale_output = scale_output,
                  padding = padding, activation_inner = 'tanh')(input_features)

        print('    model: added GraphFP layer ({} -> {})'.format('mol_a & mol_e', "512"))
        
        FPS.append(ecfps)
        FPS.append(nn_fps)
        
        FPs = merge(FPS, mode = 'concat', concat_axis= 1, name = 'mix fps')

        
        if hidden > 0:
            h1 = Dense(hidden, activation = hidden_activation)(FPs)
            h1d = Dropout(dr1)(h1)
            print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden))
            if hidden2 > 0:
                h2 = Dense(hidden2, activation = hidden_activation)(h1)
                print('    model: added {} Dense layer (-> {})'.format(hidden_activation, hidden2))
                h = Dropout(dr2)(h2)
            else:
                h = h1d
        else:
            h = FPs

    ypred = Dense(output_size, activation = output_activation)(h)
    print('    model: added output Dense layer (-> {})'.format(output_size))



    if type(use_fp) == type(None):
        model = Model(input = [input_features, ecfps], 
            output = [ypred])
    else:
        model = Model(input = [FPs], 
            output = [ypred])

    model.summary()
    model.layers

    # Compile
    if optimizer == 'adam':
        optimizer = Adam(lr = lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = lr)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr = lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta()
    else:
        print('Unrecognized optimizer')
        quit(1)

    # Custom loss to filter out NaN values in multi-task predictions
    if loss == 'custom':
        loss = mse_no_NaN
    elif loss == 'custom2':
        loss = binary_crossnetropy_no_NaN

    print('compiling...',)
    model.compile(loss = loss, optimizer = optimizer)
    print('done')

    return model

def save_model(model, loss, val_loss, fpath = '', config = {}, tstamp = ''):
    '''Saves NN model object and associated information.
    inputs:
        model - a Keras model
        loss - list of training losses 
        val_loss - list of validation losses
        fpath - root filepath to save everything to (with .json, h5, png, info)
        config - the configuration dictionary that defined this model 
        tstamp - current timestamp to log in info file'''

    # Dump data
    with open(fpath + '.json', 'w') as structure_fpath:
        json.dump(model.to_json(), structure_fpath)
    print('...saved structural information')

    # Dump weights
    model.save_weights(fpath + '.h5', overwrite = True)
    print('...saved weights')

    # # Dump image
    # plot(model, to_file = fpath + '.png')
    # print('...saved image')

    # Dump history
    save_model_history_manual(loss, val_loss, fpath + '.hist')
    print ('...saved history')

    # Write to info file
    info_fid = open(fpath + '.info', 'a')
    info_fid.write('{} saved {}\n\n'.format(fpath, tstamp))
    info_fid.write('Configuration details\n------------\n')
    info_fid.write('  {}\n'.format(config))
    info_fid.close()

    print('...saved model to {}.[json, h5, png, info]'.format(fpath))
    return True


def train_model(model, data, nb_epoch = 0, batch_size = 1, lr_func = None, patience = 10):
    '''Trains the model.
    inputs:
        model - a Keras model
        data - three dictionaries for training,
                validation, and testing separately
        nb_epoch - number of epochs to train for
        batch_size - batch_size to use on the data. This must agree with what was
                specified for data (i.e., if tensors are padded or not)
        lr_func - string which is evaluated with 'epoch' to produce the learning 
                rate at each epoch 
        patience - number of epochs to wait when no progress is being made in 
                the validation loss. a patience of -1 means that the model will
                use weights from the best-performing model during training
    outputs:
        model - a trained Keras model
        loss - list of training losses corresponding to each epoch 
        val_loss - list of validation losses corresponding to each epoch'''

    # Unpack data 
    (train, val, test) = data
    mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']
    mols_val   = val['mols'];   y_val   = val['y'];   smiles_val   = val['smiles']
    print('{} to train on'.format(len(mols_train)))
    print('{} to validate on'.format(len(mols_val)))
    print('{} to test on'.format(len(smiles_val)))

    # Create learning rate function
    if lr_func:
        lr_func_string = 'def lr(epoch):\n    return {}\n'.format(lr_func)
        exec (lr_func_string)


    # Fit (allows keyboard interrupts in the middle)
    # Because molecular graph tensors are different sizes based on N_atoms, can only do one at a time
    # (alternative is to pad with zeros and try to add some masking feature to GraphFP)
    # -> this is why batch_size == 1 is treated distinctly
    try:
        loss = []
        val_loss = []

        if batch_size == 1: # DO NOT NEED TO PAD
            wait = 0
            prev_best_val_loss = 99999999
            for i in range(nb_epoch):
                this_loss = []
                this_val_loss = []
                if lr_func: model.optimizer.lr.set_value(lr(i))
                print('Epoch {}/{}, lr = {}'.format(i + 1, nb_epoch, model.optimizer.lr.get_value()))

                # Run through training set
                print('Training...')
                training_order = list(range(len(mols_train)))
                np.random.shuffle(training_order)
                for j in training_order:
                    single_mol = mols_train[j]
                    single_y_as_array = np.reshape(y_train[j], (1, -1))
                    sloss = model.train_on_batch(
                        [np.array([single_mol[0]]), np.array([single_mol[1]])],
                        single_y_as_array
                    )
                    this_loss.append(sloss)

                # Run through testing set
                print('Validating..')
                for j in range(len(mols_val)):
                    single_mol = mols_val[j]
                    single_y_as_array = np.reshape(y_val[j], (1, -1))
                    sloss = model.test_on_batch(
                        [np.array([single_mol[0]]), np.array([single_mol[1]])],
                        single_y_as_array
                    )
                    this_val_loss.append(sloss)

                loss.append(np.mean(this_loss))
                val_loss.append(np.mean(this_val_loss))
                print('loss: {}\tval_loss: {}'.format(loss[i], val_loss[i]))

                # Check progress
                if np.mean(this_val_loss) < prev_best_val_loss:
                    wait = 0
                    prev_best_val_loss = np.mean(this_val_loss)
                    if patience == -1:
                        model.save_weights('best.h5', overwrite=True)
                else:
                    wait = wait + 1
                    print('{} epochs without val_loss progress'.format(wait))
                    if wait == patience:
                        print('stopping early!')
                        break
            if patience == -1:
                model.load_weights('best.h5')

        else: 
            # When the batch_size is larger than one, we have padded mol tensors
            # which  means we need to concatenate them but can use Keras' built-in
            # training functions with callbacks, validation_split, etc.
            if lr_func:
                callbacks = [LearningRateScheduler(lr)]
            else:
                callbacks = []
            if patience != -1:
                callbacks.append(EarlyStopping(patience = patience))

            if mols_val:
                mols = np.vstack((mols_train, mols_val))
                #mols = list(mols_train) + list(mols_val)
                feature_graphs =[]
                ecfps = []
                for i in range(len(mols)):
                    feature_graph = mols[i][0]
                    ecfp_e = mols[i][1]
                    feature_graphs.append(feature_graph)
                    ecfps.append(ecfp_e)
                ########################################################
                y = np.concatenate((y_train, y_val))
                hist = model.fit({'feature graph': np.array(feature_graphs), 'ecfps': np.array(ecfps)}, y,           
                    nb_epoch = nb_epoch, 
                    batch_size = batch_size, 
                    validation_split = (1 - float(len(mols_train))/(len(mols_val) + len(mols_train))),
                    callbacks = callbacks)
            else:
                hist = model.fit(np.array(mols_train), np.array(y_train), 
                    nb_epoch = nb_epoch, 
                    batch_size = batch_size, 
                    callbacks = callbacks)

            loss = []; val_loss = []
            if 'loss' in hist.history: loss = hist.history['loss']
            if 'val_loss' in hist.history: val_loss = hist.history['val_loss']

    except KeyboardInterrupt:
        print('User terminated training early (intentionally)')

    return (model, loss, val_loss)


# In[17]:


"""build_model(embedding_size = 512, lr = 0.01, optimizer = 'adam', depth = 2, scale_output = 0.05, 
                padding = False, hidden = 0, hidden2 = 0, loss = 'mse', hidden_activation = 'tanh',output_activation = 'softmax', 
                dr1 = 0.0, dr2 = 0.0, output_size = 1, sum_after = False,molecular_attributes = True, use_fp = None, inner_rep = 32)"""

