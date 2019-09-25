import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
import Smipar

def reaction_str_from_filename(fname):
    """Takes a file containing \n separated reaction strings to get the reaction_str"""
    with open(fname, 'r') as f:
        reaction_strs = f.read().strip().split('\n')
    return reaction_strs
        
def compound_list(reaction_strs):
    "list compound"
    # reaction_strs = self.reaction_str_from_filename()
    compound_list_a = []
    compound_list_e = []
    product_list = []
    for reaction_str in reaction_strs:
        compound_list_a.append(split_reaction_function(reaction_str)[0])
        compound_list_e.append(split_reaction_function(reaction_str)[1])
        product_list.append(split_reaction_function(reaction_str)[2])
    return compound_list_a, compound_list_e, product_list

def split_reaction_function(reaction_str):
    "Spilt the reaction string into two reactants and a products"
    split_rs = reaction_str.split('>')
    product = split_rs[1]
    reactants = split_rs[0].split('.')
    reactant_a = reactants[0]
    reactant_e = reactants[1]
    return reactant_a, reactant_e, product
#print(split_reaction_function('CC=C(C)C.Cl>CCC(C)(C)Cl'))
      
def tokenize(compound_list_a, compound_list_e,  product_list):
    """Takes a splited Reaction SMILES and return a list of tokens"""
    tokens = []
    for i in range(len(compound_list_a)):
        token = ['GO']+ Smipar.parser_list(compound_list_a[i]) + ['.'] + Smipar.parser_list(compound_list_e[i]) +['>'] + Smipar.parser_list(product_list[i])
        token.append('EOS')
        tokens = tokens + token
    return tokens
#tokenize('CC=C(C)C.Cl>CCC(C)(C)Cl')
            
def construct_vocabulary(tokens):
    """Adds tokens to the vocabulary"""
    vocab = {}
    for token in tokens:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 1
    token_list = sorted(vocab, key=vocab.get, reverse=True) 
    return token_list
# construct_vocabulary(tokenize('CC=C(C)C.Cl>CCC(C)(C)Cl'))
    
def write_to_file(List1, fname):
    """Write a list to a file."""
    with open(fname, 'w') as f:
        for s in List1:
            f.write(s)
            f.write("\n")   
    f.close()

def encode(token_list):
    """Takes a list of tokens (eg '[nH]') and encodes to array of indices"""
    vocab = {}
    for token in tokens:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 1       
    reaction_str_matrix = np.zeros(len(token_list), dtype=np.float32)
    for i, token in enumerate(token_list):
        reaction_str_matrix[i] = vocab[token]
    return reaction_str_matrix

def decode(matrix):
    "Takes an array of indices and returns the corresponding reaction_str_SMILES"
    tokens = []
    for i in matrix:
        if i == self.vocab['EOS']: break
    tokens.append(self.reversed_vocab[i])
    reaction_str = "".join(tokens)
    return reaction_str  

reaction_strs = reaction_str_from_filename('ALL_RAC_reactionstr')
compound_list = compound_list(reaction_strs)
tokens = tokenize(compound_list[0], compound_list[1], compound_list[2])
token_list = construct_vocabulary(tokens)
print("Vocabulary containing {} tokens: {}".format(len(token_list), token_list))
write_to_file(token_list, 'Voc-RAC')
reaction_str_matrix  = encode(token_list)
#print('-------------------------------------------------')
#print(reaction_str_matrix)
