
# coding: utf-8

# In[36]:


import numpy as np
import keras.backend as K
import theano.tensor as T 
import theano
import keras
from keras import activations, initializers, regularizers, Sequential
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input
from keras.engine.topology import Layer
from neural_fp import Graph, molToGraph
import rdkit
from rdkit import Chem 


class GraphFP(Layer):
    
    #嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]] Embedding层只能作为模型的第一层
    
    """Embedding layer for undirected, attributed graphs following the ideas of the extended connectivity fingerprints (ECFPs) or functional connectivity FPs (FCFPs).
    It should be the first layer in a model.
    
    # Input shape
        4D array with shape:(n_samples, n_atoms, n_atoms, n_features)`.
        
    # Output shape
        2D tensor with shape:(n_samples, output_dim)`.
        
    # Arguments
        output_dim: int > 0, size of the fingerprint
        
        inner_dim: (32+8-1)the number of attributes for each (bond, atom) pair concatenated. Does NOT include the extract is_bond_present flag.
        
        depth: radius of fingerprint (how many times to recursively mix attributes)
        
        init_output: initialization for weights in output layer
        
        activation_output: activation function for output layer. Softmax is recommended because it can help increase sparsity, making it more like a real fingerprint
        init_inner: initialization for inner weights for mixing attributes. Identity is recommended for the initialization for simplicity
        
        activation_inner: activation function for the inner layer.
        
        scale_output: scale to use for output weight initializations. Large output weights are closer to a true sparse fingerprint, but small output weights might
            be better to not get stuck in local minima (with low gradients)
            
        padding: whether to look for padding in the input tensors. 
    """

    def __init__(self, output_dim, inner_dim, depth = 2, init_output='uniform', activation_output='softmax', init_inner='identity',
            activation_inner='linear', scale_output=0.01, padding=False, **kwargs): # **kwargs表示关键字参数，它是一个dict
        
        if depth < 1:
            quit('Cannot use GraphFP with depth zero')
            
        self.init_output = initializers.get(init_output)
        self.activation_output = activations.get(activation_output)
        self.init_inner = initializers.get(init_inner)
        self.activation_inner = activations.get(activation_inner)
        self.output_dim = output_dim
        self.inner_dim = inner_dim
        self.depth = depth
        self.scale_output = scale_output
        self.padding = padding

        self.initial_weights = None
        self.input_dim = 4 # each entry is a 3D N_atom x N_atom x N_feature tensor
        if self.input_dim:
            kwargs['input_shape'] = (None, None, None,) # 3D tensor for each input
        #self.input = K.placeholder(ndim = 4)
        super(GraphFP, self).__init__(**kwargs)

        
        
        #if self.input_dim:
        #self.input_shape = input_shape #  define 3D tensor for each input # input_shape = (None, None, None,)

    def build(self, input_shape):
        
        '''Builds internal weights and paramer attribute'''
        # NOTE: NEED TO TILE AND EVALUATE SO THAT PARAMS CAN BE VARIABLES 可变
        # OTHERWISE K.GET_VALUE() DOES NOT WORK
        # keras2.0 vs keras 1.0
        # initializer = keras.initializers.RandomUniform(-1, 1)
        # config = initializer.get_config()
        # initializer = keras.initializers.RandomUniform()
        # print(keras.initializers.RandomUniform().__call__((3, 1)).eval())

        # Define template weights for inner FxF ------shape
        
        W_inner = keras.initializers.Identity().__call__((self.inner_dim, self.inner_dim)) #FxF
        b_inner = keras.initializers.Zeros().__call__((1, self.inner_dim)) # 1xF
        
        self.W_inner = K.variable(T.tile(W_inner, (self.depth + 1, 1, 1)).eval() +         keras.initializers.RandomUniform().__call__((self.depth + 1, self.inner_dim, self.inner_dim)).eval()) # T构造符号变量；T.tile(x, reps) 按照规则重复 x
        self.W_inner.name = 'T:W_inner' # eval()将字符串str当成有效的表达式来求值并返回计算结果
        self.b_inner = K.variable(T.tile(b_inner, (self.depth + 1, 1, 1)).eval()  +         keras.initializers.RandomUniform().__call__((self.depth + 1, 1, self.inner_dim)).eval())
        self.b_inner.name = 'T:b_inner'
        
        #print(self.W_inner.get_value())
        #print( self.b_inner.get_value())

        # # Concatenate third dimension (depth) so different layers can have 
        # # different weights. Now, self.W_inner[#,:,:] corresponds to the 
        # # weight matrix for layer/depth #.

        # Define template weights for output FxL
        
        W_output = keras.initializers.VarianceScaling(scale=self.scale_output, distribution='uniform').__call__((self.inner_dim, self.output_dim)) # scale_output=0.01
        b_output = keras.initializers.Zeros().__call__((1, self.output_dim))
        
        # Initialize weights tensor
        self.W_output = K.variable(T.tile(W_output, (self.depth + 1, 1, 1)).eval())
        self.W_output.name = 'T:W_output'
        self.b_output = K.variable(T.tile(b_output, (self.depth + 1, 1, 1)).eval())
        self.b_output.name = 'T:b_output'
        
        # # Concatenate third dimension (depth) so different layers can have 
        # # different weights. Now, self.W_output[#,:,:] corresponds to the 
        # # weight matrix for layer/depth #.
        
        #print(self.W_output.get_value())
        #print(self.b_output.get_value())
        
        # Pack params
        
        self.trainable_weights = [self.W_inner, 
                       self.b_inner,
                       self.W_output,
                       self.b_output]
        self.params = [self.W_inner, 
                       self.b_inner,
                       self.W_output,
                       self.b_output]
        
        super(GraphFP, self).build(input_shape)
        #return self.trainable_weights, self.params
    
#W_output = [return][0][2]
#L = GraphFP(output_dim=512, inner_dim=4, depth = 2)
#w = GraphFP.build(L, input_shape = (4,4,40,))

#c = GraphFP(output_dim=3, inner_dim=4)
#super(GraphFP, c).build("input_shape")
#c.build("input_shape")
#c = GraphFP(output_dim=3, inner_dim=4)
#super(GraphFP, c).build("input_shape")
#c.build("input_shape")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        (output, updates) = theano.scan(lambda x_one: self.get_output_singlesample(x_one), sequences = x)
        return output

    def get_output_singlesample(self, original_graph):
        '''For a 3D tensor, get the output. Avoids the need for even more complicated vectorization'''
        # Check padding
        if self.padding:
            rowsum = original_graph.sum(axis = 0) # add across
            trim = rowsum[:, -1] # last feature == bond flag
            trim_to = T.eq(trim, 0).nonzero()[0][0] # first index with no bonds
            original_graph = original_graph[:trim_to, :trim_to, :] # reduced graph

        # Get attribute values for layer zero
        # where attributes is a 2D tensor and attributes[#, :] is the vector of
        # concatenated node and edge attributes. In the first layer (depth 0), the 
        # edge attribute section is initialized to zeros. After increasing depth, howevevr,
        # this part of the vector will become non-zero.

        # The first attributes matrix is just graph_tensor[i, i, :], but we can't use that 
        # kind of advanced indexing
        # Want to extract tensor diagonal as matrix, but can't do that directly...
        # Want to loop over third dimension, so need to dimshuffle
        (attributes, updates) = theano.scan(lambda x: x.diagonal(), sequences = original_graph.dimshuffle((2, 0, 1)))
        attributes.name = 'attributes'
        # Now the attributes is (N_features x N_nodes), so we need to transpose
        attributes = attributes.T
        attributes.name = 'attributes post-transpose'

        # Get initial fingerprint
        presum_fp = self.attributes_to_fp_contribution(attributes, 0)
        fp = K.sum(presum_fp, axis = 0) # sum across atom contributions
        fp.name = 'initial fingerprint'

        # Get bond matrix
        bonds = original_graph[:, :, -1] # flag if the bond is present, (N_atom x N_atom)
        bonds.name = 'bonds'

        # Iterate through different depths, updating attributes each time
        graph = original_graph
        for depth in range(self.depth):
            (attributes, graph) = self.attributes_update(attributes, depth + 1, graph, original_graph, bonds)
            presum_fp_new = self.attributes_to_fp_contribution(attributes, depth + 1)
            presum_fp_new.name = 'presum_fp_new contribution'
            fp = fp + K.sum(presum_fp_new, axis = 0) 

        return fp

    def attributes_update(self, attributes, depth, graph, original_graph, bonds):
        '''Given the current attributes, the current depth, and the graph that the attributes
        are based on, this function will update the 2D attributes tensor'''

        ############# GET NEW ATTRIBUTE MATRIX #########################
        # New pre-activated attribute matrix v = M_i,j,: x ones((N_atom, 1)) -> (N_atom, N_features) 
        # as long as dimensions are appropriately shuffled
        shuffled_graph = graph.copy().dimshuffle((2, 0, 1)) # (N_feature x N_atom x N_atom)
        shuffled_graph.name = 'shuffled_graph'

        ones_vec = K.ones_like(attributes[:, 0]) # (N_atom x 1)
        ones_vec.name = 'ones_vec'
        (new_preactivated_attributes, updates) = theano.scan(lambda x: K.dot(x, ones_vec), sequences = shuffled_graph) # (N_features x N_atom)

        # Need to pass through an activation function still
        # Final attribute = bond flag = is not part of W_inner or b_inner
        (new_attributes, updates) = theano.scan(lambda x: self.activation_inner(
            K.dot(x, self.W_inner[depth, :, :]) + self.b_inner[depth, 0, :]), sequences = new_preactivated_attributes[:-1, :].T) # (N_atom x N_features -1)

        # Append last feature (bond flag) after the loop
        new_attributes = K.concatenate((new_attributes, attributes[:, -1:]), axis = 1)
        new_attributes.name = 'new_attributes'


        ############ UPDATE GRAPH TENSOR WITH NEW ATOM ATTRIBUTES ###################
        ### Node attribute contribution is located in every entry of graph[i,j,:] where
        ### there is a bond @ ij or when i = j (self)
        # Get atoms matrix (identity)
        atoms = T.identity_like(bonds) # (N_atom x N_atom)
        atoms.name = 'atoms_identity'
        # Combine
        bonds_or_atoms = bonds + atoms # (N_atom x N_atom)
        bonds_or_atoms.name = 'bonds_or_atoms'

        atom_indeces = T.arange(ones_vec.shape[0]) # 0 to N_atoms - 1 (indeces)
        atom_indeces.name = 'atom_indeces vector'
        ### Subtract previous node attribute contribution
        # Multiply each entry in bonds_or_atoms by the previous atom features for that column
        (old_features_to_sub, updates) = theano.scan(lambda i: T.outer(bonds_or_atoms[:, i], attributes[i, :]), 
            sequences = T.arange(ones_vec.shape[0]))
        old_features_to_sub.name = 'old_features_to_sub'

        ### Add new node attribute contribution
        # Multiply each entry in bonds_or_atoms by the previous atom features for that column
        (new_features_to_add, updates) = theano.scan(lambda i: T.outer(bonds_or_atoms[:, i], new_attributes[i, :]),
            sequences = T.arange(ones_vec.shape[0]))
        new_features_to_add.name = 'new_features_to_add'

        # Update new graph
        new_graph = graph - old_features_to_sub + new_features_to_add
        new_graph.name = 'new_graph'

        return (new_attributes, new_graph)


    def attributes_to_fp_contribution(self, attributes, depth):
        '''Given a 2D tensor of attributes where the first dimension corresponds to a single
        node, this method will apply the output sparsifying (often softmax) function and return
        the contribution to the fingerprint'''
        # Apply output activation function
        output_dot = K.dot(attributes[:, :-1], self.W_output[depth, :, :]) # ignore last attribute (bond flag)
        output_dot.name = 'output_dot'
        output_bias = self.b_output[depth, 0, :]
        output_bias.name = 'output_bias'
        output_activated = keras.activations.softmax(output_dot + output_bias)
        output_activated.name = 'output_activated'
        return output_activated

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'inner_dim' : self.inner_dim,
                  'input_dim': self.input_dim,
                  'depth' : self.depth} 
        # 'init_output' : self.init_output,'init_inner' : self.init_inner, 'activation_inner': tanch, 'activation_inner': tanch,

        base_config = super(GraphFP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[91]:


"""# Are we using a convolutional embedding or a fingerprint representation?
# if type(use_fp) == type(None): # normal mode, use convolution
model = Sequential()
model.add(GraphFP(output_dim = 512, inner_dim = 39, depth = 2, scale_output = 0.01, padding = False, activation_inner = 'tanh'))

print('    model: added GraphFP layer ({} -> {})'.format('mol_graph_tensor', "512"))

model.add(Dense(100, activation = "tanh"))
model.add(Dense(50, activation = "tanh"))
model.add(Dense(1, activation = "softmax"))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.layers"""


# In[370]:


#m = Chem.MolFromSmiles('c1ccccc1')
#original_graph =  molToGraph(m, molecular_attributes = True).dump_as_tensor()
#G = GraphFP(output_dim =512 , inner_dim =39)
#G.build()
#fp = G.get_output_singlesample(original_graph)
# print(fp.eval())


# In[371]:


# print(fp.eval().shape)


# In[ ]:


# depth for 考虑相邻的原子，跟新原子属性

