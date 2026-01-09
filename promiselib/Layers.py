import os
import re
import pandas as pd

import tensorflow as tf
import keras
from keras.layers import Bidirectional, Dense, Embedding, Flatten, LSTM, Conv1D, GRU, TimeDistributed, Masking

from promiselib.Utils import _aa1lst as aa1lst

from promiselib.Regularizers import (
    UnitNormMaxRegularizer,
    UnitNormRegularizer,
)

def AminoEmbLayer(embdim, name, reg = 'maxl2norm', mask_zero = False, **kwargs):
    '''Wrapper function to produce amino acid embedding layer

    Args:
        embdim : Embedding dimension
        name   : Embedding layer name
        reg    : Embedding regularizer. Can be a conventional Keras regularizer
                 or one of 'l2norm' or 'maxl2norm', the latter two of which
                 are defined by our library
    '''
    if reg == 'l2norm':
        embreg = UnitNormRegularizer()
    elif reg == 'maxl2norm':
        embreg = UnitNormMaxRegularizer()
    else:
        embreg = None
    return Embedding(input_dim=len(aa1lst), output_dim=embdim,
                     name=name, embeddings_regularizer=embreg,
                     mask_zero = mask_zero, **kwargs)



#------------------------------------------------------------------------------#
#                                  Operations                                  #
#------------------------------------------------------------------------------#
# The layers defined in this section do standard operations such as summming a #
# matrix, dot product and concatenation. Although these can be defined as      #
# functions, there were certain issues (at least on Tensorflow 2.18) when      #
# trying to contruct a secondary model based on another model's components     #
# wherein outputs from certain model layers could not be directly used         #
# for operations such as dot product.                                          #
#------------------------------------------------------------------------------#
@keras.saving.register_keras_serializable("promiselib")
class ExpandDim1(keras.layers.Layer):
    '''Expand last dimension'''
    def __init__(self, name):
        super().__init__(name = name)

    def call(self, inputs):
        return tf.expand_dims(inputs, 1)

    def get_config(self):
        return {'name' : self.name}

@keras.saving.register_keras_serializable("promiselib")
class ExpandDimLast(keras.layers.Layer):
    '''Expand last dimension'''
    def __init__(self, name):
        super().__init__(name = name)

    def call(self, inputs):
        return tf.expand_dims(inputs, -1)

    def get_config(self):
        return {'name' : self.name}



@keras.saving.register_keras_serializable("promiselib")
class ConcatenateD1(keras.layers.Layer):
    '''Concatenate at the first dimension'''
    def __init__(self, name):
        super().__init__(name = name)

    def call(self, inputs):
        return tf.concat(inputs, axis = 1)

    def get_config(self):
        return {'name' : self.name}

@keras.saving.register_keras_serializable("promiselib")
class ConcatenateDLast(keras.layers.Layer):
    '''Concatenate at the last dimension'''
    def __init__(self, name):
        super().__init__(name = name)

    def call(self, inputs):
        return tf.concat(inputs, axis = -1)

    def get_config(self):
        return {'name' : self.name}


@keras.saving.register_keras_serializable("promiselib")
class SplitVec(keras.layers.Layer):
    def __init__(self, name, size, wsize, mask = tf.identity):
        super().__init__(name = name)
        self.size   = size
        self.wsize  = wsize
        self.ranges = []
        self.mask   = mask
        for i in range(int(self.size / self.wsize)):
            self.ranges.append((i * self.wsize, (i+1) * self.wsize))

    def call(self, inputs):
        return [self.mask(inputs[:,i:j,...]) for i,j in self.ranges]

    def get_config(self):
        return {'name' : self.name, 'size' : self.size, 'wsize' : self.wsize, 'mask' : self.mask}

#------------------------------------------------------------------------------#
#                      ANN-related functions and classes                       #
#------------------------------------------------------------------------------#
#  Since our models use ANNs in various places and since hyperparameter        #
#  optimization is an important task during our model tuning, we define some   #
#  functions to facilitate the construction of ANNs.                           #
#------------------------------------------------------------------------------#
#  NOTE: This code should be the same as what was used in TinyHLAnet           #
#------------------------------------------------------------------------------#


def parse_dense_conf(x):
    '''Parse dense layer configuration string into a usable dictionary

    Format of the configuration string:
        <actv><nodes><kern-and-bias-reg:optional>:<activity-reg:optional>

        Example: 'sigmoid16:l2' constructs a Dense layer with 16 nodes
        that uses a sigmoid activation function with an l2-norm activity
        regularizer but no regularization for the kernal and bias.

    Args:
        x: Configuration string

    Returns:
        Dense layer with the required configuration

    '''

    # Parse the activity regularization
    sp = x.split(":")
    if len(sp) == 1:
        areg = None
    elif len(sp) == 2:
        areg = sp[1]
    else:
        raise ValueError("Invalid configuration")

    s = sp[0]
    i = 0
    slen = len(s)

    # Parse until the first number to get the activation function
    while i < slen:
        if not s[i].isalpha():
            break
        i = i + 1
    actv = s[:i]

    # Parse till the end of the number to get the number of nodes
    j = i
    while j < slen:
        if s[j].isalpha():
            break
        j = j + 1
    units = int(s[i:j])

    # If anything else is remaining, it is the regularizer
    reg = s[j:] if j < slen else None

    # Parsing custom regularizers
    if areg == "maxl2norm":
        areg = UnitNormMaxRegularizer()
    if areg == "l2norm":
        areg = UnitNormRegularizer()

    return {
        'units'                : units,
        'activation'           : actv,
        'kernel_regularizer'   : reg,
        'bias_regularizer'     : reg,
        'activity_regularizer' : areg,
    }

def dense_from_conf(x):
    '''Construct Dense layer from a configuration string

    Args:
        x: Configuration string

    Returns:
        Dense layer with the required configuration

    '''
    conf = parse_dense_conf(x)
    return Dense(**conf)

def ANN(conf, name, flatten=False, clip=None):
    '''Construct an ANN from a configuration string

    ANN Conf format:
        <layer1conf>_<layer2conf>..."

    For the layer configuration, refer to the dense_from_conf function
    '''
    mod = tf.keras.Sequential(name=name)
    if flatten:
        mod.add(Flatten())

    conf_sp = conf.split("_")
    for layer_conf in conf_sp:
        mod.add(dense_from_conf(layer_conf))

    if clip is not None:
        lower, upper = clip
        mod.add(ClipValues(f"{name}clip", lower, upper))
    return mod


#------------------------------------------------------------------------------#
#                               Stand-in classes                               #
#------------------------------------------------------------------------------#

@keras.saving.register_keras_serializable("promiselib")
class InputFolded_LSTM(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 wsize,
                 mask_val,
                 lstmdim   = 16,
                 annconf   = 'gelu8_gelu4',
                 embdim    = 0,
                 nfeatures = 1,
                 add_last_dim = False,
                 name = "InpFoldLSTM"):
        super().__init__(name = name)

        self.size      = size
        self.wsize     = wsize
        self.lstmdim   = lstmdim

        self.embdim    = embdim
        if self.embdim > 0:
            self.emb_p = True
            self.embl  = AminoEmbLayer(self.embdim, name = f"{name}_embl")
        else:
            self.emb_p = False
            self.embl  = tf.identity

        self.annconf   = annconf
        self.flatten   = Flatten()
        if self.annconf is not None:
            self.ann_p = True
            self.ann   = TimeDistributed(ANN(self.annconf, name = f"{name}_ann", flatten = True))
        else:
            self.ann_p = False
            self.ann   = tf.identity


        self.nfeatures = nfeatures
        if self.ann_p:
            self.last_size = int(re.sub('[A-Za-z]', '', self.annconf.split('_')[-1]))
        else:
            self.last_size = self.wsize * self.nfeatures


        self.mask_val  = mask_val

        # We will put hide_val in the final vectors if we want to
        # mask it
        if self.mask_val != (-10000):
            self.hide_val = tf.cast(-10000, tf.float32)
        else:
            self.hide_val = tf.cast(-20000, tf.float32)
        self.mask      = Masking(mask_value = self.hide_val)

        # Internal layers
        self.lstm      = Bidirectional(LSTM(self.lstmdim))

        # Utility layers
        self.split_vec = SplitVec(f"{self.name}_split_vec", self.size, self.wsize)
        self.exp_dim   = ExpandDim1(f"{self.name}_exp_dim")
        self.concat    = ConcatenateD1(f"{self.name}_concat")
        self.floatit   = lambda x: tf.cast(x, tf.float32)

        # Check which elements of a tensor match a specific value
        # and create an output tensor of the same shape, but
        # with 1-values wherever the value is 'v' and zero otherwise.
        self.valuecheck = lambda x, v: self.floatit(self.floatit(x) == v)

        self.is_one     = lambda x: self.valuecheck(x, 1.0)
        self.is_not_one = lambda x: self.valuecheck(self.valuecheck(x, 1.0), 0.0)

        self.maskcheck  = lambda x: self.valuecheck(x, self.mask_val)

        self.add_lst_dim = add_last_dim
        if add_last_dim:
            self.exp_last = ExpandDimLast(f"{self.name}_exp_dim_last")
        else:
            self.exp_last = tf.identity


    def call(self, inputs):
        # Expects a tensor of ndim=3: (# samples, sequence index, nfeatures)
        # If nfeatures=1, and the last dimension does not exist (like when
        # embedding needs to be done), it can be expanded in the next step
        # if add_last_dim=True is specified.
        inp = inputs

        # Step 1: Split the index tensor
        inpsp = self.split_vec(self.exp_last(inputs))

        # Step 2: Figure out which entries in each
        #         tensor piece has the mask value
        inpsp2 = [self.maskcheck(sp) for sp in inpsp]

        # Step 3: Check the mean of each tensor piece
        #         after the previous mask check. If it
        #         is one, then the output after processing
        #         the tensor piece must be masked/hidden.
        inpsp3 = [tf.reduce_mean(sp, axis = 1, keepdims = True) for sp in inpsp2]
        
        # This has the tensor that will be used to find the mask regions
        hide_v    =  tf.tile(self.concat([self.is_one(sp) for sp in inpsp3]),
                             [1, 1, int(self.last_size / self.nfeatures)])

        # This has the tensor that will be used to find the non-mask regions
        nonhide_v =  tf.tile(self.concat([self.is_not_one(sp) for sp in inpsp3]),
                             [1, 1, int(self.last_size / self.nfeatures)])


        emb   = self.embl(inp)
        emb_v = self.ann(self.concat([self.exp_dim(self.flatten(sp)) for sp in self.split_vec(emb)]))

       
        # Keep only the non-masked values. Zero out the masked values
        contr_emb = emb_v * nonhide_v

        # Fill in the masking value in areas other
        # than the non-masking region.
        hide_emb  = (tf.zeros_like(emb_v) + (self.hide_val)) * hide_v


        # Merge the information from the last two tensors to
        # get the non-masked region with last bit having the
        # the masked values in the appropriate place
        final_emb = self.mask(contr_emb + hide_emb)

        return self.lstm(final_emb)

    def get_config(self):
        return {
            'size'         : self.size,
            'wsize'        : self.wsize,
            'embdim'       : self.embdim,
            'mask_val'     : self.mask_val,
            'lstmdim'      : self.lstmdim,
            'annconf'      : self.annconf,
            'embdim'       : self.embdim,
            'nfeatures'    : self.nfeatures,
            'add_last_dim' : self.add_last_dim,
            'name'         : self.name,
        }

