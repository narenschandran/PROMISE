import os
import numpy  as np
import pandas as pd

import tensorflow as tf
import keras

from promiselib.Layers import ConcatenateDLast, ANN, AminoEmbLayer, ExpandDim1, InputFolded_LSTM
from promiselib.Utils import make_uuid
from keras.layers import *


class PROMISE(tf.keras.models.Model):
    def get_config(self):
        return {
            'max_prot_size'    : self.max_prot_size,
            'wsize'            : self.wsize,
            'use_tpm'          : self.use_tpm,
            'seq_embdim'       : self.seq_embdim,
            'seq_lstmdim'      : self.seq_lstmdim,
            'seq_annconf'      : self.seq_annconf,
            'epi_lstmdim'      : self.epi_lstmdim,
            'epi_annconf'      : self.epi_annconf,
            'prob_annconf'     : self.prob_annconf,
        }

    def __init__(self,
                 max_prot_size = 300, wsize = 50,
                 use_tpm     = False,
                 seq_embdim  = 0,
                 seq_annconf = None,
                 seq_lstmdim = 0,
                 epi_lstmdim = 0,
                 epi_annconf = None,
                 prob_annconf = 'sigmoid16_sigmoid1'):
        super().__init__()

        # Useful Operators !!!! DON'T MOVE
        # [1] zero_maker   - Set all values to zero
        # Whenever we don't want to use a specific layer, we just convert
        # the values to zeros using zero_maker. In case there are multiple
        # dimensions, we use reduce_sum later on to produce a single
        # output zero value.
        self.zero_maker   = lambda x : tf.cast(tf.zeros_like(x), tf.float32)
        self.one_maker    = lambda x : tf.cast(tf.ones_like(x), tf.float32)

        # We start off by fixing the maximum protein size
        # that can be processed by the model and also
        # ensure that it can be divided completely by the
        # input window size.
        self.max_prot_size = max_prot_size
        self.wsize         = wsize
        if self.max_prot_size % self.wsize != 0:
            raise ValueError("wsize is not a perfect multiple of max_prot_size")
        self.nwindows = self.max_prot_size // self.wsize

        # Tracking and configuring model components will be used
        # [1] TPM
        self.use_tpm     = use_tpm
        if self.use_tpm:
            print("MODEL INIT: Using [TPM] values")
            self.tpm_procl = tf.math.log1p
        else:
            print("MODEL INIT: Ignoring [TPM] values")
            self.tpm_procl = self.zero_maker



        # [2] Protein sequence
        self.seq_embdim  = seq_embdim if seq_lstmdim > 0 else 0
        self.seq_lstmdim = seq_lstmdim if seq_embdim > 0 else 0
        self.seq_annconf = seq_annconf if self.seq_embdim > 0 else None
        self.use_seq     = (self.seq_embdim > 0) and (self.seq_lstmdim > 0)
        if self.use_seq:
            print("MODEL INIT: Using [sequence] embdding")
            self.seq_layer  = InputFolded_LSTM(
                 size      = self.max_prot_size,
                 wsize     = self.wsize,
                 mask_val  = 0,
                 lstmdim   = self.seq_lstmdim,
                 annconf   = self.seq_annconf,
                 embdim    = self.seq_embdim,
                 nfeatures = 1,
                 add_last_dim = True,
                 name = "seq_inpfold_lstm")
            self.seq_procl  = lambda x: self.seq_layer(x)
        else:
            print("MODEL INIT: Ignoring [sequence] embedding")
            self.seq_procl = lambda x : tf.reduce_sum(self.zero_maker(x), axis = -1, keepdims = True)

        # [3] Epitope scan
        self.epi_lstmdim = epi_lstmdim if epi_lstmdim is not None else 0
        self.use_epi     = self.epi_lstmdim > 0
        self.epi_annconf = epi_annconf
        if self.use_epi:
            # This should produce a fixed-dimension embedding
            # for the epitope scanning input.
            print("MODEL INIT: Using [Epitope scanning] input")
            self.epi_layer  = InputFolded_LSTM(
                 size      = self.max_prot_size,
                 wsize     = self.wsize,
                 mask_val  = -1.0,
                 lstmdim   = self.epi_lstmdim,
                 annconf   = self.epi_annconf,
                 embdim    = 0,
                 nfeatures = 2,
                 add_last_dim = False,
                 name = "seq_inpfold_lstm")

            self.epi_procl  = lambda x: self.epi_layer(x)

        else:
            print("MODEL INIT: Ignoring [Epitope scan]")
            self.epi_procl = lambda x : tf.reduce_sum(tf.reduce_sum(self.zero_maker(x), axis = -1, keepdims = False), axis = -1, keepdims = True)

        # Output wrangling layers
        self.prob_annconf  = prob_annconf
        self.concat        = ConcatenateDLast(name = 'concat_proc')
        self.combn_probl   = ANN(self.prob_annconf, name = 'combn_probl')

        # self.len_procl = lambda x: tf.math.log1p(tf.reduce_sum(tf.cast(x[:,:,1] > (-1.0), tf.float32), axis = 1, keepdims = True))

        self.model_name_set()


    def model_name_set(self):

        conf      = self.get_config()
        keys      = sorted(list(conf.keys()))
        comps     = [str(conf[key]) for key in keys]
        comp_str  = "-".join(comps)
        comp_hash = make_uuid(comp_str)

        c1 = 't' if self.use_tpm else 'x'
        c2 = 's' if self.use_seq else 'x'
        c3 = 'e' if self.use_epi else 'x'

        model_base = 'model-' + ''.join([c1, c2, c3]) + "-"
        self.model_name = model_base + comp_hash

    def call(self, inputs):
        tpm_inp, seq_inp, epi_inp = inputs

        # Each of the model inputs will be processed into
        # a tensor with one dimension to be fed into the
        # final layer.
        # [1] TPM
        tpm_proc  = self.tpm_procl(tpm_inp)

        # [2] Protein sequence embedding
        seq_proc  = self.seq_procl(seq_inp)

        # [3] Epitope scanning
        epi_proc  = self.epi_procl(epi_inp)
        # len_proc  = self.len_procl(epi_inp)

        # The outputs from the previous 3 modules will be
        # integrated and processed by a final ANN.
        out     = self.combn_probl(self.concat([tpm_proc, seq_proc, epi_proc]))
        return out

    def seqemb_model_get(self):
        seqinp     = Input((self.max_prot_size,))
        seqemb     = self.seq_layer(seqinp)
        seqemb_mod = keras.Model(inputs = [seqinp], outputs = seqemb)
        return seqemb_mod

    def epiemb_model_get(self):
        epi_inp  = Input((self.max_prot_size,2))
        epiemb   = self.epi_procl(epi_inp)
        epimod   = keras.Model(inputs = [epi_inp], outputs = epiemb)
        return epimod

    def simple_model_get(self):
        tpm_proc      = Input((1,))
        seq_proc      = Input((self.seq_lstmdim * 2, ))
        epi_proc      = Input((self.epi_lstmdim * 2,))

        proc          = self.concat([tpm_proc, seq_proc, epi_proc])
        out           = self.combn_probl(proc)
        simple_mod    = keras.Model(inputs = [tpm_proc, seq_proc, epi_proc], outputs = out)
        return simple_mod
