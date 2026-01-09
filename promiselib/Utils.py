import pickle
from hashlib import shake_256, md5

import gzip
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def readPickle(file):
    """
    Convenient wrapper for reading in .pkl files.
    """
    fcon = open(file, "rb")
    obj = pickle.load(fcon)
    fcon.close()
    return obj

def savePickle(obj, file):
    """
    Convenient wrapper to write out .pkl files.
    """
    fcon = open(file, "wb")
    pickle.dump(obj, fcon)
    fcon.close()
    return file

def md5sum_compute(f):
    fcon = open(f, 'rb')
    chksum = md5(fcon.read()).hexdigest()
    fcon.close()
    return chksum

def make_uuid(s):
    """
    This function is used to generate a unique identification code
    for each model instance based on the hyperparameters used for the model.
    """
    if not isinstance(s, str):
        raise ValueError("Wrong type")
    return shake_256(s.encode('utf-8')).hexdigest(4)

def writeDict(dct, f):
    """
    Write out dictionaries into plain text format
    """
    with open(f, 'w') as fcon:
        for k, v in dct.items():
            fcon.write(f"{k}\t{v}\n")
    return f

def readDict(f):
    """
    Read dictionaries written out by writeDict
    """
    dct = {}
    with open(f, 'r') as fcon:
        for line in fcon:
            k, v = line.strip().split('\t')
            dct[k] = v
    return dct

def layernames(tfobj):
    """
    Get names of layers in a given object
    """
    return [obj.name for obj in tfobj.layers]


def _layer_get(tfobj, name):
    """
    Subset a single layer by name
    """
    ind = layernames(tfobj).index(name)
    return tfobj.layers[ind]


def layer_get(tfobj, names):
    """
    Subset one or more layers by name. If a single layer is requested (if
    'names' is a single input), then provides the layer directly. Otherwise,
    provides the layers in dictionary format.
    """
    strtype = isinstance(names, str)
    if strtype:
        return _layer_get(tfobj, names)
    else:
        return [_layer_get(tfobj, name) for name in names]


#------------------------------------------------------------------------------#
#                            Performance assessment                            #
#------------------------------------------------------------------------------#
def subset_available(ac, pr, ex_val):
    """
    Given actual values (ac) and predicted values (pr), it subsets only
    cases where the actual value is known (i.e. if the actual value is not
    equal to the exclusion value (ex)).
    """
    def to1d(x):
        if x.ndim > 2:
            raise ValueError("Too many dimensions")
        elif x.ndim == 2:
            return np.squeeze(x)
        elif x.ndim == 1:
            return x
        else:
            raise Value("Catastrophic error")

    pr1 = to1d(pr)
    ac1 = to1d(ac)
    l = ac1 != ex_val
    if not np.any(l):
        return [None, None]
    else:
        return [ac1[l], pr1[l]]


def MSE(ac, pr):
    """
    Convenient function that computes mean-squared error only for datapoints
    where the true value is known.
    """
    ac1, pr1 = subset_available(ac, pr, -1.0)
    if ac1 is None:
        return np.nan
    return np.round(np.sqrt(np.mean(np.square(pr1 - ac1))), 3)


def BCE(ac, pr):
    """
    Convenient function that computes binary crossentropy only for datapoints
    where the true value is known.
    """

    ac1, pr1 = subset_available(ac, pr, -1.0)
    if ac1 is None:
        return np.nan
    pr1 = np.clip(pr1, 0.0001, 0.9999)  # Prevent issues with log and zero values
    tmp0 = -1.0 * ((ac1 * np.log(pr1)) + ((1 - ac1) * np.log(1 - pr1)))
    return np.round(np.mean(tmp0), 3)


def AUC(ac, pr):
    """
    Convenient function that computes AUC only for datapoints where the true
    value is known.
    """
    ac1, pr1 = subset_available(ac, pr, -1.0)
    if ac1 is None:
        return np.nan
    val = roc_auc_score(ac1, pr1)
    return np.round(val, 3)


def PRAUC(ac, pr):
    """
    Convenient function that computes PRAUC only for datapoints where the
    true value is known.
    """
    ac1, pr1 = subset_available(ac, pr, -1.0)
    if ac1 is None:
        return np.nan
    val = average_precision_score(ac1, pr1)
    return np.round(val, 3)


#----------------------------------------------------------#
# The following chunk is taken from my nbslpy library as   #
# is. Only the functions necessary for converting sequence #
# data into appropriate Tensorflow inputs are placed here  #
#----------------------------------------------------------#


# The order of amino acids in _aa1lst will be used to give
# it a unique index. It is organized so that a symbol for
# unknown/missing amino acids is defined, followed by the
# standard 20 amino acids. This allows for non-standard
# amino acids to be added later without distrubting the
# original order. The unknown/missing symbol is given first
# so that it is convenient for masking with Tensorflow's
# Embedding layer.
_aa1lst = [
    # 1 unknown
    "X",

    # 20 standard Amino Acids
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y",
]

_aa1to3dct = {
    "X": "UNK",

    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

def SingleVectorize(fn, data, **kwargs):
    """
    This function is used to modify function calls
    depending on whether the main input is a list
    or a single object. In case it is a single
    object, the function is applied to it directly.
    Otherwise, the function is applied to each
    element of the list.
    """
    if isinstance(data, list) or isinstance(data, pd.core.series.Series):
        return [fn(x, **kwargs) for x in data]
    else:
        return fn(data, **kwargs)


def toaa1(aa):
    """
    User facing function for getting single-letter amino acid code
    """
    def _toaa1(aa):
        if len(aa) == 1:
            aa1 = aa if aa in _aa1to3dct.keys() else "X"
        else:
            raise ValueError(
                "Input [%s] is not a one-letter code" % (aa)
            )
        return aa1
    return SingleVectorize(_toaa1, aa)

def aaind(aa):
    def _aaind(aa):
        return _aa1lst.index(toaa1(aa))
    return np.asarray(SingleVectorize(_aaind, aa))


def seq2ind(seq):
    def _seq2ind(seq):
        if not isinstance(seq, str):
            raise ValueError("Input must be a sequence string")
        seqlst = list(seq)
        return aaind(seqlst)

    return np.asarray(SingleVectorize(_seq2ind, seq))

#----------------------------------------------------------#

def fasta_reader(f):
    f_ext = f[-3:]
    fcon = gzip.open(f, 'rt') if f_ext  == '.gz' else open(f, 'r')
    hds  = []
    seqs = []
    for raw_line in fcon:
        line = raw_line.strip()
        if line[0] == ">":
            curr_hd = line[1:]
            hds.append(curr_hd)
            seqs.append("")
        else:
            seqs[-1] = seqs[-1] + line
    fcon.close()
    return pd.DataFrame({'id': hds, 'sequence': seqs})

def seq_process(fasta_file, MAX_SIZE):
    seq_df = fasta_reader(fasta_file)
    seq_dct = {row.id: seq2ind(pad_seq(row.sequence, MAX_SIZE)) for row in seq_df.itertuples() if len(row.sequence) <= MAX_SIZE}
    return seq_dct

def seq_process_type2(fasta_file, MAX_SIZE):
    seq_df = fasta_reader(fasta_file)
    keys = []
    vals = []
    for row in seq_df.itertuples():
        if len(row.sequence) <= MAX_SIZE:
            keys.append(row.id)
            vals.append(seq2ind(pad_seq(row.sequence, MAX_SIZE)))
    return keys, np.asarray(vals)

def pad_seq(x, size):
    slen = len(x)
    if slen == size:
        return x
    elif slen < size:
        nmiss = size - slen
        return x + ("X" * nmiss)
    else:
        raise ValueError("Sequence too large")

#---------------------------------#


def split_num(x):
    return [float(y) for y in x.split(",")]


def pad_arr(x, size):
    alen = len(x)
    if alen == size:
        return x
    elif alen < size:
        nmiss = size - alen
        return x + ([-1.0] * nmiss)
    else:
        raise ValueError("Array too long")


def episcan_read(f, max_prot_size, kmer_size):
    dat = pd.read_csv(f, sep = '\t')
    dct = {}
    max_windows = max_prot_size - kmer_size + 1
    for p, v1, v2, in zip(dat.protein, dat.bindprob, dat.bindaff):
        sv1 = split_num(v1)
        if len(sv1) <= max_windows:
            sv2 = split_num(v2)
            val = np.asarray([sv1, sv2])
            dct[p] = np.transpose(val, (1, 0))
    return dct


def episcan_read_type2(f, max_prot_size, kmer_size):
    """
    Gives padded windows for use with the model directly.
    Padding is not done in the other function to save
    space.
    """
    def padfn(x):
        return pad_arr(x, max_prot_size)

    dat = pd.read_csv(f, sep = '\t')
    dct = {}
    max_windows = max_prot_size - kmer_size + 1
    keys = []
    vals = []
    for p, v1, v2, in zip(dat.protein, dat.bindprob, dat.bindaff):
        sv1 = split_num(v1)
        if len(sv1) <= max_windows:
            sv2 = split_num(v2)
            val = np.asarray([padfn(sv1), padfn(sv2)])
            keys.append(p)
            vals.append(np.transpose(val, (1, 0)))
    return keys, np.asarray(vals)

