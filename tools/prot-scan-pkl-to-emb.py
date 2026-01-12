import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse

projroot = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(projroot)

parser = argparse.ArgumentParser(
    prog       = "hla-scan-embedding",
    description= "Convert TinyHLAnet scans into embeddings"
)

parser.add_argument("scan_dir")

parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    action="store",
    default=None,
    help="Output directory",
)

parser.add_argument(
    "-l",
    "--max_protein_length",
    type=int,
    action="store",
    default=3000,
    help="Output directory",
)



args = parser.parse_args()
if args.output_dir is not None:
    odir = args.output_dir
else:
    odir = os.path.join(os.path.dirname(args.scan_dir), "prot-scan-embeddings")

if not os.path.exists(odir):
    os.makedirs(odir, exist_ok = True)


def readPickle(file):
    fcon = open(file, "rb")
    obj = pickle.load(fcon)
    fcon.close()
    return obj

def savePickle(obj, file):
    fcon = open(file, "wb")
    pickle.dump(obj, fcon)
    fcon.close()
    return file

def allele_clean(y):
    x = os.path.splitext(os.path.basename(y))[0].split('-')
    return x[0] + '-' + x[1] + '*' + x[2] + ':' + x[3]



hla_pkl_dir = args.scan_dir
hla_pkls   = {allele_clean(f) : os.path.join(hla_pkl_dir, f) for f in os.listdir(hla_pkl_dir) if f[-4:] == ".pkl"}

from promiselib.RunUtils  import ModelLoad
from promiselib.Utils     import md5sum_compute, pad_arr
mod = ModelLoad()
epimod = mod.epimodel_get()

for al, f in hla_pkls.items():
    outfname  = os.path.splitext(os.path.basename(f))[0] + ".emb.pkl"
    outf      = os.path.join(odir, outfname)
    outf_lock = outf + ".md5"
    print(outf)

    if not os.path.exists(outf_lock):
        prots = []
        vals  = []
        for k, v in readPickle(f).items():
            if len(v) <= args.max_protein_length:
                prots.append(k)
                values = pad_arr([float(vv) for vv in v], args.max_protein_length)
                vals.append(values)

        pred = epimod.predict([np.asarray(vals)])
        res  = {prots[i] : pred[i] for i in range(len(prots))}
        savePickle(res, outf)

        chksum    = md5sum_compute(outf)
        lock_fcon = open(outf_lock, 'w')
        print(lock_fcon)
        lock_fcon.write(chksum)
        lock_fcon.close()
