import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse
import re
projroot   = os.path.join(os.path.dirname(__file__), '..')
prereq_dir = os.path.join(projroot, 'prereq')
run_utils_dir = os.path.join(projroot, 'run-utils')
sys.path.append(projroot)

from promiselib.Utils import readPickle, savePickle

parser = argparse.ArgumentParser(
    prog="PROMISE",
    description="PROteome MIning for Source of Epitopes"
)

parser.add_argument("scan_dir")

parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Force re-run even if file already exists"
)

parser.add_argument(
    "-S",
    "--seq_embeddings",
    type=str,
    action="store",
    default=None
)

parser.add_argument(
    "-H",
    "--hla_embeddings_dir",
    type=str,
    action="store",
    default=None
)

parser.add_argument(
    "-a",
    "--all_genes",
    action="store_true",
)

args = parser.parse_args()

sample_dir = args.scan_dir

tpm_f  = os.path.join(sample_dir, 'tpm.tsv')
al_f   = os.path.join(sample_dir, 'alleles.txt')

outf = os.path.join(sample_dir, 'pred.tsv')
if os.path.exists(outf) and (not args.force):
    print("Found an existing prediction at:")
    print(outf)
    print("If you want to force a re-run, use the --force option")
    print("")
    sys.exit(0)

with open(al_f, "r") as fcon:
    alleles = [line.strip() for line in  fcon.readlines()]

with open(os.path.join(projroot, 'prereq', 'allowed-genes.txt'), 'r') as fcon:
    allowed_genes = [line.strip() for line in fcon]



from promiselib.RunUtils import ModelLoad

mod     = ModelLoad()
smod    = mod.simple_model_get()

seq_emb = readPickle(args.seq_embeddings)

datf = pd.read_csv(tpm_f, sep = '\t')

if not args.all_genes:
    datf = datf[datf.gene.isin(allowed_genes)].reset_index()

with open(al_f, 'r') as fcon:
    alleles = [line.strip() for line in fcon]

tpm_dct = dict(zip(datf.gene, datf.tpm))
genes = [gene for gene in tpm_dct.keys() if gene in seq_emb]
genes.sort()

pred_lst = []
for allele in alleles:
    allele_str  = re.sub("[*:]", "-", allele)
    hla_emb_f   = os.path.join(args.hla_embeddings_dir, f"{allele_str}.hlascanemb.pkl")
    hla_emb     = readPickle(hla_emb_f)
    tpm_lst     = []
    seq_lst     = []
    hla_lst     = []
    for gene in genes:
        tpm_lst.append(tpm_dct[gene])
        seq_lst.append(seq_emb[gene])
        hla_lst.append(hla_emb[gene])

    tpm_inp  = np.asarray(tpm_lst)
    seq_inp  = np.asarray(seq_lst)
    hla_inp  = np.asarray(hla_lst)
    pred     = smod.predict([np.log1p(tpm_inp), seq_inp, hla_inp])
    pred_lst.append(np.squeeze(pred))

pred_datf = pd.DataFrame(np.transpose(np.asarray(pred_lst), (1, 0)))
pred_datf.columns = [f"al{i+1}:{al}" for i, al in enumerate(alleles)]


sdatf = pd.concat([datf[datf.gene == gene] for gene in genes])
res = pd.concat([sdatf, pred_datf], axis = 1)

if 'index' in res.columns:
    res = res.drop('index', axis = 1)
outf = os.path.join(sample_dir, 'pred.tsv')
res.to_csv(outf, sep = '\t', index = False)
