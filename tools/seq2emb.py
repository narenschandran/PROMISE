# This script will convert sequences from a FASTA file into
# fixed-dimensional embeddings. This will be used to speed
# up model execution.
import os
import sys
import argparse
projroot   = os.path.join(os.path.dirname(__file__), '..')
prereq_dir = os.path.join(projroot, 'prereq')
run_utils_dir = os.path.join(projroot, 'run-utils')
sys.path.append(projroot)

parser = argparse.ArgumentParser(
    prog="Sequence processing (for model execution)",
    description="Convert amino acids into embeddings"
)

parser.add_argument("fasta_file")

parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    action="store",
    default=None,
    help="Output directory",
)

parser.add_argument(
    "-f",
    "--output_filename",
    type=str,
    action="store",
    default=None,
    help="Output file name (without directory)",
)

args = parser.parse_args()


fasta_file = args.fasta_file
odir = args.output_dir if args.output_dir is not None else os.path.dirname(fasta_file)

if not os.path.exists(odir):
    os.makedirs(odir, exist_ok = True)


if args.output_filename is None:
    fext  = fasta_file[-3:]
    fbase = os.path.splitext(os.path.basename(fasta_file))[0]
    if fext == '.gz':
        fbase = os.path.splitext(fbase)[0]
    fname = f"{fbase}.seqemb.pkl"
else:
    fname = args.output_filename

from promiselib.Utils import savePickle, seq_process_type2
from promiselib.RunUtils import ModelLoad


def seqemb_get(fa_f):
    mod = ModelLoad()
    seqemb_mod = mod.seqemb_model_get()
    seq_keys, seq_inds = seq_process_type2(fa_f, mod.max_prot_size)
    tmp = seqemb_mod.predict([seq_inds])
    dat = {seq_keys[i]: tmp[i] for i in range(len(seq_keys))}
    del tmp
    del seq_keys
    del seq_inds
    return dat


seqemb_dct = seqemb_get(fasta_file)

outf = os.path.join(odir, fname)
savePickle(seqemb_dct, outf)
