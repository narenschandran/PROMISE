# This script will convert sequences from a FASTA file into
# index-based tensors, since that is the input format
# that will be used to train Tensorflow models. It's
# unlikely you'll be interested in this script if you're
# merely running the model. In that case, you'll
# probably want to use the <seq2emb.py> script.

import os
import sys
import argparse
projroot   = os.path.join(os.path.dirname(__file__), '..')
prereq_dir = os.path.join(projroot, 'prereq')
run_utils_dir = os.path.join(projroot, 'run-utils')
sys.path.append(projroot)

from promiselib.Utils import savePickle, seq_process

parser = argparse.ArgumentParser(
    prog="Sequence processing (for model training)",
    description="Convert amino acids into indices for use with Tensorflow"
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
    fname = f"{fbase}.seq.pkl"
else:
    fname = args.output_filename


# This is the maximum protein size that will be allowed. Must be
# decided beforehand what the model's maximum allowed protein
# size will be.
MAX_SIZE = 3000
seq_dct = seq_process(fasta_file, MAX_SIZE)

outf = os.path.join(odir, fname)
savePickle(seq_dct, outf)
