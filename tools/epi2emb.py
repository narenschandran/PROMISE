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

parser.add_argument("scan_file")

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


scan_file = args.scan_file
odir = args.output_dir if args.output_dir is not None else os.path.dirname(scan_file)

if not os.path.exists(odir):
    os.makedirs(odir, exist_ok = True)


if args.output_filename is None:
    fext  = scan_file[-3:]
    fbase = os.path.splitext(os.path.basename(scan_file))[0]
    if fext == '.gz':
        fbase = os.path.splitext(fbase)[0]
    fname = f"{fbase}.hlascanemb.pkl"
else:
    fname = args.output_filename

from promiselib.Utils import savePickle, episcan_read_type2
from promiselib.RunUtils import ModelLoad

def epi_emb_get(scan_f, kmer_size = 9):
    mod = ModelLoad()
    epiemb_mod = mod.epiemb_model_get()
    scan_keys, scan_vals = episcan_read_type2(scan_f, mod.max_prot_size, kmer_size)
    tmp = epiemb_mod.predict([scan_vals])
    dat = {scan_keys[i]: tmp[i] for i in range(len(scan_keys))}
    del tmp
    del scan_keys
    del scan_vals
    return dat

KMER_SIZE = 9
epi_emb_dct = epi_emb_get(scan_file, KMER_SIZE)

outf = os.path.join(odir, fname)
savePickle(epi_emb_dct, outf)
