# This script will convert sequences from a FASTA file into
# fixed-dimensional embeddings. This will be used to speed
# up model execution. This accepts a directory as input
# so as to prevent delays that happen with initializing
# the model for each input.
import os
import sys
import re
import argparse
projroot   = os.path.join(os.path.dirname(__file__), '..')
prereq_dir = os.path.join(projroot, 'prereq')
run_utils_dir = os.path.join(projroot, 'run-utils')
sys.path.append(projroot)

parser = argparse.ArgumentParser(
    prog="Sequence processing (for model execution)",
    description="Convert amino acids into embeddings"
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
    "-a",
    "--allele-list",
    type=str,
    action="store",
    default=None,
    help="Process only for these allele"
)


args = parser.parse_args()


scan_dir = args.scan_dir
odir = args.output_dir if args.output_dir is not None else os.path.join(os.path.dirname(scan_dir), 'prot-scan-emb')

if not os.path.exists(odir):
    os.makedirs(odir, exist_ok = True)


load_in_libs = False

def allele_clean(f):
    x = f.split('-')
    return x[0] + '-' + x[1] + '*' + x[2] + ':' + x[3]

    
fs = {}
for f in os.listdir(scan_dir):
    if f[-8:] == '.scan.gz':
        allele = allele_clean(f[:-8])
        fs[allele] = os.path.join(scan_dir, f)

if args.allele_list is not None:
    with open(args.allele_list, 'r') as fcon:
        allele_list = [line.strip() for line in fcon]
else:
    allele_list = None

KMER_SIZE = 9
for allele, scan_f in fs.items():
    if allele_list is not None:
        proceed = allele in allele_list
    else:
        proceed = True
    if proceed:
        allele_str = re.sub("[*:]", "-", allele)
        outf = os.path.join(odir, f"{allele_str}.hlascanemb.pkl")
        if not os.path.exists(outf):
            if not load_in_libs:
                from promiselib.Utils import savePickle, episcan_read_type2
                from promiselib.RunUtils import ModelLoad
                mod = ModelLoad()
                epiemb_mod = mod.epiemb_model_get()
                load_in_libs = True
            print(f"Generating embeddings for: {allele}")
            scan_keys, scan_vals = episcan_read_type2(scan_f, mod.max_prot_size, KMER_SIZE)
            tmp = epiemb_mod.predict([scan_vals])
            dat = {scan_keys[i]: tmp[i] for i in range(len(scan_keys))}
            del tmp
            del scan_keys
            del scan_vals
            savePickle(dat, outf)
            del dat
