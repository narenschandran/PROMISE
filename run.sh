err_echo() {
    echo "$1" >&2
}

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJROOT="${SCRIPT_DIR}"

#-----------------------------------------------------------#
# The following paths must be set by the user before this   #
# tool can be used:                                         #
#-----------------------------------------------------------#
# TINYHLANET_DIR : Location of the TinyHLAnet tool. This    #
#                  will be used to find the location of     #
#                  the proteome scanning tools.             #
#-----------------------------------------------------------#
# HLA_SCAN_DIR   : This will be the location where the      #
#                  predictions from the TinyHLAnet tool for #
#                  the reference proteome will be stored    #
#-----------------------------------------------------------#
TINYHLANET_DIR="${PROJROOT}/bin/TinyHLAnet"
if [ -z "$TINYHLANET_DIR" ]; then
    err_echo "Please set the [TINYHLANET_DIR] variable before running the script."
    QUITPRG=1
fi

PSCAN="${TINYHLANET_DIR}/tinyhlanet-scan"
if [ ! -f "$PSCAN" ]; then
    err_echo "Unable to find the proteome scanning tool at: [$PSCAN]"
    QUITPRG=2
fi

HLA_SCAN_DIR="${PROJROOT}/data/hla-scan"
if [ -z "$HLA_SCAN_DIR" ]; then
    err_echo "Please set the [HLA_SCAN_DIR] variable before running the script."
    QUITPRG=3
fi

HLA_RES_DIR="${HLA_SCAN_DIR}/prot-scan"


#----------------------------------------------------------#
#          Setting up paths for later convenience          #
#----------------------------------------------------------#
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJROOT="${SCRIPT_DIR}"
PREREQ_DIR="${PROJROOT}/prereq"
TOOLS_DIR="${PROJROOT}/tools"
DATA_DIR="${PROJROOT}/data"
HLA_CACHE_DIR="${DATA_DIR}/hlascan-embcache"

ENVDIR="${TINYHLANET_DIR}/.env/deephlaffy"
if [ -d "$ENVDIR" ]; then
    PY="${ENVDIR}/bin/python3"
    ENVACTV="${ENVDIR}/bin/activate"
else
    PY="python3"
fi

# Remove this if you want Tensorflow warnings back
export TF_CPP_MIN_LOG_LEVEL=3


# This file should have been provided with the tool
REF_FASTA="${PREREQ_DIR}/human-proteome-canonical.fa.gz"
if [ ! -f "$REF_FASTA" ]; then
    err_echo "Unable to find the reference proteome file at: [$REF_FASTA]"
    QUITPRG=4
fi

HLA_EMB_PY="${TOOLS_DIR}/epi2emb.py"
if [ ! -f "$HLA_EMB_PY" ]; then
    err_echo "Unable to find the Proteome Scan embedder at: [$HLA_EMB_PY]"
    QUITPRG=5
fi


HLA_EMBDIR_PY="${TOOLS_DIR}/epidir2emb.py"
if [ ! -f "$HLA_EMBDIR_PY" ]; then
    err_echo "Unable to find the Proteome Scan embedder (multi-file version) at: [$HLA_EMBDIR_PY]"
    QUITPRG=6
fi


SEQ_EMB_PY="${TOOLS_DIR}/seq2emb.py"
if [ ! -f "$SEQ_EMB_PY" ]; then
    err_echo "Unable to find the Sequence embedder  at: [$SEQ_EMB_PY]"
    QUITPRG=7
fi


MODEL_RUN="${TOOLS_DIR}/model-run.py"
if [ ! -f "$MODEL_RUN" ]; then
    err_echo "Unable to find the Model runfile converter at: [$MAKE_PSCAN_EMB]"
    QUITPRG=8
fi

EPI_EXTRACT="${TOOLS_DIR}/epitopes-extract.R"
if [ ! -f "$EPI_EXTRACT" ]; then
    err_echo "Unable to find the Epitope Extracter at: [$EPI_EXTRACT]"
    QUITPRG=9
fi


EPI_FIN="${TOOLS_DIR}/epitopes-consolidate.R"
if [ ! -f "$EPI_FIN" ]; then
    err_echo "Unable to find the Epitope Consolidater at: [$EPI_FIN]"
    QUITPRG=10
fi

MTECHI_TPM="${PREREQ_DIR}/mTEChi-protein-coding-mean-tpm.tsv"
if [ ! -f "$MTECHI_TPM" ]; then
    err_echo "Unable to find the mTEChi profile at: [${MTECHI_TPM}]"
    QUITPRG=10
fi



if [ ! -z "$QUITPRG" ]; then
    echo "$QUITPRG"
    kill -9 $$
fi


#----------------------------------------------------------#


#-----------------------------------------------------------#
#                     Useful functions                      #
#-----------------------------------------------------------#
err_echo() {
    echo "$1" >&2
}

#' Generate file-system compatible names for HLA alleles
al_clean() {
    echo "$1" | sed 's/[*-:]/-/g'
}



sample_dir="$1"
if [ -z "$sample_dir" ]; then
    err_echo "No input directory provided. Exiting program..."
    kill -9 $$
elif [ ! -d "$sample_dir" ]; then
    echo "The input [$sample_dir] is not a valid directory"
    kill -9 $$
fi

if [ ! -z "$2" ]; then
    sample_id="$2"
else
    sample_id=$(basename "$sample_dir")
fi

al_lst="${sample_dir}/alleles.txt"
if [ ! -f "$al_lst" ]; then
    err_echo "The input directory [$sample_dir] does not contain an [alleles.txt] file"
    QUITPRG=7
fi

tpm_f="${sample_dir}/tpm.tsv"
if [ ! -f "$tpm_f" ]; then
    err_echo "The input directory [$sample_dir] does not contain a [tpm.tsv] file"
    QUITPRG=7
fi

if [ ! -z "$QUITPRG" ]; then
    echo "$QUITPRG"
    err_echo "Please fix the errors and try again..."
    kill -9 $$
fi



SEQ_CACHE_FNAME=$(basename "$REF_FASTA" '.fa.gz')'.seqemb.pkl'
SEQ_CACHE="${DATA_DIR}/${SEQ_CACHE_FNAME}"

if [ ! -z "$ENVACTV" ]; then
    source "$ENVACTV"
fi


#---------------------------------------------------------#
# Generate the canonical proteome sequence embeddings if  #
# they don't already exist.                               #
#---------------------------------------------------------#
if [ ! -f "$SEQ_CACHE" ]; then
    "$PY" "$SEQ_EMB_PY" -o "$DATA_DIR" -f "$SEQ_CACHE_FNAME" "$REF_FASTA"
fi

#---------------------------------------------------------#
# Execute TinyHLAnet scan of the reference proteome, and  #
# cache results to prevent needless computation.          #
#---------------------------------------------------------#
MISSING_ALLELES=""
for al in $(cat "$al_lst"); do
    al_str=$(al_clean "$al")
    al_f="${HLA_RES_DIR}/${al_str}.md5"
    if [ ! -f "$al_f" ]; then
        if [ "$MISSING_ALLELES" = "" ]; then
            MISSING_ALLELES="$al"
        else
            MISSING_ALLELES="${MISSING_ALLELES} ${al}"
        fi
    fi
done


if [ ! -z "$MISSING_ALLELES" ]; then
    "$PSCAN" -a 0.2 -b 0.5 -l <(echo "$MISSING_ALLELES" | sed 's/ /\n/g') -o "$HLA_SCAN_DIR" "$REF_FASTA" 
fi
"$PY" "$HLA_EMBDIR_PY" -o "${HLA_CACHE_DIR}" "$HLA_RES_DIR" -a "$al_lst"


"$PY" "$MODEL_RUN" -H "${HLA_CACHE_DIR}" -S "${SEQ_CACHE}" "$sample_dir"

Rscript "$EPI_EXTRACT" -E "${HLA_SCAN_DIR}/epitopes/aff0.2_bind0.5" -S "$sample_dir"

mut_f="${sample_dir}/mutated.fa"

sample_tmp_dir="${sample_dir}/mut-analysis"
mut_tinyhlanet_dir="${sample_tmp_dir}/tinyhlanet"
mut_hla_cache_dir="${sample_tmp_dir}/promise-hla-cache"
mut_seq_cache_fname='mutated.seqemb.pkl'
mut_seq_cache="${sample_tmp_dir}/${mut_seq_cache_fname}"
mut_genes_f="${sample_tmp_dir}/mut-genes.txt"
mut_tpm_f="${sample_tmp_dir}/tpm.tsv"
if [ -f "$mut_f" ]; then
    if [ ! -d "$sample_tmp_dir" ]; then
        mkdir -p "$sample_tmp_dir"
    fi

    MISSING_ALLELES=""
    for al in $(cat "$al_lst"); do
        al_str=$(al_clean "$al")
        al_f="${mut_tinyhlanet_dir}/prot-scan/${al_str}.md5"
        if [ ! -f "$al_f" ]; then
            if [ "$MISSING_ALLELES" = "" ]; then
                MISSING_ALLELES="$al"
            else
                MISSING_ALLELES="${MISSING_ALLELES} ${al}"
            fi
        fi
    done
    if [ ! -z "$MISSING_ALLELES" ]; then
        "$PSCAN" -a 0.2 -b 0.5 -l <(echo "$MISSING_ALLELES" | sed 's/ /\n/g') -o "$mut_tinyhlanet_dir" "$mut_f" 
    fi
    cp "$al_lst" "${sample_tmp_dir}"
    grep ">" "$mut_f" | sed 's/^>//' | sort | uniq  > "$mut_genes_f"

    Rscript -e "
    x <- data.frame(
        gene = readLines('${mut_genes_f}')
    )
    x[['orig_gene']] <- sapply(strsplit(x[['gene']], '[:]'), '[[', 1)
    tpm <- read.table('${tpm_f}', sep = '\t', header = T, stringsAsFactors = F, quote = '')

    common_genes <- sort(intersect(tpm[['gene']], x[['orig_gene']]))
    y <- subset(x, orig_gene %in% common_genes)
    y[['tpm']] <- tpm[['tpm']][match(y[['orig_gene']], tpm[['gene']])]
    write.table(y[,c('gene', 'tpm')], '${mut_tpm_f}', sep = '\t', row.names = F, quote = F)
    "

    if [ -f "$mut_tpm_f" ]; then
        cp "$al_lst" "${sample_tmp_dir}"

        "$PY" "$HLA_EMBDIR_PY" -o "$mut_hla_cache_dir" -a "$al_lst" "${mut_tinyhlanet_dir}/prot-scan"
        if [ ! -f "$mut_seq_cache" ]; then
            "$PY" "$SEQ_EMB_PY" -o "$sample_tmp_dir" -f "$mut_seq_cache_fname" "$mut_f"
        fi
        "$PY" "$MODEL_RUN" -a -H "${mut_hla_cache_dir}" -S "${mut_seq_cache}" "$sample_tmp_dir"

        Rscript "$EPI_EXTRACT" -E "${mut_tinyhlanet_dir}/epitopes/aff0.2_bind0.5" -S "$sample_tmp_dir"
    fi

fi


#---------------------------------------------------------#

if [ ! -z "$2" ] && [ "$2" = "mtechi" ]; then
    mtechi_tmp_dir="${sample_dir}/thymus"
    if [ ! -d "$mtechi_tmp_dir" ]; then
        mkdir -p "$mtechi_tmp_dir"
    fi

    mtechi_tpm_f="${mtechi_tmp_dir}/tpm.tsv"
    cp "$MTECHI_TPM" "${mtechi_tpm_f}"

    mtechi_al_lst="${mtechi_tmp_dir}/alleles.txt"
    cp "$al_lst" "$mtechi_al_lst"


    "$PY" "$MODEL_RUN" -H "${HLA_CACHE_DIR}" -S "${SEQ_CACHE}" "$mtechi_tmp_dir"

    Rscript "$EPI_EXTRACT" -p 0.5 -E "${HLA_SCAN_DIR}/epitopes/aff0.2_bind0.5" -S "$mtechi_tmp_dir"
    Rscript "$EPI_FIN" "$sample_dir"

fi
