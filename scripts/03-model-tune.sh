CURR_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
conf_file="${CURR_SCRIPT_DIR}/../.conf.sh"

if [ ! -f "$conf_file" ]; then
    echo "Unable to find configuration file at: [$conf_file]" >&2
    echo "Exiting..." >&2
    kill -9 $$
else
    source "$conf_file"
fi


ENVDIR="${TINYHLANET_DIR}/.env/deephlaffy"
if [ -d "$ENVDIR" ]; then
    source "${ENVDIR}/bin/activate"
    PY="${ENVDIR}/bin/python3"
else
    PY="python"
fi

TRAINPY="${CURR_SCRIPT_DIR}/train.py"
BASE_OUTPUT_DIR="${PROJROOT}/results/01-ablation-study"

run_model() {
    # Fixed hyperparameters
    local nmodels=2
    local max_prot_size=3000
    local wsize=50
    local batch_size=256
    local output_dir="${BASE_OUTPUT_DIR}/p${max_prot_size}"

    local base_cmd="CUDA_VISIBLE_DEVICES='-1' $PY $TRAINPY"
    local hyperparams="$1"
    local other_opts="-l ${max_prot_size} -w ${wsize} -d ${output_dir} -b ${batch_size}"


    local seedlim=$(echo "${nmodels} - 1" | bc)
    for seed in $(seq 0 "$seedlim"); do
        full_cmd="${base_cmd} ${other_opts} -s ${seed}"
        if [ ! -z "$hyperparams" ]; then
            full_cmd="${full_cmd} ${hyperparams}"
        fi
        echo "$full_cmd"
        if [ -z "$DRYRUN" ]; then
            echo "Running..."
            eval "$full_cmd"
        fi
    done

}

prob_conf="-O gelu64_sigmoid1"
tpm_conf="-T"
seq_conf="-S 16 -C gelu64_gelu32 -D 64"
epi_conf="-F gelu64_gelu32 -E 32"

# Full model
hyper1="${prob_conf} ${tpm_conf} ${seq_conf} ${epi_conf}"
run_model "$hyper1"

# TPM + Seq model
hyper2="${prob_conf} ${tpm_conf} ${seq_conf}"
run_model "$hyper2"

# TPM + Epi model
hyper3="${prob_conf} ${tpm_conf} ${epi_conf}"
run_model "$hyper3"

# Seq + Epi model
hyper4="${prob_conf} ${seq_conf} ${epi_conf}"
run_model "$hyper4"

# TPM model
hyper5="${prob_conf} ${tpm_conf}"
run_model "$hyper5"

# Seq model
hyper6="${prob_conf} ${seq_conf}"
run_model "$hyper6"

# Epi model
hyper7="${prob_conf} ${epi_conf}"
run_model "$hyper7"

