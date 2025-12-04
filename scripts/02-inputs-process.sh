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

tune_cache_dir="${DATA_DIR}/model-tuning/cache"

seq_cache_dir="${tune_cache_dir}/seq"
scan_cache_dir="${tune_cache_dir}/scan"

SEQ_PKL="${TOOLS_DIR}/seq2inp.py"
"$PY" "$SEQ_PKL" -o "$seq_cache_dir" "$CANON_PROTEOME_FA"

SCAN_PKL="${TOOLS_DIR}/epi2inp.py"
scan_dir="${DATA_DIR}/hla-scan/prot-scan"
for f in "${scan_dir}/"*.scan.gz; do
    echo "$f"
    "$PY" "$SCAN_PKL" -o "$scan_cache_dir" "$f"
done

DATAGEN_PY="${CURR_SCRIPT_DIR}/datasets-generate.py"
"$PY" "$DATAGEN_PY"
