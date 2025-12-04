CURR_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
conf_file="${CURR_SCRIPT_DIR}/../.conf.sh"

if [ ! -f "$conf_file" ]; then
    echo "Unable to find configuration file at: [$conf_file]" >&2
    echo "Exiting..." >&2
    kill -9 $$
else
    source "$conf_file"
fi

dir_make "$DATA_DIR"
gunzip -c "$MINP_F" | tar xf - -C "${DATA_DIR}"

al_f="${DATA_DIR}/tune-alleles.txt"
cat "${MINP_DIR}/"*'/'*'/alleles.txt' | sort | uniq > "${al_f}"

if [ ! -f "${TINYHLANET_DIR}/run-data/hlainp.pkl" ]; then
    (
        cd "${TINYHLANET_DIR}/run-data"
        bash setup.sh
    )
fi

ENVDIR="${TINYHLANET_DIR}/.env/deephlaffy"
if [ -d "$ENVDIR" ]; then
    source "${ENVDIR}/bin/activate"
fi
"${PSCAN}" -l "$al_f" -o "${DATA_DIR}/hla-scan" "$CANON_PROTEOME_FA"
