CURR_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
conf_script="${CURR_SCRIPT_DIR}/../.conf.sh"

if [ ! -f "$conf_script" ]; then
    echo "Unable to find configuration script at: [$conf_script]" >&2
    echo "Exiting..." >&2
    kill -9 $$
else
    source "$conf_script"
fi

if [ ! -d "$TINYHLANET_DIR" ]; then
    (
        dir_make "$BIN_DIR"
        cd "$BIN_DIR"
        git clone https://github.com/narenschandran/TinyHLAnet.git
    )
fi

CONDA=$(command -v 'conda')
if [ -z "$CONDA" ]; then
    echo "Unable to find [conda] command to set up TinyHLAnet." >&2
    echo "Exitting..." >&2
    kill -9 $$
fi

echo "Setting up conda environment"
(
    cd "${TINYHLANET_DIR}"
    "$CONDA" env create -f tinyhlanet.conda.yml
    "$CONDA" run -n 'tinyhlanet' bash run-data/setup.sh
)
