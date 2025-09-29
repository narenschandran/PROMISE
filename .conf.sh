# By convention, the paths defined by this file will be in upper case
PROJROOT=$(dirname "${BASH_SOURCE[0]}")

ARCHIVE_DIR="${PROJROOT}/archive"
DATA_ARCHIVE="${ARCHIVE_DIR}/data"

DATA_DIR="${PROJROOT}/data"
PREREQ_DIR="${PROJROOT}/prereq"
TMP_DIR="${PROJROOT}/tmp"

BIN_DIR="${PROJROOT}/bin"

CDHIT="${BIN_DIR}/cd-hit-v4.8.1-2019-0228/cd-hit"
if [ ! -x "$CDHIT" ]; then
    CDHIT='cdhit'
fi

# Useful functions:
#
#
err_echo() {
    echo $@ >&2
}

dir_make() {
    if [ -z "$1" ]; then
        err_echo "The [$FUNCNAME] function did not recieve any input"
        err_echo "Exiting script..."
        kill -9 $$
    fi
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
    echo "$1"
}
