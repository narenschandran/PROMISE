# By convention, the paths defined by this file will be in upper case
library(scriptloc)
PROJROOT     <- script_dir_get()

DATA_DIR     <- file.path(PROJROOT, 'data')
RES_DIR      <- file.path(PROJROOT, 'results')
PLOTS_DIR    <- file.path(PROJROOT, 'plots')
PREREQ_DIR   <- file.path(PROJROOT, 'prereq')

CANON_FA     <- file.path(PREREQ_DIR, 'human-proteome-canonical.fa.gz')
PPARAM_F     <- file.path(PREREQ_DIR, 'human-canonical-protparam.tsv')


# Useful functions
PNG_OPEN <- local({
    tmp <- png
    formals(tmp)$width  <- 3800
    formals(tmp)$height <- 3800
    formals(tmp)$res    <- 300
    tmp
})

CROP_FN <- function(f) {
    cmd <- sprintf("mogrify -trim %s", f)
    system(cmd)
    f
}
PNG_CLOSE <- function(f, crop = T) {
    dev.off()
    if (crop) CROP_FN(f)
    f
}
