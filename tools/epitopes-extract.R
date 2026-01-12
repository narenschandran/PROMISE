library(getopt)
library(data.table)

spec <- matrix(c(
    'sample_dir'  , 'S', 1, 'character',
    'epitopes_dir', 'E', 1, 'character',


    'epitope_cutoff'     , 'e', 2, 'numeric',
    'epitope_rank_cutoff', 'r', 2, 'numeric',
    'presentation_cutoff', 'p', 2, 'numeric'
), byrow = T, ncol = 4)

opt <- getopt(spec)

stopifnot((!is.null(opt$epitopes_dir)) && (!is.null(opt$sample_dir)))
if (is.null(opt$epitope_cutoff)) opt$epitope_cutoff <- 0.7
if (is.null(opt$epitope_rank_cutoff)) opt$epitope_rank_cutoff <- 3
if (is.null(opt$presentation_cutoff)) opt$presentation_cutoff <- 0.9

sample_dir <- opt$sample_dir
al_f       <- file.path(sample_dir, 'alleles.txt')
pred_f     <- file.path(sample_dir, 'pred.tsv')
tpm_f      <- file.path(sample_dir, 'tpm.tsv')

tpm <- read.table(tpm_f, sep = '\t', header = T, stringsAsFactors = F, quote = '')

x <- read.table(pred_f, sep = '\t', header = T,
                stringsAsFactors = F, check.names = F)

score_cols_l  <- grepl("^al[0-9]+:.*$", colnames(x))
stopifnot(sum(score_cols_l) > 0)
score_cols    <- colnames(x)[score_cols_l]

col2allele <- setNames(sub("^al[0-9]+:", "", score_cols), 
                       score_cols)
allele_to_str  <- function(x) gsub("[*:]", "-", x)

res_lst <- list()
for (cname in names(col2allele)) {
    allele     <- col2allele[[cname]]
    allele_str <- allele_to_str(allele)
    if (!(allele_str %in% names(res_lst))) {
        pred_score <- setNames(x[[cname]], x[["gene"]])

        epi_fname  <- paste0(allele_str, '.tsv.gz')
        epi_file   <- file.path(opt$epitopes_dir, epi_fname)

        epi_source <- names(which(pred_score >= opt$presentation_cutoff))
        epi_df     <- fread(epi_file, sep = '\t', header = T)
        best_epi_df <- subset(epi_df,
                              (bindprob >= opt$epitope_cutoff) &
                              (rank     <= opt$epitope_rank_cutoff) &
                              (protein %in% epi_source))
        best_epi_df$allele <- allele
        best_epi_df$allele_copy <- sum(col2allele %in% allele)
        best_epi_df$presentation <- pred_score[best_epi_df$protein]
        best_epi_df$tpm <- tpm$tpm[match(best_epi_df$protein, tpm$gene)]
        res_lst[[allele]] <- best_epi_df
    }
}
res <- do.call("rbind.data.frame", res_lst)

write.table(res, file.path(sample_dir, 'epitopes.tsv'),
            sep = '\t', quote = F, row.names = F)
