args <- commandArgs(trailingOnly = T)
stopifnot(length(args) == 1)
sample_dir <- args[1]

epi_read <- function(f) {
    x <- read.table(f, sep = '\t', header = T,
                    stringsAsFactors = F, quote = '')
    x$pmhc_key <- with(x, paste(allele, epitope))
    x
}



epi_f  <- file.path(sample_dir, 'epitopes.tsv')
mut_f  <- file.path(sample_dir, 'mut-analysis', 'epitopes.tsv')
thym_f <- file.path(sample_dir, 'thymus'      , 'epitopes.tsv')


if (file.exists(thym_f)) {
    epi <- if (file.exists(mut_f)) {
        rbind.data.frame(
            epi_read(epi_f),
            epi_read(mut_f)
        )
    } else {
        epi_read(epi_f)
    }

    thym <- epi_read(thym_f)

    react_epi <- subset(epi, !(pmhc_key %in% thym$pmhc_key))

    write.table(react_epi, sep = '\t', row.names = F, quote = F,
                file.path(sample_dir, 'potentially-reactive-epitopes.tsv'))


    tcr_f <- file.path(sample_dir, 'tcr.tsv')

    if (file.exists(tcr_f)) {

        # The log helps keeps values low so that
        # there's no explosion when multiplied
        rankfn <- function(x) log1p(rank(-x, ties.method = 'min'))

        matched_thym <- do.call("rbind.data.frame", lapply(split(thym, thym$allele), function(high_thym) {
            high_thym$rank_g <- with(high_thym,
                rank(rankfn(bindaff) * rankfn(bindprob) *
                     rankfn(tpm) * rankfn(presentation)))
            high_thym <- high_thym[order(high_thym$rank_g),]
            high_thym <- high_thym[!duplicated(high_thym$pmhc_key),]

            # We will be sampling 100 epitopes from
            # the top 1000
            minn <- min(1000, nrow(high_thym))
            tmp <- high_thym[seq_len(minn),]
            minn2 <- min(100, nrow(tmp))
            set.seed(1)
            inds <- sample(minn2)
            tmp2 <- tmp[inds,]
            tmp2 <- tmp2[order(tmp2$protein, tmp2$epitope_startpos),]
        }))
        rownames(matched_thym) <- NULL


        tcr_dir <- file.path(sample_dir, 'tcr-analysis')
        if (!dir.exists(tcr_dir)) dir.create(tcr_dir, recursive = T)

        write.table(matched_thym, 
                    file.path(tcr_dir, 'random-self.tsv'),
                    sep = '\t', row.names = F, quote = F)



        tcr <- read.table(tcr_f, sep = '\t', header = T,
                          stringsAsFactors  = F)

        repi_lst <- local({
            tmp <- react_epi[!duplicated(react_epi$pmhc_key),]
            split(tmp, seq_len(nrow(tmp)))
        })

        repi <- do.call("rbind.data.frame", lapply(repi_lst, function(repi) {
            tmp <- tcr
            tmp[["T-Cell-Type"]] <- "CD8"
            tmp[["Peptide"]]     <- repi$epitope
            tmp[["MHC"]]         <- repi$allele
            tmp
        }))

        mthym <- do.call("rbind.data.frame", lapply(split(matched_thym, seq_len(nrow(matched_thym))), function(mty) {
            tmp <- tcr
            tmp[["T-Cell-Type"]] <- "CD8"
            tmp[["Peptide"]]     <- mty$epitope
            tmp[["MHC"]]         <- mty$allele
            tmp
        }))


        write.table(repi, file.path(tcr_dir, 'nonself.ergo2.csv'),
                    sep = ',', row.names = F, quote = F)

        write.table(mthym, file.path(tcr_dir, 'random-self.ergo2.csv'),
                    sep = ',', row.names = F, quote = F)

        peps <- unique(sort(c(repi$Peptide, mthym$Peptide)))

        writeLines(peps, file.path(tcr_dir, 'peps.txt'))

    }
}
