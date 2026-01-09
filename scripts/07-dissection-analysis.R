library(scriptloc)
source(file.path(script_dir_get(), '..', '.conf.R'))

library(pheatmap)


mdissect_d <- file.path(RES_DIR, '02-model-dissection')
ds <- local({
    tmp <- list.dirs(mdissect_d, recursive = F)
    setNames(tmp, basename(tmp))
})
fs <- setNames(file.path(ds, 'pred.tsv'), names(ds))


tpm_df <- read.table(fs[['tpm']], sep = '\t', header = T)


{
svg(file.path(PLOTS_DIR, 'tpm-dissection.svg'))
par(cex = 1.5)
with(tpm_df, {
    plot(
        pred ~ log10(tpm + 1),
        type = 'l',
        ylab = "Gene-presentation probability",
        xlab = "TPM",
        xaxt = 'n',
        ylim = c(0, 1),
        xlim = c(0, 3.01),
        lwd = 1.5
    )
})
tks <- c(0, 1, 10, 100, 1000)
axis(1, log10(tks + 1),
     labels = as.character(tks))
abline(h = 0.5, lty = 'dotted', col = 'grey50', lwd = 1)
abline(v = log10(min(tpm_df$tpm[tpm_df$pred >= 0.5]) + 1),
       lty = 'dotted', col = 'grey50', lwd = 1)
text(x = 2, y = 0.52, labels = '50% probability')
dev.off()
}


pparam <- read.table(PPARAM_F, sep = '\t', header = T)
seq_df <- local({
    tmp <- read.table(fs[['tpm-seq']], sep = '\t', header = T)
    tmp[,colnames(tmp) != "tpm"]
})

inds <- match(seq_df$gene, pparam$gene)
cnames <- setdiff(colnames(pparam), 'gene')
seq_df[,cnames] <- pparam[inds, cnames]

probs <- quantile(seq_df$pred, c(0.25, 0.5, 0.75))
seq_df$quartile <- NA
seq_df$quartile[seq_df$pred <  quantile(seq_df$pred, 0.25)] <- "Q1"
seq_df$quartile[seq_df$pred >= quantile(seq_df$pred, 0.25)] <- "Q2"
seq_df$quartile[seq_df$pred >= quantile(seq_df$pred, 0.50)] <- "Q3"
seq_df$quartile[seq_df$pred >= quantile(seq_df$pred, 0.75)] <- "Q4"

{
svg(file.path(PLOTS_DIR, 'seq-dissection-protparam-SRCC.svg'))
corr_df <- seq_df[,!(colnames(seq_df) %in% c("gene", "quartile"))]
pparam_corr <- cor(corr_df, method = 'spearman')[-1 ,1, drop = F]
colnames(pparam_corr) <- ""
br  <- seq(-0.5, 0.5, 0.1)
cls <- rev(hcl.colors(length(br) - 1, palette = "RdBu"))
pheatmap(
    t(pparam_corr),
    breaks = br,
    color  = cls,
    cluster_rows = F,
    cluster_cols = F,
    display_numbers = T,
    number_color = "black",
    border_color = "black",
    cellwidth    = 30,
    cellheight   = 30,
    angle_col    = 45,
    legend = F,
    fontsize_number = 12,
    fontsize = 15
)
dev.off()
}

{
svg(file.path(PLOTS_DIR, 'seq-dissection-protparam-quartile.svg'))
par(mfrow = c(3, 3), cex = 0.5, mar = c(3, 3, 2, 2))
for (property in cnames) {
procfn <- if (property %in% c("Length", "MolecularWeight")) function(x) log10(x + 1) else identity
sp <- lapply(split(seq_df[[property]], seq_df$quartile), procfn)
yl <- if (property %in% c("Length", "MolecularWeight")) {
    sprintf("Log10(%s + 1)", property)
} else {
    property
}
boxplot(sp, main = yl)
}
dev.off()
}

mtechi_f <- file.path(PREREQ_DIR, 'mTEChi-protein-coding-ranks.tsv')
mtechi   <- read.table(mtechi_f, sep = '\t', header = T, stringsAsFactors = F)
gmap     <- mtechi[,c("ensembl_id", "symbol")]
stopifnot(all(seq_df$gene %in% gmap$ensembl_id))

deeploc_f <- file.path(PREREQ_DIR, 'human-canonical-deeploc.csv')
deeploc <- read.delim(deeploc_f, sep = ',', quote = '', header = T, check.names = F)

score_df  <- local({
    tmp <- seq_df[,c("gene", "pred", "quartile")]
    tmp$symbol <- gmap$symbol[match(tmp$gene, gmap$ensembl_id)]
    tmp1 <- subset(tmp[,c("gene", "symbol", "pred", "quartile")],
                   gene %in% deeploc$Protein_ID)
    inds <- match(tmp1$gene, deeploc$Protein_ID)
    cnames <- colnames(deeploc)[5:17]
    tmp1[,cnames] <- deeploc[inds, cnames]
    tmp1
})

{
score_mat <- as.matrix(score_df[,sapply(score_df, is.numeric)])
deeploc_corr <- cor(score_mat, method = 'pearson')[-1 ,1, drop = F]
colnames(deeploc_corr) <- ""
svg(file.path(PLOTS_DIR, 'seq-dissection-deeploc-SRCC.svg'))
br  <- seq(-0.5, 0.5, 0.1)
cls <- rev(hcl.colors(length(br) - 1, palette = "RdBu"))
pheatmap(
    t(deeploc_corr),
    breaks = br,
    color  = cls,
    cluster_rows = F,
    cluster_cols = F,
    display_numbers = T,
    number_color = "black",
    border_color = "black",
    cellwidth    = 30,
    cellheight   = 30,
    angle_col    = 45,
    legend = F,
    fontsize_number = 12,
    fontsize = 15
)
dev.off()
}

subclocs <- colnames(score_mat)[-1]
{
svg(file.path(PLOTS_DIR, 'seq-dissection-deeploc-quartile.svg'))
par(mfrow = c(4, 4), cex = 0.5, mar = c(3, 3, 2, 2))
for (subcloc in subclocs) {
procfn <- identity
sp <- lapply(split(score_df[[subcloc]], score_df$quartile), procfn)
yl <- paste0(subcloc, " Probability")
boxplot(sp, main = yl, ylim = c(0, 1))
}
dev.off()
}




SCRIPTS_DIR <- script_dir_get()
tmp_odir <- file.path(SCRIPTS_DIR, 'tmp')
if (!dir.exists(tmp_odir)) dir.create(tmp_odir, recursive = T)

qns <- c(0.01, 0.05, 0.1, 0.2, 0.25)

for (qn in qns) {
    tp <- quantile(score_df$pred, 1-qn)
    lw <- quantile(score_df$pred, qn)
    writeLines(sort(score_df$symbol[score_df$pred >= tp]),
               file.path(tmp_odir,
                         paste0("top-", sub("[.]", "_", qn), ".txt")))

    writeLines(sort(score_df$symbol[score_df$pred <= lw]),
               file.path(tmp_odir,
                         paste0("bot-", sub("[.]", "_", qn), ".txt")))
}


library(Boruta)

sdf3 <- local({
    ind <- match(score_df$gene, seq_df$gene)
    tmp <- seq_df[ind,]
    tmp1 <- cbind.data.frame(score_df, tmp[,3:10])
    tmp1[,colnames(tmp1) != "Plastid"]
})




subcloc_lfc <- sapply(setdiff(subclocs, "Plastid"), function(cname) {
    datf <- sdf3[,c("pred", cname, "quartile")]
    sp   <- split(datf[[cname]], datf$quartile)
    md   <- sapply(sp, median)
    q4   <- md[["Q4"]]
    q1   <- md[["Q1"]]
    round(log2(q4/q1), 3)
})

{
subcloc_lfc_mat <- t(as.matrix(subcloc_lfc))
subcloc_lfc_mat_ncol <- matrix("black",
                               nrow = nrow(subcloc_lfc_mat),
                               ncol = ncol(subcloc_lfc_mat))
subcloc_lfc_mat_ncol[abs(subcloc_lfc_mat) > 0.4] <- "white"
svg(file.path(PLOTS_DIR, 'seq-dissection-deeploc-q4q1-lfc.svg'))
br  <- seq(-0.5, 0.5, 0.1)
cls <- rev(hcl.colors(length(br) - 1, palette = "RdBu"))
pheatmap(
    subcloc_lfc_mat,
    breaks = br,
    color  = cls,
    cluster_rows = F,
    cluster_cols = F,
    display_numbers = T,
    border_color = "black",
    cellwidth    = 30,
    cellheight   = 30,
    angle_col    = 45,
    legend = F,
    fontsize_number = 12,
    fontsize = 15,
    number_color = subcloc_lfc_mat_ncol
)
dev.off()
}

prop_lfc <- sapply(setdiff(colnames(seq_df), c("gene", "pred", "quartile")), function(cname) {
    datf <- sdf3[,c("pred", cname, "quartile")]
    sp   <- split(datf[[cname]], datf$quartile)
    md   <- sapply(sp, median)
    q4   <- md[["Q4"]]
    q1   <- md[["Q1"]]
    round(log2(q4/q1), 3)
})

{
prop_lfc_mat <- t(as.matrix(prop_lfc))
prop_lfc_mat_ncol <- matrix("black",
                               nrow = nrow(prop_lfc_mat),
                               ncol = ncol(prop_lfc_mat))
prop_lfc_mat_ncol[abs(prop_lfc_mat) > 0.4] <- "white"
svg(file.path(PLOTS_DIR, 'seq-dissection-protparam-q4q1-lfc.svg'))
br  <- seq(-0.5, 0.5, 0.1)
cls <- rev(hcl.colors(length(br) - 1, palette = "RdBu"))
pheatmap(
    prop_lfc_mat,
    breaks = br,
    color  = cls,
    cluster_rows = F,
    cluster_cols = F,
    display_numbers = T,
    border_color = "black",
    cellwidth    = 30,
    cellheight   = 30,
    angle_col    = 45,
    legend = F,
    fontsize_number = 12,
    fontsize = 15,
    number_color = prop_lfc_mat_ncol
)
dev.off()
}


# gset_dir <- file.path(mdissect_d, 'genesets')
# if (!dir.exists(gset_dir)) dir.create(gset_dir, recursive = T)
# 
# gset_urls <- list(
#     'GO_BP_2025' = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2025',
#     'GO_CC_2025' = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Cellular_Component_2025',
#     'COMPARTMENTS' = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=COMPARTMENTS_Curated_2025'
# )
# 
# gset_fs <- setNames(file.path(gset_dir, paste0(names(gset_urls), '.gmt')),
#                     names(gset_urls))
# 
# 
# 
# writeLines(score_df$symbol[score_df$pred >= quantile(score_df$pred, 0.99)],
#            'top1perc.txt')
# 
# writeLines(score_df$symbol[score_df$pred <= quantile(score_df$pred, 0.01)], 
#            'bot1perc.txt')
# 
# bkg <- unique(score_df$symbol)
# gmt_read <- function(f, bkg = NULL, minsize = 10) {
#     comps <- strsplit(readLines(f), "\t")
#     gs    <- setNames(lapply(comps, `[`, c(-1, -2)),
#                       sapply(comps, `[[`, 1))
#     if (!is.null(bkg)) {
#         return(Filter(function(x) length(x) >= minsize,
#                       lapply(gs, function(g) sort(intersect(g, bkg)))))
#     } else {
#         return(gs)
#     }
# }
# 
# gsets_lst <- setNames(lapply(names(gset_fs), function(gset_db) {
#     gset_f   <- gset_fs[[gset_db]]
#     gset_url <- gset_urls[[gset_db]]
#     if (!file.exists(gset_f)) download.file(gset_url, gset_f)
#     stopifnot(file.exists(gset_f))
#     gmt_read(gset_f)
# }), names(gset_fs))
# 
# # gset_lens <- sort(unique(Reduce(c, lapply(gsets_lst, function(gsets) {
# #     sapply(gsets, length)
# # }))))
# # 
# # NSIM <- 10000
# # rand_runs <- lapply(gset_lens, function(len) {
# #     print(len)
# #     set.seed(1)
# #     replicate(NSIM, median(sample(score_df$pred, size = len)))
# # })
# # 
# # names(rand_runs) <- paste0("len", gset_lens)
# # 
# # res_lst <- lapply(names(gsets_lst), function(dbname) {
# #     print(dbname)
# #     gsets <- gsets_lst[[dbname]]
# # 
# #     gset_df <- data.frame(
# #         gset = names(gsets),
# #         db   = dbname
# #     )
# # 
# #     gset_df$mdiff <- sapply(gsets, function(gs) {
# #         ind   <- paste0("len", length(gs))
# #         vrand <- rand_runs[[ind]]
# #         v     <- median(subset(score_df, symbol %in% gs)$pred)
# #         median(v - vrand)
# #     })
# # 
# #     gset_df$higher_pval <- sapply(gsets, function(gs) {
# #         ind   <- paste0("len", length(gs))
# #         vrand <- rand_runs[[ind]]
# #         v     <- median(subset(score_df, symbol %in% gs)$pred)
# #         mean(vrand >= v)
# #     })
# #     gset_df$lower_pval <- sapply(gsets, function(gs) {
# #         ind   <- paste0("len", length(gs))
# #         vrand <- rand_runs[[ind]]
# #         v     <- median(subset(score_df, symbol %in% gs)$pred)
# #         mean(vrand <= v)
# #     })
# # 
# #     gset_df
# # })
# # 
# # 
# # res_df <- do.call("rbind.data.frame", res_lst)
# # res_df$lower_qval  <- pmax(p.adjust(res_df$lower_pval, method = 'fdr'), 10e-8)
# # res_df$higher_qval <- pmax(p.adjust(res_df$higher_pval, method = 'fdr'), 10e-8)
# # 
# # res_df$nql <- (-log10(res_df$lower_qval))
# # res_df$nqh <- (-log10(res_df$higher_qval))
# # 
# # res_df     <- res_df[order(-abs(res_df$mdiff)),]
# # 
# # s1 <- subset(res_df, nql >= 6)
# # s2 <- subset(res_df, nqh >= 6)
