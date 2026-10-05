#!/usr/bin/env Rscript
# Phase 9 -- Host transcriptome single-omics DE with DESeq2 (blueprint Section 32).
#
# Reproduces the paper's "CM vs SMA" differential expression (Supplementary
# Data MOESM4) from the raw GSE289197 counts + the clinical labels we extracted.
# Reproducing a published result is the strongest check that the counts,
# sample-to-group mapping, and DE model are all correct before we build anything
# on top of the transcriptome.
#
# Usage:
#   Rscript scripts/04_single_omics/transcriptome_deseq2.R
#
# Inputs (auto-located under the project root):
#   data/raw/host_transcriptomics/GSE289197/GSE289197_raw_counts.txt.gz
#   metadata/sample_maps/transcriptome_clinical.tsv   (has clin_match_group + join_key)
# Optional benchmark:
#   metadata/sample_maps/41467_2025_59281_MOESM4_ESM.xlsx  (paper CM vs SMA)
#
# Outputs -> results/single_omics/transcriptome/
#   deseq2_CM_vs_SMA.tsv           full results (log2FC, padj) for every gene
#   deseq2_CM_vs_SMA_sig.tsv       FDR < 0.05 & |log2FC| > 1
#   deseq2_vs_paper_overlap.txt    concordance with MOESM4 (if present)

suppressMessages({
  ok <- require(DESeq2, quietly = TRUE)
})
if (!ok) {
  stop("DESeq2 not installed. In R:\n",
       "  install.packages('BiocManager'); BiocManager::install('DESeq2')")
}

# ---- locate project root (dir containing config/ and data/) ----------------
find_root <- function() {
  d <- getwd()
  while (d != dirname(d)) {
    if (dir.exists(file.path(d, "config")) && dir.exists(file.path(d, "data"))) return(d)
    d <- dirname(d)
  }
  getwd()
}
ROOT <- Sys.getenv("MALARIA_PROJECT_ROOT", find_root())
norm_key <- function(x) toupper(gsub("[-_ ]+", "-", trimws(x)))

# ---- load counts -----------------------------------------------------------
counts_path <- file.path(ROOT, "data/raw/host_transcriptomics/GSE289197",
                         "GSE289197_raw_counts.txt.gz")
stopifnot(file.exists(counts_path))
counts <- read.delim(gzfile(counts_path), row.names = 1, check.names = FALSE)
counts <- counts[, sapply(counts, is.numeric), drop = FALSE]
message(sprintf("counts: %d genes x %d samples", nrow(counts), ncol(counts)))

# ---- load clinical labels --------------------------------------------------
clin <- read.delim(file.path(ROOT, "metadata/sample_maps/transcriptome_clinical.tsv"),
                   check.names = FALSE)
stopifnot("clin_match_group" %in% names(clin), "join_key" %in% names(clin))
clin$key <- norm_key(clin$join_key)

# map each count column to its match group
col_keys <- norm_key(colnames(counts))
grp <- clin$clin_match_group[match(col_keys, clin$key)]
names(grp) <- colnames(counts)
message("group mapping: ", sum(!is.na(grp)), "/", length(grp), " columns labelled")
print(table(grp, useNA = "ifany"))

# ---- restrict to CM vs SMA (the MOESM4 contrast) ---------------------------
keep <- !is.na(grp) & grp %in% c("CM", "SMA")
cnt <- round(as.matrix(counts[, keep, drop = FALSE]))
mode(cnt) <- "integer"
group <- factor(grp[keep], levels = c("SMA", "CM"))  # CM vs SMA (SMA = reference)
message(sprintf("DE set: %d samples (CM=%d, SMA=%d)",
                length(group), sum(group == "CM"), sum(group == "SMA")))

coldata <- data.frame(row.names = colnames(cnt), match_group = group)
cnt[is.na(cnt)] <- 0L                            # a missing count is zero
storage.mode(cnt) <- "integer"
cnt <- cnt[rowSums(cnt) >= 10, , drop = FALSE]   # drop all-near-zero genes
# drop HTSeq summary rows if present (they start with '__')
cnt <- cnt[!grepl("^__", rownames(cnt)), , drop = FALSE]

# ---- DESeq2 ----------------------------------------------------------------
dds <- DESeqDataSetFromMatrix(cnt, coldata, design = ~ match_group)
dds <- DESeq(dds, quiet = TRUE)
res <- results(dds, contrast = c("match_group", "CM", "SMA"))
res <- as.data.frame(res)
res$gene <- rownames(res)
res <- res[order(res$padj), c("gene", "baseMean", "log2FoldChange",
                              "lfcSE", "stat", "pvalue", "padj")]

# ---- map Ensembl gene IDs -> HGNC symbols (paper MOESM4 uses symbols) ------
res$symbol <- NA_character_
if (requireNamespace("org.Hs.eg.db", quietly = TRUE) &&
    requireNamespace("AnnotationDbi", quietly = TRUE)) {
  ens <- sub("\\..*$", "", res$gene)  # strip version suffix if any
  res$symbol <- suppressMessages(AnnotationDbi::mapIds(
    org.Hs.eg.db::org.Hs.eg.db, keys = ens, keytype = "ENSEMBL",
    column = "SYMBOL", multiVals = "first"))
  message(sprintf("mapped %d/%d Ensembl IDs to symbols",
                  sum(!is.na(res$symbol)), nrow(res)))
} else {
  message("(org.Hs.eg.db not installed -> cannot map to symbols; in R: ",
          "BiocManager::install('org.Hs.eg.db'))")
}

out_dir <- file.path(ROOT, "results/single_omics/transcriptome")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.table(res, file.path(out_dir, "deseq2_CM_vs_SMA.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
sig <- subset(res, !is.na(padj) & padj < 0.05 & abs(log2FoldChange) > 1)
write.table(sig, file.path(out_dir, "deseq2_CM_vs_SMA_sig.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
message(sprintf("WROTE %d genes (%d significant at FDR<0.05, |log2FC|>1)",
                nrow(res), nrow(sig)))
message("gene id example: ", paste(head(res$gene, 3), collapse = ", "))

# ---- benchmark vs the paper (MOESM4), if openxlsx is available -------------
paper_x <- file.path(ROOT, "metadata/sample_maps/41467_2025_59281_MOESM4_ESM.xlsx")
if (file.exists(paper_x) && requireNamespace("openxlsx", quietly = TRUE)) {
  pap <- openxlsx::read.xlsx(paper_x, sheet = 1)
  names(pap)[1] <- "gene"
  # match on symbol (paper uses symbols; our results carry a symbol column)
  ours_key <- toupper(ifelse(is.na(res$symbol), res$gene, res$symbol))
  common <- intersect(ours_key, toupper(pap$gene))
  line <- sprintf("Genes in common with MOESM4 (by symbol): %d\n", length(common))
  if (length(common) > 10) {
    a <- res$log2FoldChange[match(common, ours_key)]
    b <- pap$logFC[match(common, toupper(pap$gene))]
    r <- suppressWarnings(cor(a, b, use = "complete.obs"))
    rho <- suppressWarnings(cor(a, b, method = "spearman", use = "complete.obs"))
    line <- paste0(line,
      sprintf("log2FC Pearson correlation with paper : %.3f\n", r),
      sprintf("log2FC Spearman correlation with paper: %.3f\n", rho),
      sprintf("directional agreement (same sign)     : %.1f%%\n",
              100 * mean(sign(a) == sign(b), na.rm = TRUE)))
  }
  writeLines(line, file.path(out_dir, "deseq2_vs_paper_overlap.txt"))
  cat(line)
} else {
  message("(skip paper benchmark: MOESM4 or openxlsx not available; ",
          "install.packages('openxlsx') to enable)")
}
message("done. Compare deseq2_CM_vs_SMA.tsv to the paper's MOESM4 'CM vs SMA'.")
