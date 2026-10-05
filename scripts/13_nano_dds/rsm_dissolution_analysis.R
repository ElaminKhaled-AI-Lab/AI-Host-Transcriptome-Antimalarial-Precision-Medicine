#!/usr/bin/env Rscript
# Phase 20 -- classical pharmaceutical statistics for the nano-DDS factorial.
#
# WHY R AS WELL AS PYTHON.
# Python carries the deep learning (PROFILE-Net and the integration benchmark),
# where the PyTorch/scikit-learn ecosystem is strongest. R is used here for the
# classical design-of-experiments and dissolution statistics that regulators and
# formulation scientists read in their conventional form: analysis of variance
# (ANOVA) tables with F and p values, second-order response-surface models with
# lack-of-fit tests, canonical/stationary-point analysis, and the f2 similarity
# factor. The two languages are cross-checked against each other -- the Python
# response-surface coefficients and the R ones should agree, which is a genuine
# implementation check rather than a duplication of effort.
#
# Input : tables/nano_dds_dissolution_parameters.csv (written by analyze_dissolution.py)
# Output: tables/rsm_anova_<response>.csv, results/rsm_r_summary.txt,
#         figures/FR1_rsm_contours.png
#
# Usage: Rscript scripts/13_nano_dds/rsm_dissolution_analysis.R [polymer] [drug]

suppressWarnings(suppressMessages({
  ok_rsm <- requireNamespace("rsm", quietly = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
polymer_sel <- if (length(args) >= 1) args[1] else "PLGA_50_50"
drug_sel    <- if (length(args) >= 2) args[2] else "lumefantrine"

# locate project root (walk up until config/ and src/ exist)
root <- normalizePath(".")
while (!(dir.exists(file.path(root, "config")) && dir.exists(file.path(root, "src")))) {
  parent <- dirname(root)
  if (parent == root) stop("project root not found")
  root <- parent
}
tbl_dir <- file.path(root, "tables")
res_dir <- file.path(root, "results")
fig_dir <- file.path(root, "figures")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

infile <- file.path(tbl_dir, "nano_dds_dissolution_parameters.csv")
if (!file.exists(infile)) stop("run analyze_dissolution.py first: ", infile)
d <- read.csv(infile, stringsAsFactors = FALSE)

d <- subset(d, polymer == polymer_sel & drug == drug_sel)
if (nrow(d) != 27) {
  message(sprintf("note: %d rows for %s/%s (expected the 27-run factorial)",
                  nrow(d), polymer_sel, drug_sel))
}

# code the three factors to -1/0/+1 (the standard DoE coding)
code <- function(x) {
  u <- sort(unique(x)); ifelse(x == u[1], -1, ifelse(x == u[2], 0, 1))
}
d$A <- code(d$A_ratio)
d$B <- code(d$B_stabilizer_pct)
d$C <- code(d$C_energy)

responses <- c("size_nm", "pdi", "zeta_mv", "ee_pct", "DE_pct", "MDT_h",
               "t50_h", "n_peppas", "liver_targeting_index")
responses <- responses[responses %in% names(d)]

sink(file.path(res_dir, "rsm_r_summary.txt"))
cat("Classical response-surface analysis (R)\n")
cat("Polymer:", polymer_sel, "| Drug:", drug_sel, "| Runs:", nrow(d), "\n")
cat("Factors: A = polymer:drug ratio, B = stabilizer %, C = homogenization energy\n")
cat(strrep("=", 78), "\n\n")

for (resp in responses) {
  f <- as.formula(paste(resp, "~ A + B + C + A:B + A:C + B:C + I(A^2) + I(B^2) + I(C^2)"))
  fit <- lm(f, data = d)
  s <- summary(fit)
  cat("### Response:", resp, "\n")
  cat(sprintf("R^2 = %.4f   adjusted R^2 = %.4f\n", s$r.squared, s$adj.r.squared))
  print(anova(fit))
  cat("\nCoefficients:\n"); print(round(coef(s), 5))
  cat("\n", strrep("-", 78), "\n\n")

  co <- as.data.frame(round(coef(s), 6))
  co$term <- rownames(co)
  write.csv(co, file.path(tbl_dir, paste0("rsm_anova_", resp, ".csv")), row.names = FALSE)
}

# ---- f2 similarity between the centre point and every other run --------------
cat("### f2 similarity factor (centre run as reference)\n")
cat("f2 >= 50 with f1 <= 15 indicates equivalent release.\n")
if ("f2_vs_PLGA_ref" %in% names(d)) {
  cat(sprintf("mean f2 vs PLGA reference = %.2f (range %.2f - %.2f)\n",
              mean(d$f2_vs_PLGA_ref, na.rm = TRUE),
              min(d$f2_vs_PLGA_ref, na.rm = TRUE),
              max(d$f2_vs_PLGA_ref, na.rm = TRUE)))
  cat(sprintf("runs equivalent to reference: %d of %d\n",
              sum(d$f2_vs_PLGA_ref >= 50, na.rm = TRUE), nrow(d)))
}

# ---- canonical analysis with the rsm package when available ------------------
if (ok_rsm) {
  cat("\n### Canonical (stationary-point) analysis via the rsm package\n")
  library(rsm)
  for (resp in c("size_nm", "ee_pct", "DE_pct")) {
    if (!resp %in% names(d)) next
    rs <- rsm(as.formula(paste(resp, "~ SO(A, B, C)")), data = d)
    cat("\n-- ", resp, " --\n", sep = "")
    print(summary(rs))
  }
} else {
  cat("\n(note: install.packages('rsm') to add canonical stationary-point analysis)\n")
}
sink()

# ---- contour figure ----------------------------------------------------------
png(file.path(fig_dir, "FR1_rsm_contours.png"), width = 1500, height = 500, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (resp in c("size_nm", "ee_pct", "DE_pct")) {
  if (!resp %in% names(d)) next
  fit <- lm(as.formula(paste(resp, "~ A + B + C + A:B + I(A^2) + I(B^2)")), data = d)
  ga <- seq(-1, 1, length.out = 40); gb <- seq(-1, 1, length.out = 40)
  grid <- expand.grid(A = ga, B = gb, C = 0)
  grid$z <- predict(fit, newdata = grid)
  z <- matrix(grid$z, nrow = length(ga))
  contour(ga, gb, z, xlab = "A: polymer:drug ratio (coded)",
          ylab = "B: stabilizer % (coded)",
          main = paste0(resp, "\n", polymer_sel, " / ", drug_sel))
  image(ga, gb, z, col = hcl.colors(30, "YlGnBu"), add = TRUE)
  contour(ga, gb, z, add = TRUE, col = "black")
}
dev.off()

cat("R analysis complete.\n")
cat("  results/rsm_r_summary.txt\n  tables/rsm_anova_*.csv\n  figures/FR1_rsm_contours.png\n")
