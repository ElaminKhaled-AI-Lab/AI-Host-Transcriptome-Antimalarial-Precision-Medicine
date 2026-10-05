# R / Bioconductor environment (blueprint Section 91)

The R stack is used for differential expression, MOFA+ (`MOFA2`), and
supervised integration (`mixOmics` / DIABLO). Install once:

```r
# From an R session (R >= 4.3 recommended)
install.packages("BiocManager")
BiocManager::install(c(
  "DESeq2",     # host RNA differential expression
  "edgeR",      # count-based DE
  "limma",      # linear models for expression
  "mixOmics",   # DIABLO supervised integration
  "MOFA2"       # Multi-Omics Factor Analysis Plus (R interface)
))
```

Proteomics/metabolomics may need additional specialized tools
(e.g. `MSnbase`, `xcms`) depending on the raw file formats you choose to
reprocess. Add them only when you decide to reprocess raw spectra rather
than use the submitted processed matrices.

MOFA2 in R talks to the Python `mofapy2` backend installed via
`envs/environment.yml`. Point R to the same conda env with:

```r
library(reticulate)
use_condaenv("malaria", required = TRUE)
```
