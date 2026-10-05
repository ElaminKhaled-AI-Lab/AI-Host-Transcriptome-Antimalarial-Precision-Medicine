# MATLAB figure suite — Malaria_Project

Publication figures generated directly from your `results/`, `tables/` and
`reports/` files. **Arial**, 25 pt inside axes, 30 pt headings, **1200 dpi**.
Every subfigure is exported separately as **`.png`, `.pdf` and `.fig`**.

## Install

Copy the `.m` files into your project root, beside `results/` and `tables/`:

```
Malaria_Project/
├── results/   tables/   reports/      <- your data (already there)
├── mal_*.m    F01..F12_*.m            <- this suite
└── figures_matlab/                    <- created automatically
```

`mal_paths.m` auto-detects the root by looking for `results/` and `tables/`.
If it cannot, set `ROOT_OVERRIDE` at the top of that file.

## One prerequisite

F10's release-profile panels need the exact Q(t) curves, which are not in the
dissolution table. Generate them once:

```bash
python scripts/13_nano_dds/export_release_profiles.py     # -> tables/nano_dds_release_profiles.csv
```

A ready-made copy of that CSV ships in this package — just drop it into
`tables/` if you prefer not to re-run the exporter.

## Run

```matlab
cd /media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project
run_all_figures          % everything (48 subfigures, 144 files)
```

or one figure at a time, exactly as you asked:

```matlab
F09_nano_characterisation    % writes F09a..F09f as .png/.pdf/.fig
F12_precision_formulation
```

`run_all_figures` keeps going if one figure fails and lists the failures at the
end, so a single problem never discards the rest.

## Requirements

Base MATLAB only — **no toolboxes**. Deliberately avoids `boxplot`
(Statistics Toolbox); distributions are shown as mean ± SD bars and histograms.
`exportgraphics` needs R2020a+; older releases fall back to `print()`
automatically, and `mal_clim.m` handles the `clim`/`caxis` rename.

## Restyling

Everything visual lives in two files:

- `mal_cfg.m` — fonts, sizes, DPI, default figure dimensions
- `mal_polymer_colors.m` — the fixed colour per polymer, so every figure agrees

Change `C.dpi`, `C.fsAxis` or `C.fsTitle` once and all 48 subfigures follow.

## Files

| File | Role |
|---|---|
| `mal_cfg.m` | central style configuration |
| `mal_paths.m` | locate `results/`, `tables/`, `reports/`, output folder |
| `mal_figure.m` | create a styled, physically-sized figure |
| `mal_axes.m` | apply Arial / 25 pt / 30 pt to an axes |
| `mal_export.m` | write `.png` (1200 dpi) + `.pdf` (vector) + `.fig` |
| `mal_readjson.m` | `jsondecode` that survives bare `NaN` (your JSONs contain them) |
| `mal_palette.m`, `mal_polymer_colors.m` | colour-blind-safe palette, fixed per polymer |
| `mal_label.m`, `mal_clim.m` | in-panel annotations; `clim`/`caxis` compatibility |
| `F01..F12_*.m` | the twelve figures |
| `run_all_figures.m` | run them all |
| `FIGURE_INDEX.md` | every subfigure, its content and its numbers |

## Verification status — read this

I could not execute MATLAB in my environment, so these scripts are **not
runtime-tested**. What I did verify:

- every `.m` file has balanced blocks and brackets (checked programmatically);
- **every data file and every column name they reference exists in the data you
  sent**, and every JSON parses after `NaN` sanitisation.

So the data layer is proven and the structure is sound, but a plotting call
could still need a tweak on your MATLAB release. Send me any error and I'll fix
it immediately.

## MDPI compliance

`mal_minus.m` converts the ASCII hyphen-minus to the typographic minus
**U+2212 (−)** that MDPI copy-editing requires, and `mal_export.m` calls it
automatically before every export — so all 144 files comply without you doing
anything.

It converts **only** a hyphen that begins a number (`-20` → `−20`,
`gain = -0.0095` → `gain = −0.0095`). Hyphens inside words are deliberately
preserved: *Korsmeyer-Peppas*, *liver-targeting*, *one-size-fits-all*,
*well-separated* and identifiers such as `PLGA_50_50` are untouched. Numeric
tick labels are converted wholesale, which is safe because they contain nothing
but numbers.

Other MDPI points already satisfied: 1200 dpi raster **and** vector PDF, Arial
throughout, and each panel exported as an independent file.

## Important: re-run the validation after applying the ARI fix

The shipped `validate_on_real_dualrna.py` corrects a bug that coerced every
UNLABELLED sample into class 0. Re-run it so `real_validation.json` carries
trustworthy numbers, then regenerate F06:

```bash
python scripts/11_validation/validate_on_real_dualrna.py     # no flag
```

Expected after the fix: Mali reports `ari: null` (it genuinely has no severity
labels) instead of a spurious `0.000`, and Gambia reports an ARI computed over
its 45 labelled samples only (UM 25 / CM 20).
