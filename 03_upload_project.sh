#!/usr/bin/env bash
# =============================================================================
# STEP 3 of 3 — turn the project folder into a repository and upload it
#
# Run from the PROJECT ROOT:
#
#     cd /media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project
#     bash 03_upload_project.sh
#
# What it does, in order:
#   1. confirms you are in the right folder
#   2. shows the biggest files, so nothing unexpected is uploaded
#   3. writes .gitignore and .gitattributes if they are not already there
#   4. starts Git LFS for large files
#   5. initialises the repository and connects it to GitHub
#   6. refuses to continue if sequence data or a file over 100 MB is staged
#   7. shows you everything it will upload, and asks before doing it
#
# Safe to run again later: it only commits what changed.
# =============================================================================
set -uo pipefail

OWNER="ElaminKhaled-AI-Lab"
REPO="AI-Host-Transcriptome-Antimalarial-Precision-Medicine"
REMOTE="https://github.com/${OWNER}/${REPO}.git"
BRANCH="main"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ============================================================ 1. right folder?
say "[1/7] Checking you are in the project root"

if [[ ! -f "MASTER_PROJECT_BLUEPRINT.md" ]] || [[ ! -f "run_all_figures.m" ]]; then
  fail "This is not the Malaria_Project root.
       Expected MASTER_PROJECT_BLUEPRINT.md and run_all_figures.m here.
       You are in: $(pwd)
       Try:  cd /media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project"
fi
echo "      OK: $(pwd)"

# ============================================================ 2. size audit
say "[2/7] Twenty largest files git would consider"

find . -type f \
  -not -path "./.git/*" -not -path "./data/raw/*" \
  -not -path "./tmp/*"  -not -path "./logs/*" \
  -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -20 | \
  awk -F'\t' '{printf "      %9.1f MB  %s\n", $1/1048576, $2}'

TOTAL=$(du -sh --exclude=.git --exclude=data/raw --exclude=tmp --exclude=logs . 2>/dev/null | cut -f1)
echo
echo "      Folder size excluding raw data and scratch: ${TOTAL}"
echo "      GitHub hard-rejects any single file over 100 MB."

# ============================================================ 3. ignore rules
say "[3/7] Writing .gitignore and .gitattributes"

if [[ -f .gitignore ]]; then
  echo "      .gitignore already exists, leaving it alone"
else
  cat > .gitignore <<'EOF'
# Sequence data is never committed. Retrieve using the accessions in docs/.
data/raw/
*.fastq
*.fastq.gz
*.fq
*.fq.gz
*.bam
*.sam
*.cram
*.sra

# salmon index and quantification scratch
index/
*_quant/
quants/

# archives
*.zip
*.tar
*.tar.gz
*.7z

# scratch
tmp/
logs/
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.Rhistory
.RData
.Rproj.user/

# MATLAB
slprj/
*.asv
*.autosave

# environment and secrets
.env
.venv/
venv/

# OS
.DS_Store
Thumbs.db
EOF
  echo "      .gitignore written"
fi

if [[ -f .gitattributes ]]; then
  echo "      .gitattributes already exists, leaving it alone"
else
  cat > .gitattributes <<'EOF'
data/processed/**/*.tsv                     filter=lfs diff=lfs merge=lfs -text
tables/nano_dds_dissolution_parameters.csv  filter=lfs diff=lfs merge=lfs -text
tables/nano_dds_factorial.csv               filter=lfs diff=lfs merge=lfs -text
figures/**/*.png                            filter=lfs diff=lfs merge=lfs -text
figures_matlab/**/*.png                     filter=lfs diff=lfs merge=lfs -text
figures_matlab/**/*.fig                     filter=lfs diff=lfs merge=lfs -text

*.m   text eol=lf
*.py  text eol=lf
*.R   text eol=lf
*.md  text eol=lf
EOF
  echo "      .gitattributes written"
fi

# ============================================================ 4. Git LFS
say "[4/7] Git LFS"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1 || git lfs install >/dev/null 2>&1
  echo "      OK, large files will go through LFS"
else
  echo "      WARNING: git-lfs is not installed."
  echo "      Install it with:  sudo apt install git-lfs"
  echo "      Without it, large files are committed directly and stay in"
  echo "      the repository history permanently."
  read -r -p "      Continue anyway? [y/N] " a
  [[ "$a" == "y" || "$a" == "Y" ]] || exit 0
fi

# ============================================================ 5. init + remote
say "[5/7] Repository and remote"

if [[ -d .git ]]; then
  echo "      Existing repository found"
  git checkout -B "$BRANCH" >/dev/null 2>&1 || true
else
  git init -b "$BRANCH" >/dev/null
  echo "      Initialised a new repository on branch '$BRANCH'"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi
echo "      Remote: $(git remote get-url origin)"

# ============================================================ 6. stage + guard
say "[6/7] Staging files and running safety checks"

git add -A
N=$(git status --short | wc -l)
echo "      $N paths staged"

if git diff --cached --name-only | grep -qiE '\.(fastq|fq|bam|sam|cram|sra)(\.gz)?$'; then
  echo
  git diff --cached --name-only | grep -iE '\.(fastq|fq|bam|sam|cram|sra)(\.gz)?$' >&2
  fail "Sequence data is staged. Stopping.
       Those files belong in data/raw/, which .gitignore excludes."
fi
echo "      OK: no sequence data staged"

BIG=""
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [[ "$sz" -gt 104857600 ]] && BIG+="      $(( sz / 1048576 )) MB  $f"$'\n'
done < <(git diff --cached --name-only)

if [[ -n "$BIG" ]]; then
  echo
  printf '%s' "$BIG" >&2
  fail "Files over 100 MB are staged; GitHub will reject the push.
       Add them to .gitattributes for LFS, or to .gitignore."
fi
echo "      OK: nothing over 100 MB"

# ============================================================ 7. commit + push
say "[7/7] Review and upload"

echo
echo "      First 30 of the $N paths to be uploaded:"
git status --short | head -30 | sed 's/^/        /'
[[ "$N" -gt 30 ]] && echo "        ... and $(( N - 30 )) more"
echo
echo "      Destination: $REMOTE"
echo
echo "      When git asks for a password, paste your Personal Access Token."
echo
read -r -p "      Upload now? [y/N] " go
[[ "$go" == "y" || "$go" == "Y" ]] || { echo "      Cancelled. Nothing was uploaded."; exit 0; }

if git diff --cached --quiet; then
  echo "      Nothing changed since the last commit."
else
  git commit -q -m "Dual host-parasite multi-omics framework for precision pediatric antimalarial formulation

Pipeline, models, result tables, reports and MATLAB figure suite.
Raw sequence data excluded; retrieve via the accessions in docs/DATA_SOURCES.md."
  echo "      Committed."
fi

echo "      Pushing..."
if git push -u origin "$BRANCH"; then
  echo
  printf '\033[1;32m      Success.\033[0m\n'
  echo "      https://github.com/${OWNER}/${REPO}"
else
  echo
  fail "Push failed. Most common causes:
       - the repository does not exist yet on GitHub (create it first)
       - you pasted your password instead of a Personal Access Token
       - a file exceeds 100 MB
       Read the message above, fix it, and run this script again."
fi
