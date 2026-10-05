#!/usr/bin/env bash
# =====================================================================
# run.sh  --  one entry point to set up and run the Malaria_Project pipeline
#
# Usage:
#   ./run.sh setup          # create the conda/mamba env "malaria" (once)
#   ./run.sh check          # verify config + environment
#   ./run.sh download       # fetch all OPEN data + managed-access checklists
#   ./run.sh download-heavy # also pull large FASTQ + full proteomics
#   ./run.sh demo           # run transportability + formulation self-tests (no data needed)
#   ./run.sh transport      # synthetic-vs-real transportability battery
#   ./run.sh formulation    # AI de novo pediatric formulation pipeline
#   ./run.sh all            # setup skipped; runs check -> download -> demo
#   ./run.sh help
#
# Run from the project root:
#   cd /media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project
#   chmod +x run.sh
#   ./run.sh setup && ./run.sh check
# =====================================================================
set -euo pipefail

# Resolve project root to THIS script's directory (works from anywhere).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export MALARIA_PROJECT_ROOT="$ROOT"
# Make `import src...` work without installing the package.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Pick python: prefer the activated conda env, else python3.
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

banner () { echo; echo "=========================================================="; echo ">>> $*"; echo "=========================================================="; }

cmd_setup () {
  banner "Creating conda/mamba environment 'malaria'"
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -f envs/environment.yml || mamba env update -f envs/environment.yml
  elif command -v conda >/dev/null 2>&1; then
    conda env create -f envs/environment.yml || conda env update -f envs/environment.yml
  else
    echo "conda/mamba not found. Falling back to pip into the current Python."
    "$PY" -m pip install -r envs/requirements.txt
  fi
  echo "Done. Now activate it:  conda activate malaria"
  echo "Then:  ./run.sh check"
}

cmd_check () {
  banner "Phase 0: init project + environment check"
  "$PY" scripts/00_setup/00_init_project.py
  "$PY" scripts/00_setup/01_check_environment.py
}

cmd_download () {
  banner "Stage 1-6: download all OPEN data + managed-access checklists"
  "$PY" scripts/01_download/download_all.py
}

cmd_download_heavy () {
  banner "Stage 1-6 (HEAVY): also pull large FASTQ + full proteomics"
  "$PY" scripts/01_download/download_all.py --heavy
}

cmd_demo () {
  banner "Self-tests (no real data needed)"
  "$PY" scripts/07_integration_benchmarks/transportability.py --demo
  "$PY" scripts/10_formulation/run_formulation_pipeline.py --pk-target 100
  echo "Results written under: results/"
}

cmd_transport ()   { banner "Synthetic-vs-real transportability"; "$PY" scripts/07_integration_benchmarks/transportability.py --demo; }
cmd_formulation () { banner "AI de novo formulation"; "$PY" scripts/10_formulation/run_formulation_pipeline.py --pk-target 100; }

cmd_all () {
  cmd_check
  cmd_download
  cmd_demo
  banner "Pipeline bootstrap complete"
  echo "Next real-data steps (fill in once data are downloaded):"
  echo "  make qc && make single-omics && make sample-map"
  echo "  make synthetic && make benchmarks && make transportability"
  echo "  make hpamm && make treatment && make formulation && make validation"
}

cmd_help () { sed -n '2,30p' "$ROOT/run.sh"; }

case "${1:-help}" in
  setup)          cmd_setup ;;
  check)          cmd_check ;;
  download)       cmd_download ;;
  download-heavy) cmd_download_heavy ;;
  demo)           cmd_demo ;;
  transport)      cmd_transport ;;
  formulation)    cmd_formulation ;;
  all)            cmd_all ;;
  help|*)         cmd_help ;;
esac
