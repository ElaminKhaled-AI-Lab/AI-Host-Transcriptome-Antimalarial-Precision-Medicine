#!/usr/bin/env python
"""Phase 19 -- one-command runner for the whole nano-DDS formulation-AI arm.

Runs, in order:
  1. run_factorial_dds.py     -- 3^3 factorial over the polymer x drug library
  2. optimize_nano_dds.py     -- AI surrogates + Box-Behnken RSM + Bayesian opt.
  3. validate_and_report.py   -- ICH stability + literature validation + architecture
  4. make_nano_dds_figures.py -- summary figure panel (if matplotlib present)

Usage:
  python scripts/13_nano_dds/run_nano_dds_pipeline.py --drug lumefantrine
"""
import _bootstrap  # noqa: F401
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


FAILED = []


def run(script, *extra, base=None, required=True):
    """Run one step. A non-critical failure is recorded and the pipeline continues,
    so a single broken step does not discard the work that already succeeded."""
    root = base if base is not None else HERE
    cmd = [sys.executable, str(root / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        FAILED.append(script)
        print(f"\n*** step FAILED: {script} (exit {e.returncode}) -- continuing ***\n",
              flush=True)
        if required:
            print("    (later steps that depend on it may also fail)", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug", default="lumefantrine")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--noise-rsd", type=float, default=0.05)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda for the deep-learning step")
    args = ap.parse_args()
    run("run_factorial_dds.py")
    run("analyze_dissolution.py", "--replicates", str(args.replicates),
        "--noise-rsd", str(args.noise_rsd))
    run("optimize_nano_dds.py", "--drug", args.drug)
    run("validate_and_report.py")
    # formulation-omics arm (precision medicine across the whole cohort)
    omics = HERE.parent / "14_omics_formulation"
    run("build_patient_formulation_dataset.py", "--drug", args.drug, base=omics)
    run("benchmark_integration.py", "--device", args.device, base=omics)
    run("personalize_all_patients.py", base=omics)
    try:
        run("make_all_figures.py")
    except Exception as e:  # noqa: BLE001
        print(f"(figures skipped: {e})")
    if shutil.which("Rscript"):
        try:
            print("\n=== rsm_dissolution_analysis.R ===", flush=True)
            subprocess.run(["Rscript", str(HERE / "rsm_dissolution_analysis.R"),
                            "PLGA_50_50", args.drug], check=True)
        except Exception as e:  # noqa: BLE001
            print(f"(R analysis skipped: {e})")
    else:
        print("(R not found; skipping the classical RSM/ANOVA step)")
    if FAILED:
        print("\nPipeline finished WITH FAILURES in: " + ", ".join(FAILED))
        print("Everything else completed; see results/, tables/, reports/, figures/.")
        sys.exit(1)
    print("\nnano-DDS + formulation-omics pipeline complete. "
          "See results/, tables/, reports/, figures/.")


if __name__ == "__main__":
    main()
