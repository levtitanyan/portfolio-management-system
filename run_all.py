"""
run_all.py — Master pipeline runner
====================================
Runs the full pipeline from scratch:
    1. Download data
    2. Build features
    3. Prepare splits
    4. Train baselines
    5. Train LSTM
    6. Train GRU
    7. Train TCN
    8. Gather report CSVs
    9. Run portfolio backtest

Usage:
    python run_all.py                  # full pipeline
    python run_all.py --from baselines # skip data steps
    python run_all.py --from lstm      # skip data + baselines
    python run_all.py --from backtest  # only backtest
    python run_all.py --skip-dl        # skip LSTM/GRU/TCN (faster)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("download",   "src/dataset/1_download_yahoo_data.py"),
    ("features",   "src/dataset/2_build_features.py"),
    ("splits",     "src/dataset/3_prepare_model_data.py"),
    ("baselines",  "src/models/4_train_baselines.py"),
    ("lstm",       "src/models/5_train_lstm.py"),
    ("gru",        "src/models/6_train_gru.py"),
    ("tcn",        "src/models/7_train_tcn.py"),
    ("report",     "src/models/99_gather_report.py"),
    ("backtest",   "src/models/portfolio_backtest.py"),
]

DL_STEPS = {"lstm", "gru", "tcn"}


def run(script: str) -> bool:
    path = BASE_DIR / script
    if not path.exists():
        print(f"  [SKIP] {script} — file not found")
        return True

    print(f"\n{'─'*60}")
    print(f"  Running: {script}")
    print(f"{'─'*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(BASE_DIR),
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [FAILED] {script} exited with code {result.returncode}")
        return False

    print(f"\n  [DONE] {script}  ({elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from", dest="start_from", default="download",
        choices=[name for name, _ in STEPS],
        help="Start pipeline from this step (skip earlier steps)"
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Skip LSTM, GRU, TCN training"
    )
    args = parser.parse_args()

    step_names = [name for name, _ in STEPS]
    start_idx  = step_names.index(args.start_from)

    print("\n" + "="*60)
    print("  INTELLIGENT PORTFOLIO MANAGEMENT SYSTEM")
    print("  Full Pipeline Runner")
    print("="*60)
    print(f"  Starting from: {args.start_from}")
    if args.skip_dl:
        print("  Skipping deep learning steps: lstm, gru, tcn")

    total_start = time.time()
    failed = False

    for i, (name, script) in enumerate(STEPS):
        if i < start_idx:
            print(f"  [SKIP] {name}")
            continue

        if args.skip_dl and name in DL_STEPS:
            print(f"  [SKIP] {name} (--skip-dl)")
            continue

        ok = run(script)
        if not ok:
            print(f"\n  Pipeline stopped at: {name}")
            failed = True
            break

    total = time.time() - total_start
    print("\n" + "="*60)
    if failed:
        print("  PIPELINE FAILED — see error above")
    else:
        print("  PIPELINE COMPLETE")
    print(f"  Total time: {total/60:.1f} minutes")
    print("="*60)


if __name__ == "__main__":
    main()
