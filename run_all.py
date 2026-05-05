"""
run_all.py — Master pipeline runner
====================================
Runs the full pipeline from scratch for a chosen stock universe:

    Steps:
        1. download    — Download raw price data from Yahoo Finance
        2. features    — Build technical, market, and sector features
        3. splits      — Split into train / validation / test sets
        4. baselines   — Train statistical and ML baseline models
        5. lstm        — Train LSTM (default + tuned)
        6. gru         — Train GRU (default + tuned)
        7. tcn         — Train TCN (default + tuned)
        8. refresh     — Re-run backtests from saved predictions
        9. report      — Gather all metrics into report tables
        10. backtest   — Full portfolio simulation with equity curves

    Outputs per universe:
        data/{universe}/           — raw CSVs, features, splits
        outputs/{universe}/        — model weights, predictions, metrics
        outputs/{universe}/reports/    — CSV performance tables
        outputs/{universe}/reports/md/ — Markdown reports (human-readable)

Usage:
    python run_all.py                           # full pipeline, tech30
    python run_all.py --universe energy30       # full pipeline, energy30
    python run_all.py --from baselines          # skip data steps
    python run_all.py --from lstm               # skip data + baselines
    python run_all.py --from backtest           # only portfolio backtest
    python run_all.py --skip-dl                 # skip LSTM / GRU / TCN
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("download",  "src/dataset/1_download_yahoo_data.py"),
    ("features",  "src/dataset/2_build_features.py"),
    ("splits",    "src/dataset/3_prepare_model_data.py"),
    ("baselines", "src/models/4_train_baselines.py"),
    ("lstm",      "src/models/5_train_lstm.py"),
    ("gru",       "src/models/6_train_gru.py"),
    ("tcn",       "src/models/7_train_tcn.py"),
    ("refresh",   "src/models/98_refresh_backtests.py"),
    ("report",    "src/models/99_gather_report.py"),
    ("backtest",  "src/models/portfolio_backtest.py"),
    ("figures",   "src/models/generate_figures.py"),
]

DL_STEPS = {"lstm", "gru", "tcn"}


def migrate_legacy_data(universe: str) -> None:
    """
    Move flat data/ and outputs/ from the original single-universe layout
    into data/tech30/ and outputs/tech30/ so the new structure is consistent.
    Only runs once — skipped if destination already exists.
    """
    if universe != "tech30":
        return

    old_raw  = BASE_DIR / "data" / "raw"
    new_raw  = BASE_DIR / "data" / "tech30" / "raw"
    old_spl  = BASE_DIR / "data" / "splits"
    new_spl  = BASE_DIR / "data" / "tech30" / "splits"
    old_feat = BASE_DIR / "data" / "final_model_dataset.csv"
    new_feat = BASE_DIR / "data" / "tech30" / "final_model_dataset.csv"

    migrated = []
    if old_raw.exists() and not new_raw.exists():
        new_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_raw), str(new_raw))
        migrated.append("data/raw → data/tech30/raw")
    if old_spl.exists() and not new_spl.exists():
        new_spl.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_spl), str(new_spl))
        migrated.append("data/splits → data/tech30/splits")
    if old_feat.exists() and not new_feat.exists():
        new_feat.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_feat), str(new_feat))
        migrated.append("data/final_model_dataset.csv → data/tech30/")

    # outputs
    for folder in ["baselines", "lstm", "gru", "tcn", "shared", "reports"]:
        old = BASE_DIR / "outputs" / folder
        new = BASE_DIR / "outputs" / "tech30" / folder
        if old.exists() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            migrated.append(f"outputs/{folder} → outputs/tech30/{folder}")

    if migrated:
        print("\n  [MIGRATE] Moved legacy data to universe subfolder:")
        for m in migrated:
            print(f"    {m}")


def run(script: str, universe: str) -> bool:
    path = BASE_DIR / script
    if not path.exists():
        print(f"  [SKIP] {script} — file not found")
        return True

    print(f"\n{'─'*60}")
    print(f"  Running : {script}")
    print(f"  Universe: {universe}")
    print(f"{'─'*60}")
    t0 = time.time()

    env = {**os.environ, "PORTFOLIO_UNIVERSE": universe}
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(BASE_DIR),
        env=env,
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [FAILED] {script} exited with code {result.returncode}")
        return False

    print(f"\n  [DONE] {script}  ({elapsed:.0f}s)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe", default="tech30",
        choices=["tech30", "energy30"],
        help="Stock universe to use (default: tech30)",
    )
    parser.add_argument(
        "--from", dest="start_from", default="download",
        choices=[name for name, _ in STEPS],
        help="Start pipeline from this step (skip earlier steps)",
    )
    parser.add_argument(
        "--skip-dl", action="store_true",
        help="Skip LSTM, GRU, and TCN training steps",
    )
    args = parser.parse_args()

    step_names = [name for name, _ in STEPS]
    start_idx  = step_names.index(args.start_from)

    print("\n" + "=" * 60)
    print("  INTELLIGENT PORTFOLIO MANAGEMENT SYSTEM")
    print("  Full Pipeline Runner")
    print("=" * 60)
    print(f"  Universe     : {args.universe}")
    print(f"  Starting from: {args.start_from}")
    if args.skip_dl:
        print("  Skipping     : lstm, gru, tcn  (--skip-dl)")
    print(f"\n  Data   → data/{args.universe}/")
    print(f"  Outputs→ outputs/{args.universe}/")
    print(f"  Reports→ outputs/{args.universe}/reports/md/")

    # Migrate old flat layout to universe subfolders (one-time, tech30 only)
    migrate_legacy_data(args.universe)

    total_start = time.time()
    failed = False

    for i, (name, script) in enumerate(STEPS):
        if i < start_idx:
            print(f"  [SKIP] {name}")
            continue
        if args.skip_dl and name in DL_STEPS:
            print(f"  [SKIP] {name}  (--skip-dl)")
            continue
        ok = run(script, args.universe)
        if not ok:
            print(f"\n  Pipeline stopped at: {name}")
            failed = True
            break

    total = time.time() - total_start
    print("\n" + "=" * 60)
    if failed:
        print("  PIPELINE FAILED — see error above")
    else:
        print("  PIPELINE COMPLETE")
        print(f"\n  All figures + reports → outputs/figures/")
        print(f"  Per-universe CSVs    → outputs/{args.universe}/reports/")
    print(f"\n  Total time: {total / 60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
