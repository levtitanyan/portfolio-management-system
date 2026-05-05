"""
clean.py — Delete all generated files so the pipeline can re-run from scratch.

By default keeps the raw Yahoo Finance downloads (slow to re-fetch).
Pass --all to wipe everything including raw CSVs.

Usage:
    python clean.py                      # delete outputs + processed data, keep raw downloads
    python clean.py --all                # delete everything including raw downloads
    python clean.py --universe tech30    # only that universe (default: both)
"""

import argparse
import shutil
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
UNIVERSES = ["tech30", "energy30"]

# Old flat-layout folders created before the multi-universe structure was added.
# These must also be removed so the pipeline starts clean.
LEGACY_OUTPUT_DIRS = [
    "baselines", "lstm", "gru", "tcn", "shared", "reports", "portfolio",
]
LEGACY_DATA_ITEMS = [
    "processed", "splits", "final_model_dataset.csv",
]


def remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"  deleted  {path.relative_to(BASE_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean pipeline outputs")
    parser.add_argument(
        "--universe", choices=UNIVERSES + ["all"], default="all",
        help="Universe to clean (default: all)",
    )
    parser.add_argument(
        "--all", dest="delete_raw", action="store_true",
        help="Also delete raw Yahoo Finance downloads (pipeline will re-download)",
    )
    args = parser.parse_args()

    targets = UNIVERSES if args.universe == "all" else [args.universe]

    print(f"\n  Cleaning: {targets}  |  delete raw: {args.delete_raw}\n")

    # ── Per-universe outputs ───────────────────────────────────────────────────
    for universe in targets:
        remove(BASE_DIR / "outputs" / universe)

        if args.delete_raw:
            remove(BASE_DIR / "data" / universe)
        else:
            data_dir = BASE_DIR / "data" / universe
            remove(data_dir / "final_model_dataset.csv")
            remove(data_dir / "processed")
            remove(data_dir / "splits")

    # ── Legacy flat-layout outputs (pre-universe structure) ────────────────────
    for name in LEGACY_OUTPUT_DIRS:
        remove(BASE_DIR / "outputs" / name)

    # ── Legacy flat-layout data ────────────────────────────────────────────────
    for name in LEGACY_DATA_ITEMS:
        remove(BASE_DIR / "data" / name)

    if args.delete_raw:
        remove(BASE_DIR / "data" / "raw")

    # ── Shared figures folder ──────────────────────────────────────────────────
    remove(BASE_DIR / "outputs" / "figures")

    # ── Stale PNGs in project root ─────────────────────────────────────────────
    for png in BASE_DIR.glob("figure*.png"):
        png.unlink()
        print(f"  deleted  {png.name}")

    # ── Remove root outputs/ and data/ if now empty ────────────────────────────
    for root in [BASE_DIR / "outputs", BASE_DIR / "data"]:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
            print(f"  deleted  {root.relative_to(BASE_DIR)}/")

    print("\n  Done. Both outputs/ and data/ are gone.")
    if args.delete_raw:
        print("\n  Now run:")
        print("    python run_all.py --universe tech30")
        print("    python run_all.py --universe energy30")
    else:
        print("\n  Raw downloads kept. Now run:")
        print("    python run_all.py --from features --universe tech30")
        print("    python run_all.py --from features --universe energy30")


if __name__ == "__main__":
    main()
