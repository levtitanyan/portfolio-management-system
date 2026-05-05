"""
generate_figures.py — Produce poster figures from all available universe reports.

Loads outputs/{universe}/reports/ for every universe that has been run,
then saves 5 publication-quality PNG figures plus copies of the markdown
reports into outputs/figures/ — one dedicated folder for everything
you need for presentations and the capstone poster.

Output:
    outputs/figures/
        figure1_returns.png          — long-only returns bar chart
        figure2_equity_curves.png    — portfolio equity curves
        figure3_risk_return.png      — Sharpe vs return scatter
        figure4_heatmap.png          — per-stock directional accuracy heatmap
        figure5_family_comparison.png— baseline vs deep-learning comparison
        tech30_performance_report.md
        tech30_backtest_report.md
        energy30_performance_report.md
        energy30_backtest_report.md   (only if energy30 has been run)

Run:
    python src/models/generate_figures.py
    python run_all.py --universe tech30   (called automatically as the last step)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parents[2]
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# ── Color palette ──────────────────────────────────────────────────────────────

PAL = {
    "tech30":   "#1D4ED8",
    "energy30": "#C2410C",
    "baseline": "#3B82F6",
    "deep":     "#7C3AED",
    "buyhold":  "#059669",
    "pos":      "#16A34A",
    "neg":      "#DC2626",
    "bg":       "#FFFFFF",
    "text":     "#111827",
    "muted":    "#6B7280",
    "grid":     "#E5E7EB",
}

UNIV_CMAP = {
    "tech30":   ["#1D4ED8", "#2563EB", "#93C5FD"],
    "energy30": ["#C2410C", "#D97706", "#FCD34D"],
}

DEEP_MODELS = {
    "lstm_default", "lstm_tuned",
    "gru_default",  "gru_tuned",
    "tcn_default",  "tcn_tuned",
}

MODEL_LABEL = {
    "historical_mean":   "Hist. Mean",
    "naive_last_return": "Naive Last",
    "linear_regression": "Linear Reg.",
    "ridge_regression":  "Ridge Reg.",
    "random_forest":     "Rand. Forest",
    "lstm_default":      "LSTM",
    "lstm_tuned":        "LSTM (tuned)",
    "gru_default":       "GRU",
    "gru_tuned":         "GRU (tuned)",
    "tcn_default":       "TCN",
    "tcn_tuned":         "TCN (tuned)",
    "buy_and_hold":      "Buy & Hold",
}

MODEL_ORDER = [
    "historical_mean", "naive_last_return", "linear_regression",
    "ridge_regression", "random_forest",
    "lstm_default", "lstm_tuned", "gru_default", "gru_tuned",
    "tcn_default", "tcn_tuned",
]

plt.rcParams.update({
    "figure.facecolor":  PAL["bg"],
    "axes.facecolor":    PAL["bg"],
    "axes.edgecolor":    "#CBD5E1",
    "axes.labelcolor":   PAL["text"],
    "axes.titlecolor":   PAL["text"],
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        PAL["grid"],
    "grid.linewidth":    0.8,
    "text.color":        PAL["text"],
    "xtick.color":       PAL["text"],
    "ytick.color":       PAL["text"],
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.titlesize":    15,
    "axes.labelsize":    13,
    "legend.fontsize":   11,
    "legend.framealpha": 0.95,
    "figure.dpi":        130,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PAL["bg"],
})

HORIZONS   = ["1d", "5d"]
UNIV_NAMES = ["tech30", "energy30"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_universe(universe: str) -> dict | None:
    rdir = BASE_DIR / "outputs" / universe / "reports"
    if not rdir.exists():
        return None
    out: dict = {"rdir": rdir}
    for h in HORIZONS:
        for key, fname in [
            ("model",  f"model_performance_{h}.csv"),
            ("stock",  f"stock_model_performance_{h}.csv"),
            ("equity", f"equity_curves_{h}.csv"),
        ]:
            p = rdir / fname
            out[f"{key}_{h}"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    for h in HORIZONS:
        df = out[f"model_{h}"]
        if not df.empty and "model" in df.columns:
            df["family"] = df["model"].apply(
                lambda m: "Deep Learning" if m in DEEP_MODELS else "Baseline"
            )
    return out


def _drop_errored(df: pd.DataFrame) -> pd.DataFrame:
    err = "backtest_error"
    if err in df.columns:
        return df[df[err].isna() | (df[err].astype(str).str.strip() == "nan")].copy()
    return df.copy()


# ── Figure 1 — Long-only returns bar chart ────────────────────────────────────

def figure1_returns(univs: dict, avail: list[str]) -> None:
    H = "5d"
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 9), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, univ in zip(axes, avail):
        df = _drop_errored(univs[univ][f"model_{H}"])
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        df = df.sort_values("long_only_total_return", ascending=True)
        labels  = [MODEL_LABEL.get(m, m) for m in df["model"]]
        returns = (df["long_only_total_return"] * 100).tolist()
        colors  = [PAL["deep"] if m in DEEP_MODELS else PAL["baseline"]
                   for m in df["model"]]

        bars = ax.barh(labels, returns, color=colors, height=0.62,
                       alpha=0.88, edgecolor="white", linewidth=0.6)

        span = max(abs(r) for r in returns) if returns else 1.0
        for bar, val in zip(bars, returns):
            ha   = "left" if val >= 0 else "right"
            xpos = val + span * 0.02 if val >= 0 else val - span * 0.02
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", va="center", ha=ha,
                    fontsize=10, fontweight="600", color=PAL["text"])

        bh_col = "buy_and_hold_benchmark_total_return"
        if bh_col in df.columns:
            bh = float(df[bh_col].dropna().iloc[0]) * 100
            ax.axvline(bh, color=PAL["buyhold"], linestyle="--",
                       linewidth=2.5, zorder=5)
            ax.text(bh + span * 0.015, len(labels) - 0.5,
                    f"B&H {bh:+.1f}%", color=PAL["buyhold"],
                    fontsize=10, fontweight="600", va="top")

        ax.axvline(0, color=PAL["text"], linewidth=0.7, alpha=0.3)
        col = PAL.get(univ, PAL["text"])
        ax.set_title(f"{univ.upper()}\n5-Day Long-Only Total Return",
                     fontsize=16, fontweight="bold", color=col, pad=14)
        ax.set_xlabel("Total Return (%)", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=PAL["baseline"], label="Baseline Models"),
        mpatches.Patch(color=PAL["deep"],     label="Deep Learning Models"),
        Line2D([0], [0], color=PAL["buyhold"], linestyle="--",
               linewidth=2.5, label="Buy & Hold Benchmark"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=12, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Long-Only Portfolio Returns — All Models",
                 fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_returns.png")
    plt.close()
    print("  figure1_returns.png")


# ── Figure 2 — Equity curves ──────────────────────────────────────────────────

def figure2_equity_curves(univs: dict, avail: list[str]) -> None:
    H     = "5d"
    TOP_K = 5
    TOP_N = 3

    n = len(avail)
    fig, axes = plt.subplots(n, 1, figsize=(16, 6 * n + 1))
    if n == 1:
        axes = [axes]

    for ax, univ in zip(axes, avail):
        eq  = univs[univ][f"equity_{H}"].copy()
        mdf = _drop_errored(univs[univ][f"model_{H}"])

        if eq.empty:
            ax.text(0.5, 0.5,
                    f"No equity curve data for {univ}.\n"
                    f"Run: python run_all.py --universe {univ}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=13, color=PAL["muted"])
            ax.set_title(f"{univ.upper()} — No equity data", fontsize=15,
                         color=PAL.get(univ, PAL["text"]))
            continue

        eq["Date"] = pd.to_datetime(eq["Date"])

        if not mdf.empty and "long_only_total_return" in mdf.columns:
            top_models = (mdf.sort_values("long_only_total_return", ascending=False)
                             ["model"].head(TOP_N).tolist())
        else:
            top_models = [m for m in eq["model"].unique()
                          if m != "buy_and_hold"][:TOP_N]

        palette = UNIV_CMAP.get(univ, ["#333333"] * 3)

        for i, model_name in enumerate(top_models):
            sub = eq[(eq["model"] == model_name) &
                     (eq["strategy"] == "long_only") &
                     (eq["top_k"] == TOP_K)].sort_values("Date")
            if sub.empty:
                sub = eq[(eq["model"] == model_name) &
                         (eq["strategy"] == "long_only")]
                if not sub.empty:
                    sub = sub[sub["top_k"] == sub["top_k"].min()].sort_values("Date")
            if sub.empty:
                continue
            lbl = MODEL_LABEL.get(model_name, model_name)
            ax.plot(sub["Date"], sub["portfolio_value"],
                    linewidth=2.5, color=palette[i % len(palette)],
                    label=lbl, zorder=3)

        bh = eq[eq["model"] == "buy_and_hold"].sort_values("Date")
        if not bh.empty:
            ax.plot(bh["Date"], bh["portfolio_value"],
                    linewidth=2.5, color=PAL["buyhold"], linestyle="--",
                    label="Buy & Hold", zorder=2, alpha=0.9)

        ax.axhline(10_000, color=PAL["muted"], linewidth=1,
                   linestyle=":", alpha=0.55, label="Initial $10,000")

        col = PAL.get(univ, PAL["text"])
        ax.set_title(
            f"{univ.upper()} — Top {TOP_N} Models vs Buy-and-Hold"
            f" (5-Day Long-Only, Top {TOP_K} stocks)",
            fontsize=14, fontweight="bold", color=col, pad=12,
        )
        ax.set_ylabel("Portfolio Value ($)", fontsize=13)
        ax.set_xlabel("Date", fontsize=12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Portfolio Equity Curves — Test Period",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_equity_curves.png")
    plt.close()
    print("  figure2_equity_curves.png")


# ── Figure 3 — Risk-return scatter ────────────────────────────────────────────

def figure3_risk_return(univs: dict, avail: list[str]) -> None:
    H = "5d"

    fig, ax = plt.subplots(figsize=(13, 9))

    MARKERS = {"tech30": "o", "energy30": "^"}
    EDGES   = {"tech30": PAL["tech30"], "energy30": PAL["energy30"]}

    bh_handles: list = []

    for univ in avail:
        df = _drop_errored(univs[univ][f"model_{H}"])
        if df.empty:
            continue

        marker   = MARKERS.get(univ, "o")
        edge_col = EDGES.get(univ, PAL["text"])

        for _, row in df.iterrows():
            m = row["model"]
            x = float(row.get("long_only_sharpe_ratio", np.nan))
            y = float(row.get("long_only_total_return", np.nan)) * 100
            if np.isnan(x) or np.isnan(y):
                continue
            fam_col = PAL["deep"] if m in DEEP_MODELS else PAL["baseline"]
            ax.scatter(x, y, s=190, color=fam_col, marker=marker,
                       edgecolors=edge_col, linewidth=2.5, alpha=0.9, zorder=4)

        df2 = df.copy()
        df2["_r"] = df2["long_only_total_return"] * 100
        df2["_s"] = df2["long_only_sharpe_ratio"]
        df2 = df2.dropna(subset=["_r", "_s"])
        to_label = pd.concat(
            [df2.nlargest(2, "_r"), df2.nsmallest(1, "_r")]
        ).drop_duplicates()
        for _, row in to_label.iterrows():
            lbl = MODEL_LABEL.get(row["model"], row["model"]) + f"\n({univ})"
            ax.annotate(lbl, (float(row["_s"]), float(row["_r"])),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=9, color=PAL["text"], alpha=0.85,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=PAL["grid"], alpha=0.85))

        bh_col = "buy_and_hold_benchmark_total_return"
        if bh_col in df.columns:
            bh  = float(df[bh_col].dropna().iloc[0]) * 100
            col = EDGES.get(univ, PAL["buyhold"])
            ax.axhline(bh, color=col, linestyle=":", linewidth=1.8, alpha=0.65)
            bh_handles.append(
                Line2D([0], [0], color=col, linestyle=":", linewidth=1.8,
                       label=f"{univ} B&H ({bh:+.1f}%)")
            )

    ax.axhline(0, color=PAL["text"], linewidth=0.7, alpha=0.2)
    ax.axvline(0, color=PAL["text"], linewidth=0.7, alpha=0.2)

    fam_handles = [
        mpatches.Patch(color=PAL["baseline"], label="Baseline Models"),
        mpatches.Patch(color=PAL["deep"],     label="Deep Learning Models"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PAL["muted"], markeredgecolor=PAL["tech30"],
               markeredgewidth=2.5, markersize=11, label="Tech30"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=PAL["muted"], markeredgecolor=PAL["energy30"],
               markeredgewidth=2.5, markersize=11, label="Energy30"),
    ]
    ax.legend(handles=fam_handles + bh_handles, loc="upper left",
              fontsize=11, framealpha=0.92)
    ax.set_xlabel("Long-Only Sharpe Ratio", fontsize=14)
    ax.set_ylabel("Long-Only Total Return (%)", fontsize=14)
    ax.set_title("Risk-Return Trade-off — 5-Day Horizon",
                 fontsize=18, fontweight="bold", pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_risk_return.png")
    plt.close()
    print("  figure3_risk_return.png")


# ── Figure 4 — Directional accuracy heatmap ───────────────────────────────────

def figure4_heatmap(univs: dict, avail: list[str]) -> None:
    H = "5d"
    n = len(avail)
    fig, axes = plt.subplots(n, 1, figsize=(24, 5 * n + 1))
    if n == 1:
        axes = [axes]

    cmap = sns.diverging_palette(10, 240, s=85, l=55, as_cmap=True)

    for ax, univ in zip(axes, avail):
        df = univs[univ][f"stock_{H}"].copy()
        if (df.empty or "ticker" not in df.columns
                or "directional_accuracy" not in df.columns):
            ax.text(0.5, 0.5, f"No per-stock data for {univ}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=13)
            continue

        pivot = df.pivot_table(index="model", columns="ticker",
                               values="directional_accuracy", aggfunc="mean")

        order = [m for m in MODEL_ORDER if m in pivot.index]
        order += [m for m in pivot.index if m not in order]
        pivot = pivot.reindex(index=order)

        col_means = pivot.mean(axis=0)
        pivot = pivot[col_means.sort_values(ascending=False).index]

        row_labels = [MODEL_LABEL.get(m, m) for m in pivot.index]

        sns.heatmap(
            pivot * 100,
            ax=ax,
            cmap=cmap,
            center=50,
            vmin=44, vmax=62,
            annot=True, fmt=".1f",
            annot_kws={"size": 7.5, "color": PAL["text"]},
            linewidths=0.4, linecolor=PAL["grid"],
            xticklabels=pivot.columns.tolist(),
            yticklabels=row_labels,
            cbar_kws={"label": "Dir. Accuracy (%)", "shrink": 0.6},
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           ha="right", fontsize=9)
        col = PAL.get(univ, PAL["text"])
        ax.set_title(
            f"{univ.upper()} — Per-Stock Directional Accuracy % "
            f"(5-Day, sorted by predictability)",
            fontsize=14, fontweight="bold", color=col, pad=12,
        )
        ax.set_xlabel("Ticker  (left = most predictable)", fontsize=12)
        ax.set_ylabel("")

    fig.suptitle("Directional Accuracy by Model and Stock",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_heatmap.png")
    plt.close()
    print("  figure4_heatmap.png")


# ── Figure 5 — Baseline vs deep learning comparison ──────────────────────────

def figure5_family_comparison(univs: dict, avail: list[str]) -> None:
    H = "5d"

    METRICS = [
        ("long_only_total_return",    "Long-Only\nReturn (%)",      True),
        ("long_only_sharpe_ratio",    "Long-Only\nSharpe Ratio",    False),
        ("long_only_max_drawdown",    "Max Drawdown (%)",           True),
        ("test_directional_accuracy", "Directional\nAccuracy (%)",  True),
    ]

    families = ["Baseline", "Deep Learning"]

    agg: dict = {}
    for univ in avail:
        df = univs[univ][f"model_{H}"].copy()
        if df.empty:
            continue
        agg[univ] = {}
        for fam in families:
            subset = df[df["family"] == fam]
            agg[univ][fam] = {}
            for col, _, scale in METRICS:
                if col in subset.columns:
                    val = float(subset[col].mean())
                    agg[univ][fam][col] = val * 100 if scale else val
                else:
                    agg[univ][fam][col] = np.nan

    if not agg:
        print("  figure5 skipped — no data")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 8))

    for ax, (col, title, scale) in zip(axes, METRICS):
        x      = np.arange(len(families))
        n_univ = len(avail)
        width  = 0.7 / n_univ

        for i, univ in enumerate(avail):
            if univ not in agg:
                continue
            vals   = [agg[univ].get(fam, {}).get(col, np.nan) for fam in families]
            clr    = PAL.get(univ, "#888888")
            offset = (i - (n_univ - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width, label=univ.upper(),
                            color=clr, alpha=0.85, edgecolor="white", linewidth=0.6)

            for bar, val in zip(bars, vals):
                if np.isnan(val):
                    continue
                abs_vals = [v for v in vals if not np.isnan(v)]
                yoff = max(abs(v) for v in abs_vals) * 0.03 if abs_vals else 0.01
                ya   = bar.get_height() + yoff if val >= 0 else bar.get_height() - yoff
                va   = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, ya,
                        f"{val:.1f}", ha="center", va=va,
                        fontsize=9.5, fontweight="600", color=PAL["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(families, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axhline(0, color=PAL["text"], linewidth=0.7, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if scale:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
            )

    handles = [
        mpatches.Patch(color=PAL.get(u, "#888"), label=u.upper())
        for u in avail
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(avail),
               fontsize=13, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Baseline vs Deep Learning — Average Metrics (5-Day Horizon)",
        fontsize=18, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_family_comparison.png")
    plt.close()
    print("  figure5_family_comparison.png")


# ── Copy markdown reports into figures folder ─────────────────────────────────

def copy_reports(avail: list[str]) -> None:
    for univ in avail:
        md_dir = BASE_DIR / "outputs" / univ / "reports" / "md"
        for md_file in md_dir.glob("*.md"):
            dest = FIGURES_DIR / f"{univ}_{md_file.name}"
            shutil.copy2(md_file, dest)
            print(f"  {dest.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    univs = {}
    for u in UNIV_NAMES:
        d = load_universe(u)
        if d is not None:
            univs[u] = d

    avail = list(univs.keys())

    if not avail:
        print("[SKIP] No universe reports found. Run the pipeline first.")
        return

    print(f"\n  Universes loaded: {avail}")
    print(f"  Saving to: {FIGURES_DIR.relative_to(BASE_DIR)}/\n")

    figure1_returns(univs, avail)
    figure2_equity_curves(univs, avail)
    figure3_risk_return(univs, avail)
    figure4_heatmap(univs, avail)
    figure5_family_comparison(univs, avail)
    copy_reports(avail)

    print(f"\n  All outputs saved to outputs/figures/")
    for f in sorted(FIGURES_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
