#!/usr/bin/env python3
"""
poster_figures.py  —  5 publication-quality poster visualizations.
Run:  python poster_figures.py
Saves 300-dpi PNGs to outputs/figures/
"""
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm

BASE   = Path(__file__).resolve().parent
OUTDIR = BASE / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY   = "#1D3557"
BLUE   = "#2563EB"
CORAL  = "#E05252"
AMBER  = "#D97706"
PURPLE = "#7C3AED"
MINT   = "#16A34A"
GRAY   = "#6B7280"
LGRAY  = "#F3F4F6"

UNI_COL = {"tech30": NAVY,  "energy30": CORAL}
UNI_LAB = {"tech30": "Tech-30", "energy30": "Energy-30"}
UNI_MRK = {"tech30": "o",   "energy30": "D"}

MDL_COL = {
    "historical_mean":   GRAY,
    "linear_regression": GRAY,
    "ridge_regression":  BLUE,
    "random_forest":     AMBER,
    "lstm_default":      NAVY,
    "gru_default":       CORAL,
    "tcn_default":       PURPLE,
    "buy_and_hold":      MINT,
}
MDL_LAB = {
    "historical_mean":   "Hist. Mean",
    "linear_regression": "Linear Reg.",
    "ridge_regression":  "Ridge Reg.",
    "random_forest":     "Random Forest",
    "lstm_default":      "LSTM",
    "gru_default":       "GRU",
    "tcn_default":       "TCN",
    "buy_and_hold":      "Buy & Hold",
}
HORIZONS = ["1d", "5d", "10d", "30d"]
H_LAB    = {"1d": "1-Day", "5d": "5-Day", "10d": "10-Day", "30d": "30-Day"}
FOCUS    = ["historical_mean", "ridge_regression", "random_forest",
            "lstm_default", "gru_default", "tcn_default"]
F_LAB    = ["Hist. Mean", "Ridge", "Rand. Forest", "LSTM", "GRU", "TCN"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.color":        "#B0B0B0",
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titlepad":     10,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "legend.fontsize":   9,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#DDDDDD",
})


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_metrics(uni: str) -> pd.DataFrame:
    frames = []
    for h in HORIZONS:
        p = BASE / "outputs" / uni / "reports" / f"metrics_{h}.csv"
        if p.exists():
            d = pd.read_csv(p); d["universe"] = uni; frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_equity(uni: str, h: str) -> pd.DataFrame:
    p = BASE / "outputs" / uni / "reports" / f"equity_curves_{h}.csv"
    d = pd.read_csv(p, parse_dates=["Date"]); d["universe"] = uni
    return d


def best_lo(df: pd.DataFrame, metric: str = "sharpe_ratio") -> pd.DataFrame:
    lo = df[df["strategy"] == "long_only"].copy()
    return (lo.sort_values(metric, ascending=False)
              .groupby(["model", "horizon", "universe"], as_index=False)
              .first())


# ── Figure 1  ─ Equity Curves: Best Strategy vs Buy & Hold ────────────────────
def fig1_equity_curves():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Portfolio Growth: Best Long-Only Strategies vs Buy & Hold  (2024–2026, after transaction costs)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, uni in zip(axes, ["tech30", "energy30"]):
        met = load_metrics(uni)
        bl  = best_lo(met)

        # pick best horizon overall (by mean sharpe across FOCUS models)
        h_sharpe = (bl[bl["model"].isin(FOCUS)]
                    .groupby("horizon")["sharpe_ratio"].mean()
                    .sort_values(ascending=False))
        best_h = h_sharpe.index[0] if not h_sharpe.empty else "10d"

        eq = load_equity(uni, best_h)

        # best DL model at best_h
        dl_row = (bl[(bl["model"].isin(["lstm_default","gru_default","tcn_default"]))
                     & (bl["horizon"] == best_h)]
                  .sort_values("sharpe_ratio", ascending=False).iloc[:1])
        # best baseline at best_h
        bs_row = (bl[(bl["model"].isin(["ridge_regression","random_forest","historical_mean"]))
                     & (bl["horizon"] == best_h)]
                  .sort_values("sharpe_ratio", ascending=False).iloc[:1])

        show = []
        for row in [dl_row, bs_row]:
            if not row.empty:
                show.append((row.iloc[0]["model"], int(row.iloc[0]["top_k"])))

        # B&H
        bh = eq[(eq["model"] == "buy_and_hold")].sort_values("Date")
        ax.plot(bh["Date"], bh["portfolio_value"],
                color=MINT, lw=2.5, ls="--", zorder=3, label="Buy & Hold")
        bh_end = bh["portfolio_value"].iloc[-1]
        ax.annotate(f"${bh_end:,.0f}",
                    xy=(bh["Date"].iloc[-1], bh_end),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=MINT, va="center", fontweight="bold")

        for model, topk in show:
            sub = eq[(eq["model"] == model) &
                     (eq["strategy"] == "long_only") &
                     (eq["top_k"] == topk)].sort_values("Date")
            if sub.empty:
                continue
            col   = MDL_COL[model]
            label = f"{MDL_LAB[model]} (top-{topk})"
            ax.plot(sub["Date"], sub["portfolio_value"],
                    color=col, lw=2.5, zorder=5, label=label)
            end = sub["portfolio_value"].iloc[-1]
            ret = (end / 10000 - 1) * 100
            ax.annotate(f"${end:,.0f}  ({ret:+.0f}%)",
                        xy=(sub["Date"].iloc[-1], end),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=col, va="center", fontweight="bold")

        ax.axhline(10_000, color="#AAAAAA", lw=0.8, ls=":")
        ax.text(eq["Date"].min(), 10_050, "Starting $10,000",
                fontsize=7.5, color="#AAAAAA")
        ax.set_title(f"{UNI_LAB[uni]} Universe  ·  {H_LAB[best_h]} Horizon",
                     fontweight="bold", color=UNI_COL[uni])
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(loc="upper left")

    plt.tight_layout()
    out = OUTDIR / "figure1_equity_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Figure 2  ─ The Horizon Effect: Sharpe Ratio by Prediction Horizon ────────
def fig2_horizon_effect():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle(
        "The Horizon Effect: How Prediction Window Affects Risk-Adjusted Returns (Sharpe Ratio)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    x = np.arange(len(HORIZONS))

    for ax, uni in zip(axes, ["tech30", "energy30"]):
        met   = load_metrics(uni)
        bl    = best_lo(met)
        bh_sh = met[met["model"] == "buy_and_hold"]["sharpe_ratio"].mean()

        for model in FOCUS:
            rows = bl[bl["model"] == model].copy()
            rows["h_idx"] = rows["horizon"].map({h: i for i, h in enumerate(HORIZONS)})
            rows = rows.sort_values("h_idx")
            if rows.empty:
                continue
            col  = MDL_COL[model]
            vals = [rows[rows["horizon"] == h]["sharpe_ratio"].values[0]
                    if not rows[rows["horizon"] == h].empty else np.nan
                    for h in HORIZONS]
            lw   = 2.5 if model in ("lstm_default","gru_default","tcn_default") else 1.8
            ls   = "-" if model in ("lstm_default","gru_default","tcn_default") else "--"
            ms   = 8  if model in ("lstm_default","gru_default","tcn_default") else 6
            mk   = ("o" if model == "lstm_default" else
                    "s" if model == "gru_default" else
                    "^" if model == "tcn_default" else "D")
            ax.plot(x, vals, color=col, lw=lw, ls=ls, marker=mk,
                    markersize=ms, label=MDL_LAB[model], zorder=4)

        # B&H reference band
        ax.axhline(bh_sh, color=MINT, lw=2, ls=":", zorder=3)
        ax.text(3.05, bh_sh, f"B&H Sharpe\n{bh_sh:.2f}",
                fontsize=8, color=MINT, va="center", fontweight="bold")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="-")
        ax.fill_between([-0.3, 3.3], 0, bh_sh, alpha=0.04, color=MINT, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([H_LAB[h] for h in HORIZONS])
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Sharpe Ratio (long-only, best top-K)")
        ax.set_title(f"{UNI_LAB[uni]} Universe", fontweight="bold",
                     color=UNI_COL[uni])
        ax.set_xlim(-0.3, 3.5)
        ax.legend(loc="upper left", ncol=2)

    plt.tight_layout()
    out = OUTDIR / "figure2_horizon_effect.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Figure 3  ─ Risk-Return Scatter: All Strategies Mapped ────────────────────
def fig3_risk_return():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(
        "Risk vs. Return: All Long-Only Strategies Across Horizons and Top-K Selections",
        fontsize=14, fontweight="bold", pad=12,
    )

    all_data = pd.concat([load_metrics(u) for u in ["tech30","energy30"]], ignore_index=True)
    lo = all_data[(all_data["strategy"] == "long_only") &
                  (all_data["model"].isin(FOCUS))].copy()

    horizon_markers = {"1d": "X", "5d": "o", "10d": "s", "30d": "D"}
    horizon_sizes   = {"1d": 50,  "5d": 80,  "10d": 100, "30d": 90}

    # scatter by model colour × horizon marker × universe edge
    for (model, h), grp in lo.groupby(["model", "horizon"]):
        for uni, ugrp in grp.groupby("universe"):
            if ugrp.empty:
                continue
            ec = "black" if uni == "tech30" else "#888888"
            lw = 1.2 if uni == "tech30" else 0.7
            ax.scatter(
                ugrp["annualized_return"] * 100,
                ugrp["sharpe_ratio"],
                c=MDL_COL[model],
                marker=horizon_markers[h],
                s=horizon_sizes[h],
                edgecolors=ec, linewidths=lw,
                alpha=0.80, zorder=4,
            )

    # B&H reference crosses for each universe
    bh_vals = {"tech30": (20.2, 1.413), "energy30": (17.8, 0.464)}
    for uni, (ret, sh) in bh_vals.items():
        ax.scatter(ret, sh, marker="*", s=350, color=MINT,
                   edgecolors="black", linewidths=1, zorder=8,
                   label=f"Buy & Hold — {UNI_LAB[uni]}" if uni == "tech30"
                   else f"Buy & Hold — {UNI_LAB[uni]}")
        ax.annotate(f"B&H\n{UNI_LAB[uni]}",
                    xy=(ret, sh), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=MINT, fontweight="bold")

    ax.axhline(0, color="#AAAAAA", lw=0.9)
    ax.axvline(0, color="#AAAAAA", lw=0.9)

    # model legend
    model_handles = [mpatches.Patch(color=MDL_COL[m], label=MDL_LAB[m]) for m in FOCUS]
    model_legend  = ax.legend(handles=model_handles, title="Model",
                              loc="upper left", framealpha=0.95)
    ax.add_artist(model_legend)

    # horizon legend
    h_handles = [Line2D([0],[0], marker=horizon_markers[h], color="gray",
                        markersize=7, linestyle="None",
                        label=H_LAB[h]) for h in HORIZONS]
    ax.legend(handles=h_handles, title="Horizon", loc="lower right", framealpha=0.95)

    # universe legend (edge style)
    uni_handles = [
        Line2D([0],[0], marker="o", color="gray", markersize=8,
               markeredgecolor="black", markeredgewidth=1.2,
               linestyle="None", label="Tech-30"),
        Line2D([0],[0], marker="o", color="gray", markersize=8,
               markeredgecolor="#888888", markeredgewidth=0.7,
               linestyle="None", label="Energy-30"),
    ]
    uni_legend = ax.legend(handles=uni_handles, title="Universe",
                           loc="upper right", framealpha=0.95)
    ax.add_artist(uni_legend)

    ax.set_xlabel("Annualized Return (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    out = OUTDIR / "figure3_risk_return.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Figure 4  ─ Sharpe Ratio Heatmap: Model × Horizon ────────────────────────
def fig4_sharpe_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sharpe Ratio Heatmap: Which Model × Horizon Combinations Beat Buy & Hold?",
        fontsize=14, fontweight="bold", y=1.02,
    )

    bh_sharpes = {"tech30": 1.413, "energy30": 0.464}

    for ax, uni in zip(axes, ["tech30", "energy30"]):
        met = load_metrics(uni)
        bl  = best_lo(met)
        bh_sh = bh_sharpes[uni]

        # pivot: rows = FOCUS models, cols = horizons
        pivot = pd.DataFrame(index=F_LAB, columns=[H_LAB[h] for h in HORIZONS], dtype=float)
        for model, label in zip(FOCUS, F_LAB):
            for h in HORIZONS:
                row = bl[(bl["model"] == model) & (bl["horizon"] == h)]
                if not row.empty:
                    pivot.loc[label, H_LAB[h]] = float(row.iloc[0]["sharpe_ratio"])

        data = pivot.values.astype(float)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        vcenter = bh_sh
        vcenter = np.clip(vcenter, vmin + 0.01, vmax - 0.01)
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlBu

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Sharpe Ratio", fontsize=9)
        cb.ax.axhline(bh_sh, color="black", lw=1.5, ls="--")
        cb.ax.text(1.08, (bh_sh - vmin) / (vmax - vmin),
                   f"B&H\n{bh_sh:.2f}", transform=cb.ax.transAxes,
                   fontsize=7.5, va="center", color="black")

        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([H_LAB[h] for h in HORIZONS])
        ax.set_yticks(range(len(F_LAB)))
        ax.set_yticklabels(F_LAB)
        ax.set_xlabel("Prediction Horizon")

        # annotate cells
        for i in range(len(F_LAB)):
            for j in range(len(HORIZONS)):
                val = data[i, j]
                if not np.isnan(val):
                    star = "★" if val > bh_sh else ""
                    txt  = f"{val:.2f}{star}"
                    tc   = "white" if abs(val - vcenter) > (vmax - vmin) * 0.35 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8.5, color=tc, fontweight="bold")

        ax.set_title(f"{UNI_LAB[uni]} Universe\n(★ = beats Buy & Hold Sharpe)",
                     fontweight="bold", color=UNI_COL[uni])
        ax.grid(False)

    plt.tight_layout()
    out = OUTDIR / "figure4_sharpe_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Figure 5  ─ Universe Comparison: Tech-30 vs Energy-30 ─────────────────────
def fig5_universe_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Tech-30 vs Energy-30: Best Sharpe Ratio and Maximum Drawdown per Model Family",
        fontsize=14, fontweight="bold", y=1.02,
    )

    all_data = pd.concat([load_metrics(u) for u in ["tech30","energy30"]], ignore_index=True)
    bl       = best_lo(all_data)
    bl       = bl[bl["model"].isin(FOCUS)]
    best     = (bl.sort_values("sharpe_ratio", ascending=False)
                  .groupby(["model", "universe"], as_index=False)
                  .first())

    x      = np.arange(len(FOCUS))
    width  = 0.35
    unis   = ["tech30", "energy30"]
    offset = [-width/2, width/2]

    bh_sharpe = {"tech30": 1.413, "energy30": 0.464}
    bh_mdd    = {"tech30": -17.7, "energy30": -25.6}

    # LEFT: Sharpe ratio
    ax = axes[0]
    for i, (uni, off) in enumerate(zip(unis, offset)):
        vals = [best[(best["model"] == m) & (best["universe"] == uni)]["sharpe_ratio"].values
                for m in FOCUS]
        vals = [float(v[0]) if len(v) else np.nan for v in vals]
        bars = ax.bar(x + off, vals, width, label=UNI_LAB[uni],
                      color=UNI_COL[uni], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color=UNI_COL[uni])

    ax.axhline(bh_sharpe["tech30"], color=NAVY,  lw=1.5, ls="--", alpha=0.7,
               label=f"B&H Tech-30 ({bh_sharpe['tech30']:.2f})")
    ax.axhline(bh_sharpe["energy30"], color=CORAL, lw=1.5, ls=":",  alpha=0.7,
               label=f"B&H Energy-30 ({bh_sharpe['energy30']:.2f})")
    ax.axhline(0, color="#AAAAAA", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(F_LAB, rotation=15, ha="right")
    ax.set_ylabel("Sharpe Ratio  (best horizon & top-K)")
    ax.set_title("Risk-Adjusted Performance (Sharpe Ratio)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # RIGHT: Max Drawdown
    ax = axes[1]
    for i, (uni, off) in enumerate(zip(unis, offset)):
        vals = [best[(best["model"] == m) & (best["universe"] == uni)]["max_drawdown"].values
                for m in FOCUS]
        vals = [float(v[0]) * 100 if len(v) else np.nan for v in vals]
        bars = ax.bar(x + off, vals, width, label=UNI_LAB[uni],
                      color=UNI_COL[uni], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val - 0.5,
                        f"{val:.1f}%", ha="center", va="top", fontsize=7.5,
                        fontweight="bold", color=UNI_COL[uni])

    ax.axhline(bh_mdd["tech30"],   color=NAVY,  lw=1.5, ls="--", alpha=0.7,
               label=f"B&H Tech-30 ({bh_mdd['tech30']:.1f}%)")
    ax.axhline(bh_mdd["energy30"], color=CORAL, lw=1.5, ls=":",  alpha=0.7,
               label=f"B&H Energy-30 ({bh_mdd['energy30']:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(F_LAB, rotation=15, ha="right")
    ax.set_ylabel("Maximum Drawdown (%)  — smaller is better")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Downside Risk (Maximum Drawdown)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = OUTDIR / "figure5_universe_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Generating poster figures …\n")
    fig1_equity_curves()
    fig2_horizon_effect()
    fig3_risk_return()
    fig4_sharpe_heatmap()
    fig5_universe_comparison()
    print(f"\n  All figures saved to: {OUTDIR}\n")
