from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _set_style():
    mpl.rcParams.update({
        "figure.figsize": (6.0, 3.2),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "font.size": 10,
    })


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure df column is numeric; allow "inf" strings, convert to np.inf
    df = df.copy()
    df["df"] = df["df"].apply(lambda v: np.inf if (isinstance(v, str) and v.lower() == "inf") else v)
    df["df"] = df["df"].astype(float)
    # Sort degrees of freedom with finite first, then ∞
    df["df_sortkey"] = df["df"].apply(lambda x: (0, x) if np.isfinite(x) else (1, np.inf))
    return df.sort_values(["method", "df_sortkey"])


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, as_index=False)
          .agg(mse_mean=("mse", "mean"),
               mse_std=("mse", "std"),
               n=("mse", "count"))
    )
    g["mse_se"] = g["mse_std"] / np.sqrt(g["n"].clip(lower=1))
    # 95% CI using normal approx
    g["mse_lo"] = g["mse_mean"] - 1.96 * g["mse_se"]
    g["mse_hi"] = g["mse_mean"] + 1.96 * g["mse_se"]
    return g


def _df_ticklabels(df_vals: np.ndarray) -> list[str]:
    labels = []
    for v in df_vals:
        if np.isfinite(v):
            # show as integer when close
            labels.append(str(int(v)) if abs(v - int(v)) < 1e-9 else f"{v:g}")
        else:
            labels.append("∞")
    return labels


def plot_mse_vs_df(df: pd.DataFrame, outpath: Path):
    _set_style()
    df = _clean_df(df)
    agg = _aggregate(df, ["method", "df"])

    # x order
    x_vals = np.sort(agg["df"].unique(), kind="mergesort")
    # Put finite first, then ∞
    x_vals = np.array(sorted(x_vals, key=lambda v: (not np.isfinite(v), v)))
    x_labels = _df_ticklabels(x_vals)
    x_pos = np.arange(len(x_vals))  # categorical positions 0..K-1


    # Map method -> series
    methods = ["OLS", "LAD", "Huber"]
    fig, ax = plt.subplots()

    for m in methods:
        sub = agg[agg["method"] == m].set_index("df").reindex(x_vals)
        y = sub["mse_mean"].values
        lo = sub["mse_lo"].values
        hi = sub["mse_hi"].values

        ax.plot(x_pos, y, marker="o", label=m)
        ax.fill_between(x_pos, lo, hi, alpha=0.2)

    ax.set_xlabel("Degrees of freedom (t distribution)")
    ax.set_ylabel("Coefficient MSE (mean ± 95% CI)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title("MSE vs Tail Heaviness")
    ax.legend(ncols=3, loc="upper right")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def plot_small_multiples(df: pd.DataFrame, facet_by: str, outdir: Path):
    """
    facet_by: one of {"snr", "gamma"} to produce small multiples panes.
    """
    assert facet_by in {"snr", "gamma"}
    _set_style()
    df = _clean_df(df)

    panels = sorted(df[facet_by].unique())
    for val in panels:
        sub = df[df[facet_by] == val]
        if sub.empty:
            continue
        agg = _aggregate(sub, ["method", "df"])
        x_vals = np.sort(agg["df"].unique(), kind="mergesort")
        # Put finite first, then ∞
        x_vals = np.array(sorted(x_vals, key=lambda v: (not np.isfinite(v), v)))
        x_labels = _df_ticklabels(x_vals)
        x_pos = np.arange(len(x_vals))  # categorical positions 0..K-1

        fig, ax = plt.subplots()
        for m in ["OLS", "LAD", "Huber"]:
            sub = agg[agg["method"] == m].set_index("df").reindex(x_vals)
            y = sub["mse_mean"].values
            lo = sub["mse_lo"].values
            hi = sub["mse_hi"].values

            ax.plot(x_pos, y, marker="o", label=m)
            ax.fill_between(x_pos, lo, hi, alpha=0.2)

        pretty = f"{facet_by.upper()}={val:g}" if isinstance(val, (float, np.floating)) else f"{facet_by.upper()}={val}"
        ax.set_title(f"MSE vs df — {pretty}")
        ax.set_xlabel("Degrees of freedom (t distribution)")
        ax.set_ylabel("Coefficient MSE (mean ± 95% CI)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(ncols=3, loc="upper right")
        fig.tight_layout()

        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"mse_vs_df_by_{facet_by}_{str(val).replace('.', 'p').replace(' ', '')}.png"
        fig.savefig(outdir / fname)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize robust regression simulation results.")
    ap.add_argument("--input", type=Path, default=Path("results/data/simulation_results.csv"))
    ap.add_argument("--out-main", type=Path, default=Path("results/figures/mse_vs_df.png"))
    ap.add_argument("--small-multiples", action="store_true", help="Also save faceted plots by SNR and gamma.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # Basic sanity: require columns
    required = {"method", "df", "mse"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")

    # Main figure
    plot_mse_vs_df(df, args.out_main)

    if args.small_multiples:
        plot_small_multiples(df, facet_by="snr",   outdir=Path("results/figures/small_multiples_snr"))
        plot_small_multiples(df, facet_by="gamma", outdir=Path("results/figures/small_multiples_gamma"))

    print("Saved main figure to:", args.out_main)
    if args.small_multiples:
        print("Saved small multiples into results/figures/small_multiples_{snr,gamma}/")


if __name__ == "__main__":
    main()
