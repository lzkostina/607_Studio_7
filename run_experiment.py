##################### Parameters to vary ############################
## Tail heaviness: tails of error distribution F (Student-t degrees of freedom) dfs
## Aspect ratio: \gamma = p/n controls the dimensionality regime \gamma s
## Design structure: correlation among predictors \rho s
## Signal-to-noise ratio: overall signal strength snrs
#####################################################################

from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.setup import generate_full
from src.estimators import fit_ols, fit_lad, fit_huber
from src.evaluation import mse_beta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robust regression simulation study.")
    p.add_argument("--n", type=int, default=200, help="Base sample size (rows).")
    p.add_argument("--dfs", type=float, nargs="+", default=[1, 2, 3, 20, math.inf],
                   help="Degrees of freedom for t errors; use 'inf' for Gaussian.")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                   help="Aspect ratios p/n.")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.5],
                   help="AR(1) correlation values.")
    p.add_argument("--snrs", type=float, nargs="+", default=[1, 5, 10],
                   help="Target SNRs.")
    p.add_argument("--reps", type=int, default=30, help="Replicates per condition.")
    p.add_argument("--beta-sparsity", type=int, default=None,
                   help="If set, use k-sparse beta; else dense.")
    p.add_argument("--beta-scale", type=float, default=1.0,
                   help="Std of nonzero beta entries.")
    p.add_argument("--center-X", action="store_true", help="Center columns of X.")
    p.add_argument("--standardize-X", action="store_true", help="Z-score columns of X.")
    p.add_argument("--output", type=Path, default=Path("results/data/simulation_results.csv"),
                   help="Where to write the CSV of results.")
    p.add_argument("--master-seed", type=int, default=20251014,
                   help="Master seed for reproducibility.")
    p.add_argument("--quiet", action="store_true", help="Disable progress bar.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build full grid of conditions
    grid = list(product(args.dfs, args.gammas, args.rhos, args.snrs, range(args.reps)))
    total_jobs = len(grid)

    # Pre-allocate distinct, reproducible seeds per job using SeedSequence
    ss = np.random.SeedSequence(args.master_seed)
    children = ss.spawn(total_jobs)

    rows = []
    iterator = enumerate(grid)
    if not args.quiet:
        iterator = tqdm(iterator, total=total_jobs, desc="Simulating")

    for i, (df, gamma, rho, snr, rep) in iterator:
        # Compute p from gamma and n
        p = max(1, int(round(gamma * args.n)))

        # Derive a job-specific int seed from the child SeedSequence
        child_rng = np.random.default_rng(children[i])
        job_seed = int(child_rng.integers(0, 2**31 - 1))

        # --- generate one dataset ---
        y, X, beta, meta = generate_full(
            n=args.n,
            p=p,
            rho=float(rho),
            df=float(df),
            snr=float(snr),            # SNR targeting (sigma2 derived inside)
            sigma2=None,
            beta_sparsity=args.beta_sparsity,
            beta_scale=args.beta_scale,
            center_X=args.center_X,
            standardize_X=args.standardize_X,
            seed=job_seed,
        )

        # --- fit estimators ---
        # No intercept in DGP; keep fit_intercept=False so β̂ is comparable to true β
        bhat_ols   = fit_ols(X, y, fit_intercept=False)
        bhat_lad   = fit_lad(X, y, fit_intercept=False, alpha=0.0)   # pure LAD
        bhat_huber = fit_huber(X, y, fit_intercept=False, epsilon=1.35)

        # --- evaluate ---
        mse_ols   = mse_beta(bhat_ols, beta)
        mse_lad   = mse_beta(bhat_lad, beta)
        mse_huber = mse_beta(bhat_huber, beta)

        # --- record one row per method ---
        base = dict(
            n=args.n, p=p, gamma=float(gamma), df=float(df), rho=float(rho),
            snr=float(snr), rep=int(rep),
            center_X=bool(args.center_X), standardize_X=bool(args.standardize_X),
            beta_sparsity=(None if args.beta_sparsity is None else int(args.beta_sparsity)),
            beta_scale=float(args.beta_scale),
            seed=int(job_seed),
            sigma2=float(meta["sigma2"]),
            empirical_snr=float(meta["empirical_snr"]),
        )
        rows.append({**base, "method": "OLS",   "mse": float(mse_ols)})
        rows.append({**base, "method": "LAD",   "mse": float(mse_lad)})
        rows.append({**base, "method": "Huber", "mse": float(mse_huber)})

    # Write results
    df_results = pd.DataFrame(rows)
    df_results.to_csv(args.output, index=False)

    # Tiny summary in stdout
    summary = (
        df_results
        .groupby(["method", "df"], as_index=False)["mse"]
        .mean()
        .sort_values(["method", "df"])
        .head(10)
    )
    print("\nWrote:", args.output)
    print("Preview (mean MSE by method, df):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

