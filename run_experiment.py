##################### Parameters to vary ############################
## Tail heaviness: tails of error distribution F (Student-t degrees of freedom) dfs
## Aspect ratio: \gamma = p/n controls the dimensionality regime \gamma s
## Design structure: correlation among predictors \rho s
## Signal-to-noise ratio: overall signal strength snrs
##
##    n : int
##        Number of observations.
##    df : float
##        Degrees of freedom for Student-t error distribution.
##    gamma : float
##        Aspect ratio p/n.
##    rho : float
##        AR(1) correlation parameter in [0, 1). Cov(X_j, X_k) = rho**||j k||
##    snr : float
##        Target signal-to-noise ratio defined as (beta^T X^T X beta) / sigma^2
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
    p.add_argument("--n", type=int, default=500, help="Base sample size (rows).")
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



