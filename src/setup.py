######################## Stage 1:  Basic implementation ########################
## Data generating process: X ~ N(0, Σ) with AR(1) covariance Σ[j,k] = rho**||j k||
################################################################################

import numpy as np
import math

EPS = 1e-12

def auto_regressive_cov(p: int, rho: float) -> np.ndarray:
    """
    Construct a covariance matrix Sigma with entries:
        Sigma[j, k] = rho ** ||j - k||,   0 <= rho < 1

    Parameters:
    ----------
    p : int
        Dimensionality (number of predictors)
    rho : float
        Autocorrelation parameter in [0, 1)

    Returns:
    --------
    Sigma : (p, p) ndarray (float64)
    """

    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
    if not (0 <= rho < 1):
        raise ValueError("rho must be in [0, 1)")
    idx = np.arange(p)

    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_data(n: int, p: int, rho: float, seed: int | None = None) -> np.ndarray:
    """
    Draw X ~ N(0, Σ) with AR(1) covariance Σ[j,k] = rho**||j k||.

    Parameters:
    ----------
        n : int
            Dimensionality (number of observations)
        p : int
            Dimensionality (number of predictors)
        rho : float
            Autocorrelation parameter in [0, 1)
        seed : int | None
            Seed for random number generator

    Returns:
    -------
        X : (n, p) ndarray (float64)
    """
    # inputs check
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
    if not (0 <= rho < 1):
        raise ValueError("rho must be in [0, 1)")

    # Build AR(1) covariance and its (stabilized) Cholesky factor
    Sigma = auto_regressive_cov(p, rho)
    L = np.linalg.cholesky(Sigma + EPS * np.eye(p))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n, p))
    X = Z @ L.T
    return X


def generate_errors(n: int, df: float, sigma2: float, seed: int | None = None) -> np.ndarray:
    """
    Generate errors vector \varepsilon of length n

    Parameters:
    ----------
    n : int
        Number of observations
    df : float
        Degrees of freedom for t distribution; use math.inf for Gaussian
    sigma2 : float
        Target noise variance
    seed : int | None
        Random seed

    Returns:
    -------
    eps : (n, ) ndarray
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative.")
    rng = np.random.default_rng(seed)

    # Gaussian branch (df treated as infinite)
    if not math.isfinite(df):
        return rng.normal(loc=0.0, scale=math.sqrt(max(sigma2, 0.0)), size=n)

    if df <= 0:
        raise ValueError("df must be positive or math.inf.")

    # Student-t draw
    t = rng.standard_t(df, size=n)

    # Center
    t = t - t.mean()

    # Scale to target sample variance ≈ sigma2
    cur_var = t.var(ddof=1)
    # Guard against rare numerical zero variance
    if cur_var <= EPS:
        # fall back to Gaussian if pathologically degenerate
        return rng.normal(loc=0.0, scale=math.sqrt(max(sigma2, 0.0)), size=n)

    scale = math.sqrt(max(sigma2, 0.0) / cur_var)
    eps = t * scale
    return eps



