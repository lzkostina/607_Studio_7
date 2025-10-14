######################## Stage 1:  Basic implementation ########################
## Data generating process: (y, X, \beta) for y = X \beta + \eps
# X ~ N(0, \Sigma) with AR(1) covariance \Sigma[j,k] = rho**||j k||
# \eps_i ~ iid F (where F is some univariate error distribution)

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


def generate_full(n: int, p: int, rho: float, df: float,
    sigma2: float | None = None,
    snr: float | None = None,      # target SNR = (beta^T X^T X beta) / sigma2
    beta_sparsity: int | None = None,  # None = dense; else exactly k nonzeros
    beta_scale: float = 1.0,       # std of nonzero beta entries
    center_X: bool = False,
    standardize_X: bool = False,
    seed: int | None = None,
    ):
    """
    Generate (y, X, beta) for y = X beta + eps.

    Models:
    -----
    - Fixed noise: pass sigma2 (snr can be None).
    - SNR targeting: pass snr (sigma2 can be None), we set
        sigma2 = (beta^T X^T X beta) / snr
      using the realized X and beta.

    Returns:
    -------
    y : (n, ) ndarray
    X : (n, p) ndarray
    beta : (p, ) ndarray
    meta : dict  (sigma2, empirical_snr, df, rho, snr_target, seed, etc.)

    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer.")

    if sigma2 is None and snr is None:
        raise ValueError("Provide either sigma2 or snr.")

    if sigma2 is not None and sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative.")

    if snr is not None and snr <= 0:
        raise ValueError("snr must be positive.")

    rng = np.random.default_rng(seed)
    # Split the master RNG stream so design and errors aren’t coupled
    seed_X  = int(rng.integers(2**31 - 1))
    seed_eps = int(rng.integers(2**31 - 1))

    # 1) Design
    X = generate_data(n=n, p=p, rho=rho, seed=seed_X)

    if center_X or standardize_X:
        X = X.astype(float, copy=True)
        if center_X:
            X -= X.mean(axis=0, keepdims=True)
        if standardize_X:
            sd = X.std(axis=0, ddof=1, keepdims=True)
            sd = np.where(sd <= EPS, 1.0, sd)
            X /= sd

    # 2) Beta (dense or k-sparse)
    if beta_sparsity is None:
        beta = rng.normal(loc=0.0, scale=beta_scale, size=p)
    else:
        if not (1 <= beta_sparsity <= p):
            raise ValueError("beta_sparsity must be in [1, p].")
        beta = np.zeros(p)
        idx = rng.choice(p, size=beta_sparsity, replace=False)
        beta[idx] = rng.normal(loc=0.0, scale=beta_scale, size=beta_sparsity)

    # 3) Choose sigma2 (fixed or via SNR)
    if sigma2 is None:
        # SNR targeting: sigma2 = (beta^T X^T X beta) / snr
        signal_energy = float(beta @ (X.T @ (X @ beta)))
        if signal_energy <= EPS:
            sigma2 = 1.0  # degenerate case: give reasonable noise
        else:
            sigma2 = signal_energy / float(snr)

    # 4) Errors
    eps = generate_errors(n=n, df=df, sigma2=float(sigma2), seed=seed_eps)

    # 5) Response
    y = X @ beta + eps

    # 6) Meta info
    noise_var = float(eps.var(ddof=1))
    signal_energy = float(beta @ (X.T @ (X @ beta)))
    empirical_snr = math.inf if noise_var <= EPS else signal_energy / noise_var

    meta = dict(
        sigma2=float(sigma2),
        empirical_snr=float(empirical_snr),
        df=float(df),
        rho=float(rho),
        snr_target=(None if snr is None else float(snr)),
        seed=int(seed) if seed is not None else None,
        center_X=bool(center_X),
        standardize_X=bool(standardize_X),
        beta_sparsity=(None if beta_sparsity is None else int(beta_sparsity)),
    )
    return y, X, beta, meta




