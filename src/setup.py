######################## Stage 1:  Basic implementation ########################
## Data generating process: X ~ N(0, Σ) with AR(1) covariance Σ[j,k] = rho**||j k||
################################################################################

import numpy as np

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
    Z = rng.standard_normal(size=(n, p))  # iid N(0,1)
    X = Z @ L.T                           # induces Cov(X) ≈ Σ (finite-sample)
    return X




