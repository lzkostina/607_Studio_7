##################### Parameters to vary ############################
## Tail heaviness: tails of error distribution F (Student-t degrees of freedom) dfs
## Aspect ratio: \gamma = p/n controls the dimensionality regime \gamma s
## Design structure: correlation among predictors \rho s
## Signal-to-noise ratio: overall signal strength snrs
#####################################################################

"""
    n : int
        Number of observations.
    df : float
        Degrees of freedom for Student-t error distribution.
    gamma : float
        Aspect ratio p/n.
    rho : float
        AR(1) correlation parameter in [0, 1). Cov(X_j, X_k) = rho**|j-k|.
    snr : float
        Target signal-to-noise ratio defined as (beta^T X^T X beta) / sigma^2.
"""
PARAM_GRID = {
    "dfs": [1, 2, 3, 20, float("inf")],
    "gammas": [0.2, 0.5, 0.8],   # p/n
    "rhos": [0.0, 0.25, 0.5, 0.75],
    "snrs": [1, 5, 10],
    "reps": 30,
    "n": 500,                    # base n
}