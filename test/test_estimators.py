import numpy as np
from src.setup import generate_full
from src.estimators import (
    fit_ols, fit_lad, fit_huber
    )

def _tiny_problem(seed=0):
    y, X, beta, _ = generate_full(n=200, p=10, rho=0.3, df=np.inf, snr=10.0, seed=seed)
    return X, y, beta

def test_fit_ols_shapes():
    X, y, beta = _tiny_problem()
    bhat = fit_ols(X, y, fit_intercept=False)
    assert bhat.shape == beta.shape

def test_fit_lad_shapes():
    X, y, beta = _tiny_problem()
    bhat = fit_lad(X, y, fit_intercept=False)
    assert bhat.shape == beta.shape

def test_fit_huber_shapes():
    X, y, beta = _tiny_problem()
    bhat = fit_huber(X, y, fit_intercept=False)
    assert bhat.shape == beta.shape

