import numpy as np

from src.estimators import fit_lad, fit_huber
from src.evaluation import mse_beta
from test.test_estimators import  _tiny_problem


def test_fit_lad_and_huber_shapes_and_mse():
    X, y, beta = _tiny_problem(seed=1)
    bhat_lad = fit_lad(X, y, fit_intercept=False, alpha=0.0)
    bhat_hub = fit_huber(X, y, fit_intercept=False, epsilon=1.35)
    assert bhat_lad.shape == beta.shape
    assert bhat_hub.shape == beta.shape
    # sanity: errors shouldn't be NaN and MSEs should be finite
    assert np.isfinite(mse_beta(bhat_lad, beta))
    assert np.isfinite(mse_beta(bhat_hub, beta))