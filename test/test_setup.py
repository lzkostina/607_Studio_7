
import numpy as np
import pytest

from src.setup import auto_regressive_cov


def test_ar1_cov_shape_and_diagonal():
    S =  auto_regressive_cov(5, 0.5)
    assert S.shape == (5, 5)
    assert np.allclose(np.diag(S), 1.0)


def test_ar1_cov_values():
    rho = 0.5
    S =  auto_regressive_cov(4, rho)
    expected = np.array([
        [1.0, rho, rho**2, rho**3],
        [rho, 1.0, rho, rho**2],
        [rho**2, rho, 1.0, rho],
        [rho**3, rho**2, rho, 1.0],
    ])
    assert np.allclose(S, expected)


def test_ar1_cov_rho_bounds():
    with pytest.raises(ValueError):
        auto_regressive_cov(3, -0.1)  # correct interval is 0 <= rho < 1
    with pytest.raises(ValueError):
        auto_regressive_cov(3, 1.0)   # correct interval is 0 <= rho < 1
    with pytest.raises(ValueError):
        auto_regressive_cov(0, 0.5)   # p must be positive int


def test_ar1_cov_positive_definite_small_p():
    # For 0 <= rho < 1, AR(1) covariance should be PD; Cholesky should succeed.
    for rho in (0.0, 0.2, 0.8, 0.999):
        S =  auto_regressive_cov(8, rho)
        L = np.linalg.cholesky(S + 1e-12 * np.eye(S.shape[0]))
        # Sanity: reconstruct
        recon = L @ L.T
        assert np.allclose(recon, S + 1e-12 * np.eye(S.shape[0]), rtol=1e-6, atol=1e-8)


# # ---------- placeholder for the next step (generate_design) ----------
#
# @pytest.mark.xfail(reason="Implement generate_design(n, p, rho) first")
# def test_generate_design_smoke():
#     from src.simulate import generate_design  # will exist next step
#     X = generate_design(n=50, p=10, rho=0.5, seed=123)
#     assert X.shape == (50, 10)
#     # Columns should be roughly centered for large n; with n=50 just smoke-check:
#     assert np.isfinite(X).all()
