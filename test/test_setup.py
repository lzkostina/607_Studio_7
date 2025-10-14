import numpy as np
import pytest
import math

from src.setup import (
    EPS,
    auto_regressive_cov, generate_data,
    generate_errors, generate_full
)

####################### auto_regressive_cov tests #######################

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


########################## generate_data tests ###########################

def test_generate_data_shape_and_finiteness():
    X = generate_data(n=50, p=10, rho=0.5, seed=123)
    assert X.shape == (50, 10)
    assert np.isfinite(X).all()


def test_generate_data_seed_reproducibility():
    X1 = generate_data(80, 7, rho=0.3, seed=42)
    X2 = generate_data(80, 7, rho=0.3, seed=42)
    X3 = generate_data(80, 7, rho=0.3, seed=99)
    assert np.allclose(X1, X2)
    # Different seeds should almost surely differ
    assert not np.allclose(X1, X3)


def test_generate_data_cov_matches_ar1():
    # With moderately large n, sample covariance should resemble Σ
    n, p, rho = 1000, 6, 0.6
    X = generate_data(n, p, rho, seed=7)
    S_emp = np.cov(X, rowvar=False, ddof=1)
    S_the =  auto_regressive_cov(p, rho)
    # Loose tolerances since n is finite
    assert np.allclose(S_emp, S_the, rtol=0.15, atol=0.15)


################## generate_errors tests ###############################

def test_generate_errors_bad_args():
    import pytest
    with pytest.raises(ValueError):
        generate_errors(n=0, df=3.0, sigma2=1.0, seed=1)
    with pytest.raises(ValueError):
        generate_errors(n=10, df=-1.0, sigma2=1.0, seed=1)
    with pytest.raises(ValueError):
        generate_errors(n=10, df=3.0, sigma2=-0.1, seed=1)


def test_generate_errors_t():
    n = 4000
    sigma2 = 3.0
    for df in (1.0, 2.0, 3.0, 20.0):  # includes infinite-variance cases
        eps = generate_errors(n=n, df=df, sigma2=sigma2, seed=7)
        s2 = eps.var(ddof=1)
        assert np.isfinite(s2)
        assert abs(s2 - sigma2) / sigma2 < 0.25  # looser for heavy tails


def test_generate_errors_infinite():
    n = 2000
    sigma2 = 2.5
    eps = generate_errors(n=n, df=math.inf, sigma2=sigma2, seed=123)
    assert eps.shape == (n,)
    # sample variance should be close to sigma2
    s2 = eps.var(ddof=1)
    assert np.isfinite(s2)
    assert abs(s2 - sigma2) / sigma2 < 0.15  # tolerant finite-sample check


####################### fenerate_full tests ###########################

def test_generate_data_shapes_and_reproducibility():
    y1, X1, b1, m1 = generate_full(n=120, p=15, rho=0.4, df=3.0, snr=5.0, seed=2025)
    y2, X2, b2, m2 = generate_full(n=120, p=15, rho=0.4, df=3.0, snr=5.0, seed=2025)
    assert X1.shape == (120, 15) and y1.shape == (120,) and b1.shape == (15,)
    # reproducibility with same seed
    assert np.allclose(X1, X2) and np.allclose(y1, y2) and np.allclose(b1, b2)

def test_generate_data_snr_targeting_is_reasonable():
    # empirical SNR should be in the right ballpark
    target = 10.0
    y, X, b, meta = generate_full(n=600, p=40, rho=0.2, df=3.0, snr=target, seed=7)
    emp = meta["empirical_snr"]
    assert np.isfinite(emp)
    assert abs(emp - target) / target < 0.35  # finite-sample tolerance

def test_generate_data_sparse_beta_count():
    k = 5
    y, X, b, meta = generate_full(n=100, p=50, rho=0.0, df=math.inf, sigma2=1.0, beta_sparsity=k, seed=11)
    assert (np.count_nonzero(b) == k)