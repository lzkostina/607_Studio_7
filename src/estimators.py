import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor


def fit_ols(X: np.ndarray, y: np.ndarray, fit_intercept: bool = False) -> np.ndarray:
    """
    Description:
        Ordinary Least Squares
        Returns beta_hat (length p)
        Sensitive to outliers and heavy tails
    """
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    return model.coef_ if not fit_intercept else np.r_[model.intercept_, model.coef_]


def fit_lad(X: np.ndarray, y: np.ndarray, fit_intercept: bool = False, alpha: float = 0.0) -> np.ndarray:
    """
    Description:
        Least Absolute Deviation
        Returns beta_hat (length p)
        Minimizes the sum of absolute residuals; robust to outliers
    """
    model = QuantileRegressor(quantile=0.5, alpha=alpha, fit_intercept=fit_intercept, solver="highs")
    model.fit(X, y)
    return model.coef_ if not fit_intercept else np.r_[model.intercept_, model.coef_]


def fit_huber(X: np.ndarray, y: np.ndarray, fit_intercept: bool = False, epsilon: float = 1.35) -> np.ndarray:
    """
    Description:
        Huber regression
        Returns beta_hat (length p)
        Interpolates between OLS and LAD
    """
    model = HuberRegressor(epsilon=epsilon, fit_intercept=fit_intercept)
    model.fit(X, y)
    return model.coef_ if not fit_intercept else np.r_[model.intercept_, model.coef_]
