import numpy as np

def mse_beta(beta_hat: np.ndarray, beta_true: np.ndarray) -> float:
    """
    Mean squared error between estimated and true coefficients.
    Returns:
        MSE between estimated and true coefficients
    """
    return float(np.mean((np.asarray(beta_hat) - np.asarray(beta_true))**2))