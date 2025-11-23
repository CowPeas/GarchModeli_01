"""
Robust estimation modülü.

Outlier'lara karşı robust M-estimators ve Huber loss.
"""

import numpy as np


class RobustGRM:
    """Outlier'lara robust GRM estimation."""
    
    @staticmethod
    def huber_loss(u: np.ndarray, delta: float = 1.35) -> np.ndarray:
        """
        Huber loss function.
        
        Parameters
        ----------
        u : np.ndarray
            Residuals
        delta : float, optional
            Threshold
            
        Returns
        -------
        np.ndarray
            Loss values
        """
        return np.where(
            np.abs(u) <= delta,
            0.5 * u ** 2,
            delta * np.abs(u) - 0.5 * delta ** 2
        )
    
    @staticmethod
    def iteratively_reweighted_least_squares(
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        IRLS algorithm for robust regression.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target
        max_iter : int, optional
            Maximum iterations
        tol : float, optional
            Convergence tolerance
            
        Returns
        -------
        np.ndarray
            Robust coefficients
        """
        n, p = X.shape
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        for iteration in range(max_iter):
            theta_old = theta.copy()
            
            # Residuals
            residuals = y - X @ theta
            
            # MAD scale estimate
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = 1.4826 * mad
            
            # Weights (Huber)
            u = residuals / (scale + 1e-10)
            weights = np.where(
                np.abs(u) <= 1.35,
                1.0,
                1.35 / np.abs(u)
            )
            
            # Weighted least squares
            W = np.diag(weights)
            try:
                theta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            except:
                break
            
            # Convergence check
            if np.linalg.norm(theta - theta_old) < tol:
                break
        
        return theta

