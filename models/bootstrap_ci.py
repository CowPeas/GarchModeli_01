"""Bootstrap Confidence Intervals Module - PEP8/PEP257 Compliant.

Provides bootstrap-based confidence interval estimation for model metrics.
"""

import numpy as np
from typing import Tuple


class BootstrapCI:
    """Bootstrap confidence interval estimator.
    
    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = None
    ):
        """Initialize Bootstrap CI estimator."""
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def compute_rmse_ci(
        self,
        errors1: np.ndarray,
        errors2: np.ndarray
    ) -> Tuple[float, float]:
        """Compute bootstrap CI for RMSE difference.
        
        Parameters
        ----------
        errors1 : np.ndarray
            Errors from model 1 (baseline).
        errors2 : np.ndarray
            Errors from model 2 (proposed).
            
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound) of CI for RMSE difference.
        """
        n = len(errors1)
        rmse_diffs = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            
            e1_boot = errors1[indices]
            e2_boot = errors2[indices]
            
            # Compute RMSE difference
            rmse1 = np.sqrt(np.mean(e1_boot ** 2))
            rmse2 = np.sqrt(np.mean(e2_boot ** 2))
            rmse_diff = rmse1 - rmse2
            
            rmse_diffs.append(rmse_diff)
        
        rmse_diffs = np.array(rmse_diffs)
        
        # Compute percentile-based CI
        alpha = 1 - self.confidence_level
        lower = np.percentile(rmse_diffs, alpha / 2 * 100)
        upper = np.percentile(rmse_diffs, (1 - alpha / 2) * 100)
        
        return float(lower), float(upper)
    
    def compute_metric_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable
    ) -> Tuple[float, float]:
        """Compute bootstrap CI for any metric.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        metric_func : callable
            Function that computes metric from (y_true, y_pred).
            
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound) of CI.
        """
        n = len(y_true)
        metric_values = []
        
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            
            yt_boot = y_true[indices]
            yp_boot = y_pred[indices]
            
            metric_val = metric_func(yt_boot, yp_boot)
            metric_values.append(metric_val)
        
        metric_values = np.array(metric_values)
        
        alpha = 1 - self.confidence_level
        lower = np.percentile(metric_values, alpha / 2 * 100)
        upper = np.percentile(metric_values, (1 - alpha / 2) * 100)
        
        return float(lower), float(upper)

