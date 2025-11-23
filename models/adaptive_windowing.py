"""
Adaptive windowing ve change point detection.

Non-stationary time series için CUSUM tabanlı structural break detection.
"""

import numpy as np
from typing import List


class AdaptiveWindowGRM:
    """Non-stationary time series için adaptive windowing."""
    
    def __init__(self, base_window: int = 252):
        """
        AdaptiveWindowGRM başlatıcı.
        
        Parameters
        ----------
        base_window : int, optional
            Base window size
        """
        self.base_window = base_window
        self.lambda_forgetting = np.exp(-1 / base_window)
        self.change_points = []
    
    def detect_change_points(
        self,
        residuals: np.ndarray,
        k: float = 0.5,
        h: float = 5.0
    ) -> List[int]:
        """
        CUSUM test ile structural break detection.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals
        k : float, optional
            Allowance
        h : float, optional
            Threshold
            
        Returns
        -------
        List[int]
            Change point indices
        """
        n = len(residuals)
        S = np.zeros(n)
        change_points = []
        
        mu_0 = np.mean(residuals[:min(50, n)])
        
        for t in range(1, n):
            S[t] = max(0, S[t - 1] + (residuals[t] - mu_0) - k)
            
            if S[t] > h:
                change_points.append(t)
                S[t] = 0  # Reset
                mu_0 = np.mean(residuals[max(0, t - 50):t])
        
        self.change_points = change_points
        return change_points

