"""
GRM için feature engineering modülü.

Bu modül, rejim tespiti için optimal feature extraction ve
standardization işlemlerini sağlar.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Tuple, Dict
import warnings


class GRMFeatureEngineer:
    """
    GRM için optimal feature extraction.
    
    7-boyutlu feature space:
    1. Mass (volatility)
    2. Spin (autocorrelation)
    3. Time since shock
    4. Kurtosis
    5. Skewness
    6. Local trend
    7. Entropy
    """
    
    @staticmethod
    def extract_regime_features(
        residuals: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Multi-dimensional feature extraction.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        window : int, optional
            Rolling window size
            
        Returns
        -------
        np.ndarray
            Feature matrix (n_samples, 7)
        """
        n = len(residuals)
        features = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            for t in range(window, n):
                window_data = residuals[t - window:t]
                window_data_clean = window_data[~np.isnan(window_data)]
                
                if len(window_data_clean) < 3:
                    # Skip insufficient data
                    continue
                
                # 1. Mass (variance)
                mass = np.var(window_data_clean)
                
                # 2. Spin (ACF lag-1)
                if len(window_data_clean) > 1:
                    try:
                        corr = np.corrcoef(
                            window_data_clean[:-1],
                            window_data_clean[1:]
                        )[0, 1]
                        spin = corr if not np.isnan(corr) else 0.0
                    except:
                        spin = 0.0
                else:
                    spin = 0.0
                
                # 3. Time since shock
                shock_threshold = np.percentile(
                    np.abs(residuals[:t]), 95
                )
                shock_times = np.where(
                    np.abs(residuals[:t]) > shock_threshold
                )[0]
                tau = (t - shock_times[-1]) if len(shock_times) > 0 else window
                tau_normalized = min(tau / window, 1.0)
                
                # 4. Kurtosis
                kurt = scipy_stats.kurtosis(window_data_clean)
                if np.isnan(kurt) or np.isinf(kurt):
                    kurt = 0.0
                
                # 5. Skewness
                skew = scipy_stats.skew(window_data_clean)
                if np.isnan(skew) or np.isinf(skew):
                    skew = 0.0
                
                # 6. Local trend (linear regression slope)
                x = np.arange(len(window_data_clean))
                try:
                    slope = np.polyfit(x, window_data_clean, 1)[0]
                    if np.isnan(slope) or np.isinf(slope):
                        slope = 0.0
                except:
                    slope = 0.0
                
                # 7. Entropy (discretized)
                try:
                    hist, _ = np.histogram(
                        window_data_clean, bins=10, density=True
                    )
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        entropy = -np.sum(hist * np.log(hist + 1e-10))
                    else:
                        entropy = 0.0
                except:
                    entropy = 0.0
                
                features.append([
                    mass, spin, tau_normalized, kurt, skew, slope, entropy
                ])
        
        return np.array(features)
    
    @staticmethod
    def standardize_features(
        X: np.ndarray,
        clip_sigma: float = 5.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Z-score standardization with outlier clipping.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        clip_sigma : float, optional
            Clip threshold (±σ)
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            (standardized features, scaler params)
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)
        
        X_std = (X - mean) / std
        
        # Clip outliers
        X_std = np.clip(X_std, -clip_sigma, clip_sigma)
        
        scaler_params = {
            'mean': mean,
            'std': std,
            'clip_sigma': clip_sigma
        }
        
        return X_std, scaler_params
    
    @staticmethod
    def transform(
        residuals: np.ndarray,
        scaler_params: Dict,
        window: int = 20
    ) -> np.ndarray:
        """
        Transform new data using fitted scaler.
        
        Parameters
        ----------
        residuals : np.ndarray
            New residuals
        scaler_params : Dict
            Scaler parameters from fit
        window : int, optional
            Window size
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        # Extract features
        X = GRMFeatureEngineer.extract_regime_features(residuals, window)
        
        # Standardize
        mean = scaler_params['mean']
        std = scaler_params['std']
        clip_sigma = scaler_params.get('clip_sigma', 5.0)
        
        X_std = (X - mean) / std
        X_std = np.clip(X_std, -clip_sigma, clip_sigma)
        
        return X_std

