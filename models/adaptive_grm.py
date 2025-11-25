"""Adaptive GRM Module - PEP8/PEP257 Compliant.

Implements adaptive alpha adjustment based on volatility regime.
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveAlphaGRM:
    """GRM with adaptive alpha based on volatility.
    
    Adjusts alpha parameter dynamically based on current market volatility:
    - High volatility → Higher alpha (stronger curvature effect)
    - Low volatility → Lower alpha (weaker curvature effect)
    
    Parameters
    ----------
    base_alpha : float
        Base alpha value.
    beta : float
        Beta decay parameter.
    window_size : int
        Window size for volatility estimation.
    alpha_range : Tuple[float, float]
        (min_alpha, max_alpha) range.
    volatility_window : int
        Window for volatility computation.
    adaptation_speed : float
        Speed of alpha adaptation (0-1).
    """
    
    def __init__(
        self,
        base_alpha: float = 1.0,
        beta: float = 0.05,
        window_size: int = 20,
        alpha_range: Tuple[float, float] = (0.1, 5.0),
        volatility_window: int = 50,
        adaptation_speed: float = 0.5
    ):
        """Initialize adaptive GRM."""
        self.base_alpha = base_alpha
        self.beta = beta
        self.window_size = window_size
        self.alpha_min, self.alpha_max = alpha_range
        self.volatility_window = volatility_window
        self.adaptation_speed = adaptation_speed
        
        self.regime_labels = None
        self.body_params = []
        self.current_alpha = base_alpha
        
        # Volatility history
        self.volatility_history = []
        self.alpha_history = []
    
    def fit(self, residuals: np.ndarray) -> 'AdaptiveAlphaGRM':
        """Fit the model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Training residuals.
            
        Returns
        -------
        self
            Fitted model.
        """
        # Compute volatility profile
        volatility_profile = self._compute_volatility_profile(residuals)
        
        # Store for later use
        self.train_volatility = volatility_profile
        self.train_residuals = residuals
        
        # Initialize body params (simplified - single regime for now)
        self.body_params = [{
            'body_id': 0,
            'alpha': self.base_alpha,
            'beta': self.beta,
            'grm_model': None  # Placeholder
        }]
        
        logger.info(f"  Adaptive GRM fitted with base alpha={self.base_alpha:.2f}")
        logger.info(f"  Adaptation range: [{self.alpha_min:.2f}, {self.alpha_max:.2f}]")
        
        return self
    
    def _compute_volatility_profile(
        self,
        residuals: np.ndarray
    ) -> np.ndarray:
        """Compute rolling volatility profile.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals array.
            
        Returns
        -------
        np.ndarray
            Volatility profile.
        """
        n = len(residuals)
        volatility = np.zeros(n)
        
        for t in range(self.volatility_window, n):
            window = residuals[t - self.volatility_window:t]
            volatility[t] = np.std(window)
        
        # Fill initial values with first valid volatility
        if n > self.volatility_window:
            volatility[:self.volatility_window] = volatility[self.volatility_window]
        
        return volatility
    
    def _adapt_alpha(
        self,
        current_volatility: float,
        historical_volatility: np.ndarray
    ) -> float:
        """Adapt alpha based on current volatility.
        
        Parameters
        ----------
        current_volatility : float
            Current market volatility.
        historical_volatility : np.ndarray
            Historical volatility values.
            
        Returns
        -------
        float
            Adapted alpha value.
        """
        # Normalize volatility (z-score)
        mean_vol = np.mean(historical_volatility)
        std_vol = np.std(historical_volatility)
        
        if std_vol < 1e-10:
            return self.base_alpha
        
        vol_z_score = (current_volatility - mean_vol) / std_vol
        
        # Map z-score to alpha range
        # High volatility (z > 0) → Higher alpha
        # Low volatility (z < 0) → Lower alpha
        
        # Sigmoid transformation
        sigmoid = 1.0 / (1.0 + np.exp(-vol_z_score))
        
        # Map to alpha range
        target_alpha = self.alpha_min + sigmoid * (self.alpha_max - self.alpha_min)
        
        # Smooth adaptation
        adapted_alpha = (
            self.adaptation_speed * target_alpha +
            (1 - self.adaptation_speed) * self.current_alpha
        )
        
        # Clip to range
        adapted_alpha = np.clip(adapted_alpha, self.alpha_min, self.alpha_max)
        
        return adapted_alpha
    
    def predict(
        self,
        residuals: np.ndarray,
        current_time: int,
        baseline_pred: float
    ) -> Tuple[float, float, float, int]:
        """Make prediction with adaptive alpha.
        
        Parameters
        ----------
        residuals : np.ndarray
            Full residuals array.
        current_time : int
            Current time index.
        baseline_pred : float
            Baseline prediction.
            
        Returns
        -------
        Tuple[float, float, float, int]
            (baseline_pred, grm_correction, final_pred, regime_id)
        """
        if current_time < self.window_size:
            return baseline_pred, 0.0, baseline_pred, 0
        
        # Compute current volatility
        vol_start = max(0, current_time - self.volatility_window)
        current_vol = np.std(residuals[vol_start:current_time])
        
        # Adapt alpha
        if hasattr(self, 'train_volatility'):
            adapted_alpha = self._adapt_alpha(current_vol, self.train_volatility)
        else:
            adapted_alpha = self.base_alpha
        
        self.current_alpha = adapted_alpha
        
        # Store history
        self.volatility_history.append(current_vol)
        self.alpha_history.append(adapted_alpha)
        
        # Get recent residuals window
        recent_residuals = residuals[current_time - self.window_size:current_time]
        
        # Compute mass (variance)
        if len(recent_residuals) > 1:
            mass = np.var(recent_residuals)
        else:
            mass = 0.0
        
        last_residual = recent_residuals[-1]
        
        # Compute curvature with adaptive alpha
        if not np.isnan(mass) and not np.isnan(last_residual):
            grm_correction = adapted_alpha * mass * np.sign(last_residual)
            
            # Apply beta decay if configured
            if self.beta > 0:
                # Simplified decay
                tau = 5.0
                decay = 1.0 / (1.0 + self.beta * tau)
                grm_correction *= decay
        else:
            grm_correction = 0.0
        
        final_pred = baseline_pred + grm_correction
        
        return baseline_pred, grm_correction, final_pred, 0
    
    def get_adaptation_stats(self) -> dict:
        """Get statistics about alpha adaptation.
        
        Returns
        -------
        dict
            Statistics dictionary.
        """
        if len(self.alpha_history) == 0:
            return {}
        
        alpha_array = np.array(self.alpha_history)
        vol_array = np.array(self.volatility_history)
        
        return {
            'mean_alpha': np.mean(alpha_array),
            'std_alpha': np.std(alpha_array),
            'min_alpha': np.min(alpha_array),
            'max_alpha': np.max(alpha_array),
            'mean_volatility': np.mean(vol_array),
            'std_volatility': np.std(vol_array),
            'correlation': np.corrcoef(alpha_array, vol_array)[0, 1]
        }


class MultiRegimeAdaptiveGRM(AdaptiveAlphaGRM):
    """Adaptive GRM with multiple regimes.
    
    Extends AdaptiveAlphaGRM to work with regime detection.
    
    Parameters
    ----------
    base_alpha : float
        Base alpha value.
    beta : float
        Beta decay parameter.
    window_size : int
        Window size.
    alpha_range : Tuple[float, float]
        Alpha range for adaptation.
    volatility_window : int
        Volatility computation window.
    adaptation_speed : float
        Adaptation speed.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize multi-regime adaptive GRM."""
        super().__init__(*args, **kwargs)
        self.regime_alphas = {}
    
    def fit(self, residuals: np.ndarray) -> 'MultiRegimeAdaptiveGRM':
        """Fit model with regime-specific adaptation.
        
        Parameters
        ----------
        residuals : np.ndarray
            Training residuals.
            
        Returns
        -------
        self
            Fitted model.
        """
        super().fit(residuals)
        
        # Compute regime-specific volatility profiles
        if self.regime_labels is not None:
            unique_regimes = np.unique(self.regime_labels[self.regime_labels != -1])
            
            for regime_id in unique_regimes:
                regime_mask = self.regime_labels == regime_id
                regime_residuals = residuals[regime_mask]
                
                if len(regime_residuals) > self.volatility_window:
                    regime_vol = self._compute_volatility_profile(regime_residuals)
                    self.regime_alphas[regime_id] = {
                        'mean_volatility': np.mean(regime_vol),
                        'std_volatility': np.std(regime_vol)
                    }
        
        return self

