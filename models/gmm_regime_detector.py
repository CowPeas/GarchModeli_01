"""Gaussian Mixture Model (GMM) regime detection module.

This module provides an alternative to DBSCAN for regime detection using
Gaussian Mixture Models. GMM guarantees a fixed number of regimes and
handles imbalanced data better than density-based clustering.

PEP8 compliant | PEP257 compliant
"""

from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMRegimeDetector:
    """Gaussian Mixture Model-based regime detector.
    
    This class uses GMM for time series regime detection, offering several
    advantages over DBSCAN:
    - Guaranteed K regimes (no dominant regime problem)
    - Probabilistic regime assignment
    - Better handling of imbalanced regimes
    - BIC/AIC for automatic model selection
    
    Attributes
    ----------
    n_components : int
        Number of regime components.
    covariance_type : str
        GMM covariance type ('full', 'tied', 'diag', 'spherical').
    random_state : int
        Random seed for reproducibility.
    gmm : GaussianMixture
        Fitted GMM model.
    scaler : StandardScaler
        Feature scaler.
    
    Examples
    --------
    >>> detector = GMMRegimeDetector(n_components=3)
    >>> labels = detector.fit_predict(features)
    >>> probs = detector.predict_proba(new_features)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        max_iter: int = 200,
        n_init: int = 10,
        random_state: int = 42
    ):
        """Initialize GMM regime detector.
        
        Parameters
        ----------
        n_components : int, optional
            Number of mixture components (regimes), by default 3.
        covariance_type : str, optional
            Type of covariance parameters, by default 'full'.
            Options: 'full', 'tied', 'diag', 'spherical'.
        max_iter : int, optional
            Maximum EM iterations, by default 200.
        n_init : int, optional
            Number of initializations, by default 10.
        random_state : int, optional
            Random seed, by default 42.
        
        Raises
        ------
        ValueError
            If n_components < 1 or invalid covariance_type.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        
        valid_cov_types = ['full', 'tied', 'diag', 'spherical']
        if covariance_type not in valid_cov_types:
            raise ValueError(
                f"covariance_type must be one of {valid_cov_types}, "
                f"got {covariance_type}"
            )
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        
        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        
        self.bic_: float = np.inf
        self.aic_: float = np.inf
        self.converged_: bool = False
    
    def fit(self, features: np.ndarray) -> 'GMMRegimeDetector':
        """Fit GMM to features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        self
            Fitted detector.
        
        Raises
        ------
        ValueError
            If features array is invalid.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D array, got shape {features.shape}"
            )
        
        if len(features) < self.n_components:
            raise ValueError(
                f"n_samples ({len(features)}) must be >= n_components "
                f"({self.n_components})"
            )
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )
        
        self.gmm.fit(features_scaled)
        
        # Store metrics
        self.bic_ = self.gmm.bic(features_scaled)
        self.aic_ = self.gmm.aic(features_scaled)
        self.converged_ = self.gmm.converged_
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels for features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Regime labels of shape (n_samples,).
        
        Raises
        ------
        RuntimeError
            If detector not fitted yet.
        """
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("Must call fit() before predict()")
        
        features_scaled = self.scaler.transform(features)
        return self.gmm.predict(features_scaled)
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit GMM and predict regime labels.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        
        Returns
        -------
        np.ndarray
            Regime labels.
        """
        self.fit(features)
        return self.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime membership probabilities.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_components).
            Each row sums to 1.
        
        Raises
        ------
        RuntimeError
            If detector not fitted yet.
        
        Examples
        --------
        >>> probs = detector.predict_proba(features)
        >>> print(probs[0])  # Probabilities for first sample
        [0.8, 0.15, 0.05]  # Mostly regime 0
        """
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        
        features_scaled = self.scaler.transform(features)
        return self.gmm.predict_proba(features_scaled)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model selection metrics.
        
        Returns
        -------
        dict
            Dictionary containing BIC, AIC, and convergence status.
        
        Raises
        ------
        RuntimeError
            If detector not fitted yet.
        """
        if self.gmm is None:
            raise RuntimeError("Must call fit() before get_metrics()")
        
        return {
            'bic': self.bic_,
            'aic': self.aic_,
            'converged': self.converged_,
            'n_iter': self.gmm.n_iter_,
            'lower_bound': self.gmm.lower_bound_
        }
    
    def get_regime_statistics(
        self,
        features: np.ndarray
    ) -> Dict[int, Dict]:
        """Compute statistics for each detected regime.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix used for fitting.
        
        Returns
        -------
        dict
            Per-regime statistics including means, covariances, and sizes.
        
        Raises
        ------
        RuntimeError
            If detector not fitted yet.
        """
        if self.gmm is None:
            raise RuntimeError("Must call fit() before get_regime_statistics()")
        
        labels = self.predict(features)
        
        stats = {}
        for regime_id in range(self.n_components):
            regime_mask = labels == regime_id
            n_samples = regime_mask.sum()
            
            stats[regime_id] = {
                'n_samples': int(n_samples),
                'proportion': float(n_samples / len(labels)),
                'mean': self.gmm.means_[regime_id].tolist(),
                'covariance': self.gmm.covariances_[regime_id].tolist()
                if self.covariance_type == 'full'
                else None,
                'weight': float(self.gmm.weights_[regime_id])
            }
        
        return stats


def auto_select_gmm_components(
    features: np.ndarray,
    max_components: int = 10,
    criterion: str = 'bic'
) -> Tuple[int, GMMRegimeDetector]:
    """Automatically select optimal number of GMM components.
    
    Uses BIC or AIC to select the best number of components by fitting
    GMMs with different component counts and comparing information criteria.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    max_components : int, optional
        Maximum components to try, by default 10.
    criterion : str, optional
        Selection criterion ('bic' or 'aic'), by default 'bic'.
    
    Returns
    -------
    tuple of (int, GMMRegimeDetector)
        Optimal number of components and fitted detector.
    
    Raises
    ------
    ValueError
        If invalid criterion or max_components < 2.
    
    Examples
    --------
    >>> n_opt, detector = auto_select_gmm_components(features, max_components=5)
    >>> print(f"Optimal components: {n_opt}")
    3
    """
    if criterion not in ['bic', 'aic']:
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion}")
    
    if max_components < 2:
        raise ValueError(f"max_components must be >= 2, got {max_components}")
    
    # Limit by sample size
    max_components = min(max_components, len(features) // 10)
    
    best_score = np.inf
    best_n = 2
    best_detector = None
    
    scores = []
    
    for n in range(2, max_components + 1):
        detector = GMMRegimeDetector(n_components=n)
        detector.fit(features)
        
        if criterion == 'bic':
            score = detector.bic_
        else:
            score = detector.aic_
        
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_n = n
            best_detector = detector
    
    return best_n, best_detector


def compare_regime_methods(
    features: np.ndarray,
    n_components: int = 3
) -> Dict:
    """Compare GMM with different covariance types.
    
    Fits GMM with different covariance structures and returns comparison
    metrics to help select the best model.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    n_components : int, optional
        Number of components to test, by default 3.
    
    Returns
    -------
    dict
        Comparison results for each covariance type.
    
    Examples
    --------
    >>> results = compare_regime_methods(features)
    >>> print(results['full']['bic'])
    """
    cov_types = ['full', 'tied', 'diag', 'spherical']
    
    results = {}
    
    for cov_type in cov_types:
        detector = GMMRegimeDetector(
            n_components=n_components,
            covariance_type=cov_type
        )
        
        detector.fit(features)
        metrics = detector.get_metrics()
        
        results[cov_type] = {
            'bic': metrics['bic'],
            'aic': metrics['aic'],
            'converged': metrics['converged'],
            'n_iter': metrics['n_iter']
        }
    
    # Find best model
    best_cov_type = min(
        results.keys(),
        key=lambda k: results[k]['bic']
    )
    
    results['best'] = best_cov_type
    
    return results

