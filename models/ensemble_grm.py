"""Ensemble GRM Module - PEP8/PEP257 Compliant.

Implements ensemble methods for GRM models with weighted averaging.
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleGRM:
    """Ensemble of multiple GRM models with different hyperparameters.
    
    Combines predictions from multiple GRM models using weighted averaging
    based on validation performance.
    
    Parameters
    ----------
    models : List
        List of GRM model instances.
    weights : Optional[np.ndarray]
        Model weights. If None, equal weights are used.
    weight_method : str
        Method to compute weights: 'equal', 'performance', 'inverse_error'.
    """
    
    def __init__(
        self,
        models: Optional[List] = None,
        weights: Optional[np.ndarray] = None,
        weight_method: str = 'performance'
    ):
        """Initialize ensemble."""
        self.models = models if models is not None else []
        self.weights = weights
        self.weight_method = weight_method
        
        self.fitted = False
        self.performance_scores = []
    
    def add_model(self, model) -> 'EnsembleGRM':
        """Add a model to the ensemble.
        
        Parameters
        ----------
        model
            GRM model instance.
            
        Returns
        -------
        self
            For chaining.
        """
        self.models.append(model)
        return self
    
    def fit(
        self,
        residuals: np.ndarray,
        validation_residuals: Optional[np.ndarray] = None
    ) -> 'EnsembleGRM':
        """Fit all models in ensemble.
        
        Parameters
        ----------
        residuals : np.ndarray
            Training residuals.
        validation_residuals : np.ndarray, optional
            Validation residuals for weight computation.
            
        Returns
        -------
        self
            Fitted ensemble.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"  ENSEMBLE GRM - Training {len(self.models)} models")
        logger.info(f"{'='*80}\n")
        
        for i, model in enumerate(self.models):
            logger.info(f"[{i+1}/{len(self.models)}] Training model...")
            try:
                model.fit(residuals)
                logger.info(f"  ✓ Model {i+1} trained successfully")
            except Exception as e:
                logger.error(f"  ✗ Model {i+1} failed: {e}")
        
        # Compute weights based on validation performance
        if validation_residuals is not None and self.weight_method != 'equal':
            self._compute_weights(residuals, validation_residuals)
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.fitted = True
        
        logger.info(f"\n{'='*80}")
        logger.info(f"  ENSEMBLE WEIGHTS")
        logger.info(f"{'='*80}")
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            logger.info(f"  Model {i+1}: weight={weight:.4f}")
        logger.info(f"{'='*80}\n")
        
        return self
    
    def _compute_weights(
        self,
        train_residuals: np.ndarray,
        val_residuals: np.ndarray
    ):
        """Compute model weights based on validation performance.
        
        Parameters
        ----------
        train_residuals : np.ndarray
            Training residuals.
        val_residuals : np.ndarray
            Validation residuals.
        """
        errors = []
        
        for model in self.models:
            # Generate predictions on validation set
            predictions = []
            
            for i in range(len(val_residuals)):
                if i < model.window_size:
                    predictions.append(0.0)
                else:
                    # Full residuals array
                    full_res = np.concatenate([train_residuals, val_residuals[:i]])
                    current_time = len(train_residuals) + i
                    
                    try:
                        _, correction, _, _ = model.predict(
                            full_res,
                            current_time=current_time,
                            baseline_pred=0.0
                        )
                        predictions.append(correction)
                    except Exception:
                        predictions.append(0.0)
            
            predictions = np.array(predictions)
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((val_residuals - predictions) ** 2))
            errors.append(rmse)
            self.performance_scores.append(rmse)
        
        errors = np.array(errors)
        
        # Compute weights
        if self.weight_method == 'inverse_error':
            # Inverse error weighting
            weights = 1.0 / (errors + 1e-10)
            weights = weights / np.sum(weights)
        elif self.weight_method == 'performance':
            # Softmax weighting (lower error = higher weight)
            weights = np.exp(-errors / np.mean(errors))
            weights = weights / np.sum(weights)
        else:
            # Equal weights
            weights = np.ones(len(self.models)) / len(self.models)
        
        self.weights = weights
    
    def predict(
        self,
        residuals: np.ndarray,
        current_time: int,
        baseline_pred: float
    ) -> Tuple[float, float, float, int]:
        """Make ensemble prediction.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals array.
        current_time : int
            Current time index.
        baseline_pred : float
            Baseline prediction.
            
        Returns
        -------
        Tuple[float, float, float, int]
            (baseline_pred, ensemble_correction, final_pred, dominant_regime)
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet!")
        
        # Get predictions from all models
        corrections = []
        regimes = []
        
        for model in self.models:
            try:
                _, correction, _, regime = model.predict(
                    residuals,
                    current_time=current_time,
                    baseline_pred=baseline_pred
                )
                corrections.append(correction)
                regimes.append(regime)
            except Exception:
                corrections.append(0.0)
                regimes.append(0)
        
        corrections = np.array(corrections)
        
        # Weighted average
        ensemble_correction = np.sum(self.weights * corrections)
        
        # Dominant regime (most common)
        dominant_regime = int(np.bincount(regimes).argmax())
        
        final_pred = baseline_pred + ensemble_correction
        
        return baseline_pred, ensemble_correction, final_pred, dominant_regime
    
    def get_model_contributions(
        self,
        residuals: np.ndarray,
        current_time: int,
        baseline_pred: float
    ) -> Dict[int, Dict]:
        """Get individual model contributions.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals array.
        current_time : int
            Current time index.
        baseline_pred : float
            Baseline prediction.
            
        Returns
        -------
        Dict[int, Dict]
            Dictionary with model contributions.
        """
        contributions = {}
        
        for i, model in enumerate(self.models):
            try:
                _, correction, final, regime = model.predict(
                    residuals,
                    current_time=current_time,
                    baseline_pred=baseline_pred
                )
                contributions[i] = {
                    'correction': correction,
                    'weighted_correction': self.weights[i] * correction,
                    'weight': self.weights[i],
                    'regime': regime,
                    'final_prediction': final
                }
            except Exception as e:
                contributions[i] = {
                    'correction': 0.0,
                    'weighted_correction': 0.0,
                    'weight': self.weights[i],
                    'regime': -1,
                    'error': str(e)
                }
        
        return contributions


def create_ensemble_from_grid(
    param_combinations: List[Dict],
    model_class,
    regime_labels: np.ndarray,
    weight_method: str = 'performance'
) -> EnsembleGRM:
    """Create ensemble from grid of parameters.
    
    Parameters
    ----------
    param_combinations : List[Dict]
        List of parameter dictionaries.
    model_class : class
        GRM model class.
    regime_labels : np.ndarray
        Regime labels for models.
    weight_method : str
        Weight computation method.
        
    Returns
    -------
    EnsembleGRM
        Ensemble with models initialized.
    """
    ensemble = EnsembleGRM(weight_method=weight_method)
    
    for params in param_combinations:
        model = model_class(**params)
        model.regime_labels = regime_labels
        ensemble.add_model(model)
    
    return ensemble

