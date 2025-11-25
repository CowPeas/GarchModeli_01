"""GRM Hyperparameter Tuning Module - PEP8/PEP257 Compliant.

Provides grid search and Bayesian optimization for GRM hyperparameters.
"""

import logging
from typing import Dict, List, Tuple, Optional
from itertools import product

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class GRMGridSearch:
    """Grid search for GRM hyperparameters.
    
    Parameters
    ----------
    param_grid : Dict[str, List]
        Parameter grid to search.
    cv_splits : int
        Number of cross-validation splits.
    scoring : str
        Scoring metric ('rmse', 'mae', 'mape').
    verbose : bool
        Whether to print progress.
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List],
        cv_splits: int = 5,
        scoring: str = 'rmse',
        verbose: bool = True
    ):
        """Initialize grid search."""
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.verbose = verbose
        
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
    
    def fit(
        self,
        residuals: np.ndarray,
        regime_labels: np.ndarray,
        model_class,
        **fit_params
    ) -> 'GRMGridSearch':
        """Fit grid search.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from baseline model.
        regime_labels : np.ndarray
            Regime labels.
        model_class : class
            GRM model class to use.
        **fit_params
            Additional parameters for model fitting.
            
        Returns
        -------
        self
            Fitted grid search object.
        """
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"  GRM GRID SEARCH - {len(param_combinations)} combinations")
            logger.info(f"{'='*80}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        best_score = float('inf')
        best_params = None
        
        for i, param_vals in enumerate(param_combinations):
            params = dict(zip(param_names, param_vals))
            
            if self.verbose:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                logger.info(f"\n[{i+1}/{len(param_combinations)}] Testing: {param_str}")
            
            # Cross-validation
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(residuals):
                res_train = residuals[train_idx]
                res_val = residuals[val_idx]
                labels_train = regime_labels[train_idx]
                
                # Train model
                try:
                    model = model_class(**params, **fit_params)
                    model.regime_labels = labels_train
                    model.fit(res_train)
                    
                    # Predict on validation
                    predictions = []
                    for j in range(len(res_val)):
                        if j < model.window_size:
                            predictions.append(0.0)
                        else:
                            # Create full residuals array
                            full_res = np.concatenate([res_train, res_val[:j]])
                            current_time = len(res_train) + j
                            
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
                    
                    # Compute score
                    if self.scoring == 'rmse':
                        score = np.sqrt(np.mean((res_val - predictions) ** 2))
                    elif self.scoring == 'mae':
                        score = np.mean(np.abs(res_val - predictions))
                    elif self.scoring == 'mape':
                        score = np.mean(np.abs((res_val - predictions) / (res_val + 1e-10))) * 100
                    else:
                        score = np.sqrt(np.mean((res_val - predictions) ** 2))
                    
                    cv_scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"  Error in CV fold: {e}")
                    cv_scores.append(float('inf'))
            
            # Mean CV score
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            if self.verbose:
                logger.info(f"  CV {self.scoring.upper()}: {mean_score:.6f} ± {std_score:.6f}")
            
            # Store results
            self.cv_results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            # Update best
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                if self.verbose:
                    logger.info(f"  ✓ NEW BEST!")
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"  BEST PARAMETERS")
            logger.info(f"{'='*80}")
            if self.best_params_ is not None:
                for k, v in self.best_params_.items():
                    logger.info(f"  {k}: {v}")
                logger.info(f"\n  Best {self.scoring.upper()}: {self.best_score_:.6f}")
            else:
                logger.warning("  NO VALID PARAMETERS FOUND! All combinations failed.")
                logger.warning("  Check your model class and parameter grid.")
            logger.info(f"{'='*80}\n")
        
        return self
    
    def get_best_model(self, model_class, **fit_params):
        """Get model with best parameters.
        
        Parameters
        ----------
        model_class : class
            Model class to instantiate.
        **fit_params
            Additional fit parameters.
            
        Returns
        -------
        model
            Model instance with best parameters.
        """
        if self.best_params_ is None:
            raise ValueError("Grid search not fitted yet!")
        
        return model_class(**self.best_params_, **fit_params)


def quick_tune_grm(
    residuals: np.ndarray,
    regime_labels: np.ndarray,
    model_class,
    alpha_range: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    beta_range: List[float] = [0.01, 0.05, 0.1, 0.5],
    window_sizes: List[int] = [10, 20, 30],
    cv_splits: int = 3,
    verbose: bool = True
) -> Tuple[Dict, float]:
    """Quick GRM hyperparameter tuning.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from baseline.
    regime_labels : np.ndarray
        Regime labels.
    model_class : class
        GRM model class.
    alpha_range : List[float]
        Alpha values to try.
    beta_range : List[float]
        Beta values to try.
    window_sizes : List[int]
        Window sizes to try.
    cv_splits : int
        Cross-validation splits.
    verbose : bool
        Verbose output.
        
    Returns
    -------
    Tuple[Dict, float]
        (best_params, best_score)
    """
    param_grid = {
        'alpha': alpha_range,
        'beta': beta_range,
        'window_size': window_sizes
    }
    
    grid_search = GRMGridSearch(
        param_grid=param_grid,
        cv_splits=cv_splits,
        scoring='rmse',
        verbose=verbose
    )
    
    grid_search.fit(residuals, regime_labels, model_class)
    
    return grid_search.best_params_, grid_search.best_score_

