"""Test Improved GRM Models - PEP8/PEP257 Compliant.

Tests hyperparameter tuning, ensemble methods, and adaptive alpha.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd

from models import (
    RealDataLoader,
    BaselineARIMA,
    MultiBodyGRM,
    GRMGridSearch,
    EnsembleGRM,
    AdaptiveAlphaGRM,
    GRMFeatureEngineer,
    GMMRegimeDetector,
    WindowStratifiedSplit,
    create_ensemble_from_grid
)
from config_enhanced import (
    REAL_DATA_CONFIG,
    HYPERPARAMETER_CONFIG,
    ENSEMBLE_CONFIG,
    ADAPTIVE_CONFIG,
    MULTI_ASSET_TEST_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_asset_improved(ticker: str = 'BTC-USD'):
    """Test improved GRM on single asset.
    
    Parameters
    ----------
    ticker : str
        Asset ticker symbol.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"  TESTING IMPROVED GRM - {ticker}")
    logger.info(f"{'='*80}\n")
    
    # 1. Load data
    logger.info("STEP 1: Loading data...")
    data_loader = RealDataLoader(data_source='yahoo')
    df, metadata = data_loader.load_yahoo_finance(
        ticker=ticker,
        start_date=REAL_DATA_CONFIG.get('start_date', '2015-01-01'),
        end_date=REAL_DATA_CONFIG.get('end_date', '2025-11-09'),
        column='Close',
        verify_ssl=False
    )
    
    logger.info(f"  ✓ Loaded {len(df)} observations")
    
    # 2. Baseline ARIMA
    logger.info("\nSTEP 2: Fitting baseline ARIMA...")
    baseline = BaselineARIMA()
    baseline.fit(df['returns'].values, order=(1, 0, 1))
    
    baseline_pred = baseline.predict(steps=len(df))
    residuals = df['returns'].values - baseline_pred
    
    baseline_rmse = np.sqrt(np.mean(residuals ** 2))
    logger.info(f"  ✓ Baseline RMSE: {baseline_rmse:.6f}")
    
    # 3. Regime detection
    logger.info("\nSTEP 3: Detecting regimes (GMM)...")
    features = GRMFeatureEngineer.extract_regime_features(residuals, window=20)
    
    gmm = GMMRegimeDetector(n_components=10, random_state=42)
    regime_labels = gmm.fit_predict(features)
    
    unique_regimes = np.unique(regime_labels)
    logger.info(f"  ✓ Detected {len(unique_regimes)} regimes")
    
    # 4. Split data
    logger.info("\nSTEP 4: Splitting data...")
    df_adjusted = df.iloc[20:].copy()  # Drop first 20 for feature engineering
    df_adjusted.reset_index(drop=True, inplace=True)  # Reset index to 0-based
    
    splitter = WindowStratifiedSplit(
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25
    )
    train_df, val_df, test_df = splitter.split(
        df_adjusted,
        regime_labels
    )
    
    logger.info(f"  ✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get train residuals
    train_baseline = BaselineARIMA()
    train_baseline.fit(train_df['returns'].values, order=(1, 0, 1))
    train_pred = train_baseline.predict(steps=len(train_df))
    train_residuals = train_df['returns'].values - train_pred
    
    # Get validation residuals
    val_pred = train_baseline.predict(steps=len(val_df))
    val_residuals = val_df['returns'].values - val_pred
    
    # Get test predictions
    test_pred = train_baseline.predict(steps=len(test_df))
    
    # === METHOD 1: Hyperparameter Tuning ===
    logger.info(f"\n{'='*80}")
    logger.info("  METHOD 1: GRID SEARCH HYPERPARAMETER TUNING")
    logger.info(f"{'='*80}")
    
    if HYPERPARAMETER_CONFIG['enable_tuning']:
        param_grid = {
            'alpha': HYPERPARAMETER_CONFIG['alpha_range'],
            'beta': HYPERPARAMETER_CONFIG['beta_range'],
            'window_size': HYPERPARAMETER_CONFIG['window_sizes']
        }
        
        grid_search = GRMGridSearch(
            param_grid=param_grid,
            cv_splits=HYPERPARAMETER_CONFIG['cv_splits'],
            scoring='rmse',
            verbose=True
        )
        
        # Get regime labels for train set
        # df_adjusted starts at original index 20, so regime_labels[0] corresponds to df index 20
        # train_df.index contains original df indices, we need to map them to regime_labels
        df_adjusted_start = df_adjusted.index[0]  # Usually 20
        
        # Map train_df indices to regime_labels indices
        train_regime_indices = []
        for idx in train_df.index:
            regime_idx = idx - df_adjusted_start
            if 0 <= regime_idx < len(regime_labels):
                train_regime_indices.append(regime_idx)
        
        train_regime_labels = regime_labels[train_regime_indices]
        
        grid_search.fit(
            train_residuals,
            train_regime_labels,
            MultiBodyGRM
        )
        
        if grid_search.best_params_ is not None:
            logger.info(f"\n  Best parameters found:")
            for k, v in grid_search.best_params_.items():
                logger.info(f"    {k}: {v}")
            logger.info(f"  Best CV RMSE: {grid_search.best_score_:.6f}")
        else:
            logger.warning("\n  Grid search failed! Skipping hyperparameter tuning.")
            logger.warning("  Using default parameters instead.")
    
    # === METHOD 2: Ensemble Methods ===
    logger.info(f"\n{'='*80}")
    logger.info("  METHOD 2: ENSEMBLE GRM")
    logger.info(f"{'='*80}")
    
    if ENSEMBLE_CONFIG['enable_ensemble']:
        # Map train indices to regime_labels indices (same mapping as grid search)
        df_adjusted_start = df_adjusted.index[0]
        train_regime_indices = []
        for idx in train_df.index:
            regime_idx = idx - df_adjusted_start
            if 0 <= regime_idx < len(regime_labels):
                train_regime_indices.append(regime_idx)
        
        train_regime_labels = regime_labels[train_regime_indices]
        
        ensemble = create_ensemble_from_grid(
            ENSEMBLE_CONFIG['param_combinations'],
            MultiBodyGRM,
            train_regime_labels,
            weight_method=ENSEMBLE_CONFIG['weight_method']
        )
        
        ensemble.fit(train_residuals, val_residuals)
        
        logger.info(f"  ✓ Ensemble of {len(ensemble.models)} models trained")
    
    # === METHOD 3: Adaptive Alpha ===
    logger.info(f"\n{'='*80}")
    logger.info("  METHOD 3: ADAPTIVE ALPHA GRM")
    logger.info(f"{'='*80}")
    
    if ADAPTIVE_CONFIG['enable_adaptive']:
        adaptive_grm = AdaptiveAlphaGRM(
            base_alpha=ADAPTIVE_CONFIG['base_alpha'],
            beta=0.1,
            window_size=20,
            alpha_range=ADAPTIVE_CONFIG['alpha_range'],
            volatility_window=ADAPTIVE_CONFIG['volatility_window'],
            adaptation_speed=ADAPTIVE_CONFIG['adaptation_speed']
        )
        
        adaptive_grm.fit(train_residuals)
        
        logger.info(f"  ✓ Adaptive GRM trained")
        logger.info(f"    Base α: {ADAPTIVE_CONFIG['base_alpha']}")
        logger.info(f"    Range: {ADAPTIVE_CONFIG['alpha_range']}")
    
    # === EVALUATE ON TEST SET ===
    logger.info(f"\n{'='*80}")
    logger.info("  EVALUATION ON TEST SET")
    logger.info(f"{'='*80}\n")
    
    # Prepare full residuals
    full_train_pred = train_baseline.predict(steps=len(train_df) + len(test_df))
    full_residuals = np.concatenate([
        train_df['returns'].values,
        test_df['returns'].values
    ]) - full_train_pred
    
    train_len = len(train_df)
    
    # Test ensemble if enabled
    if ENSEMBLE_CONFIG['enable_ensemble']:
        ensemble_corrections = []
        
        for i in range(len(test_df)):
            current_time = train_len + i
            
            try:
                _, correction, _, _ = ensemble.predict(
                    full_residuals,
                    current_time=current_time,
                    baseline_pred=test_pred[i]
                )
                ensemble_corrections.append(correction)
            except Exception:
                ensemble_corrections.append(0.0)
        
        ensemble_pred = test_pred + np.array(ensemble_corrections)
        ensemble_rmse = np.sqrt(np.mean((test_df['returns'].values - ensemble_pred) ** 2))
        
        improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100
        
        logger.info(f"ENSEMBLE RESULTS:")
        logger.info(f"  Baseline RMSE:  {baseline_rmse:.6f}")
        logger.info(f"  Ensemble RMSE:  {ensemble_rmse:.6f}")
        logger.info(f"  Improvement:    {improvement:+.2f}%")
        logger.info(f"  Corrections applied: {np.sum(np.array(ensemble_corrections) != 0)}/{len(test_df)}")
        logger.info(f"  Mean |correction|: {np.mean(np.abs(ensemble_corrections)):.6f}")
    
    # Test adaptive if enabled
    if ADAPTIVE_CONFIG['enable_adaptive']:
        adaptive_corrections = []
        
        for i in range(len(test_df)):
            current_time = train_len + i
            
            try:
                _, correction, _, _ = adaptive_grm.predict(
                    full_residuals,
                    current_time=current_time,
                    baseline_pred=test_pred[i]
                )
                adaptive_corrections.append(correction)
            except Exception:
                adaptive_corrections.append(0.0)
        
        adaptive_pred = test_pred + np.array(adaptive_corrections)
        adaptive_rmse = np.sqrt(np.mean((test_df['returns'].values - adaptive_pred) ** 2))
        
        improvement = (baseline_rmse - adaptive_rmse) / baseline_rmse * 100
        
        stats = adaptive_grm.get_adaptation_stats()
        
        logger.info(f"\nADAPTIVE RESULTS:")
        logger.info(f"  Baseline RMSE:  {baseline_rmse:.6f}")
        logger.info(f"  Adaptive RMSE:  {adaptive_rmse:.6f}")
        logger.info(f"  Improvement:    {improvement:+.2f}%")
        logger.info(f"  Mean α: {stats.get('mean_alpha', 0):.3f}")
        logger.info(f"  α range: [{stats.get('min_alpha', 0):.3f}, {stats.get('max_alpha', 0):.3f}]")
        logger.info(f"  α-volatility correlation: {stats.get('correlation', 0):.3f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("  TEST COMPLETED")
    logger.info(f"{'='*80}\n")


def test_multi_asset():
    """Test on multiple assets."""
    logger.info(f"\n{'='*80}")
    logger.info("  MULTI-ASSET COMPARISON")
    logger.info(f"{'='*80}\n")
    
    results = []
    
    for asset_config in MULTI_ASSET_TEST_CONFIG['assets']:
        ticker = asset_config['ticker']
        logger.info(f"\nTesting {ticker}...")
        
        try:
            test_single_asset_improved(ticker)
            results.append({'ticker': ticker, 'status': 'SUCCESS'})
        except Exception as e:
            logger.error(f"  Error testing {ticker}: {e}")
            results.append({'ticker': ticker, 'status': 'FAILED', 'error': str(e)})
    
    logger.info(f"\n{'='*80}")
    logger.info("  MULTI-ASSET SUMMARY")
    logger.info(f"{'='*80}")
    for result in results:
        status = result['status']
        logger.info(f"  {result['ticker']}: {status}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    # Test single asset with all improvements
    test_single_asset_improved('BTC-USD')
    
    # Test multiple assets if enabled
    if MULTI_ASSET_TEST_CONFIG.get('enable', False):
        test_multi_asset()

