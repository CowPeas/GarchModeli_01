"""
Enhanced configuration for GRM project.

Bu config dosyası, advanced features için ek ayarları içerir.
"""

from config_phase3 import *  # Base config'i import et

# OVERRIDE: Extended data period for better regime coverage
REAL_DATA_CONFIG = {
    **REAL_DATA_CONFIG,
    'start_date': '2015-01-01',  # ← 10 YEARS instead of 5
    'end_date': '2025-11-09',
    'period': '10y'              # ← Extended period
}

# ENHANCED FEATURES CONFIG
ENHANCED_FEATURES_CONFIG = {
    'use_stratified_split': True,       # Stratified split kullan
    'use_auto_tuned_dbscan': True,      # Auto-tuned DBSCAN kullan
    'validate_regime_coverage': True,   # Regime coverage validate et
    'generate_detailed_reports': True   # Detaylı raporlar oluştur
}

# STRATIFIED SPLIT CONFIG (override SPLIT_CONFIG if enabled)
STRATIFIED_SPLIT_CONFIG = {
    'enable': True,
    'window_size': 30,    # Window size for stratified split (days)
    'train_ratio': 0.60,  # ← Increased for better training
    'val_ratio': 0.15,
    'test_ratio': 0.25,   # ← Sufficient for regime coverage
    'preserve_temporal_order': True,
    'min_regime_samples': 50  # ← More stringent (from 20)
}

# DBSCAN AUTO-TUNING CONFIG
DBSCAN_AUTO_TUNE_CONFIG = {
    'enable': True,
    'K_desired': 5,                     # Hedef cluster sayısı
    'eps_range_factor': (0.5, 1.5),    # Elbow'un kaç katı test edilsin
    'min_silhouette_score': 0.3,       # Minimum kabul edilebilir silhouette
    'max_outlier_ratio': 0.3           # Maximum outlier oranı
}

# REGIME COVERAGE VALIDATION CONFIG
REGIME_COVERAGE_CONFIG = {
    'enable': True,
    'min_coverage_ratio': 0.5,         # Minimum coverage oranı
    'min_test_regimes': 3,             # Minimum test rejim sayısı
    'min_samples_per_regime': 20,      # Her rejimde minimum sample
    'generate_report': True,            # Rapor oluştur
    'fail_on_inadequate': False        # Yetersiz coverage'da fail mi
}

# MULTI-ASSET TEST CONFIG (Future)
MULTI_ASSET_CONFIG = {
    'enable': False,  # Henüz implement edilmedi
    'assets': ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F'],
    'run_parallel': False,
    'compare_results': True
}

# OUTPUT PATHS (enhanced)
ENHANCED_OUTPUT_PATHS = {
    **OUTPUT_PATHS,
    'results_dir': './results',  # ← Added missing key
    'stratified_split_report': './results/stratified_split_report.txt',
    'regime_coverage_report': './results/regime_coverage_report.txt',
    'dbscan_tuning_report': './results/dbscan_tuning_report.txt',
    'split_comparison': './results/split_strategy_comparison.csv'
}

# Override OUTPUT_PATHS with enhanced version
OUTPUT_PATHS = ENHANCED_OUTPUT_PATHS

# Feature engineering config
FEATURE_ENGINEERING_CONFIG = {
    'window_size': 20,
    'n_features': 7,  # mass, spin, tau, kurtosis, skewness, slope, entropy
    'standardize': True,
    'clip_sigma': 5.0
}

# Logging config
ENHANCED_LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': True,
    'log_dir': './logs',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# ==============================================================================
# IMPROVED PARAMETERS FOR BETTER PERFORMANCE
# ==============================================================================

# AGGRESSIVE SCHWARZSCHILD CONFIG (for stronger corrections)
SCHWARZSCHILD_CONFIG = {
    **SCHWARZSCHILD_CONFIG,
    'alpha': 2.0,  # ← Increased from 0.1 to 2.0 (20x stronger!)
    'beta': 0.1,   # ← Increased from 0.05
    'window_size': 20
}

# HYPERPARAMETER TUNING CONFIG
HYPERPARAMETER_CONFIG = {
    'enable_tuning': True,
    'alpha_range': [0.5, 1.0, 2.0, 5.0],  # Aggressive alpha values
    'beta_range': [0.01, 0.05, 0.1, 0.5],
    'window_sizes': [10, 15, 20, 30],
    'cv_splits': 3,
    'scoring': 'rmse'
}

# ENSEMBLE CONFIG
ENSEMBLE_CONFIG = {
    'enable_ensemble': True,
    'n_models': 5,  # Number of models in ensemble
    'weight_method': 'performance',  # 'equal', 'performance', 'inverse_error'
    'param_combinations': [
        {'alpha': 0.5, 'beta': 0.05, 'window_size': 20},
        {'alpha': 1.0, 'beta': 0.1, 'window_size': 20},
        {'alpha': 2.0, 'beta': 0.1, 'window_size': 15},
        {'alpha': 5.0, 'beta': 0.05, 'window_size': 20},
        {'alpha': 1.0, 'beta': 0.5, 'window_size': 30}
    ]
}

# ADAPTIVE ALPHA CONFIG
ADAPTIVE_CONFIG = {
    'enable_adaptive': True,
    'base_alpha': 2.0,  # Higher base for BTC volatility
    'alpha_range': (0.5, 5.0),  # (min, max)
    'volatility_window': 50,
    'adaptation_speed': 0.5  # 0 = no adaptation, 1 = instant adaptation
}

# MULTI-ASSET TEST CONFIG (Updated for testing)
MULTI_ASSET_TEST_CONFIG = {
    'enable': True,
    'assets': [
        {'ticker': 'BTC-USD', 'name': 'Bitcoin', 'aggressive_alpha': 2.0},
        {'ticker': 'ETH-USD', 'name': 'Ethereum', 'aggressive_alpha': 2.0},
        {'ticker': 'SPY', 'name': 'S&P 500 ETF', 'aggressive_alpha': 1.0},  # Less volatile
    ],
    'run_parallel': False,
    'compare_results': True,
    'generate_comparison_table': True
}

