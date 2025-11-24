"""
Enhanced configuration for GRM project.

Bu config dosyası, advanced features için ek ayarları içerir.
"""

from config_phase3 import *  # Base config'i import et

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
    'train_ratio': 0.50,
    'val_ratio': 0.15,
    'test_ratio': 0.35,
    'preserve_temporal_order': True,
    'min_regime_samples': 20
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
    'stratified_split_report': './results/stratified_split_report.txt',
    'regime_coverage_report': './results/regime_coverage_report.txt',
    'dbscan_tuning_report': './results/dbscan_tuning_report.txt',
    'split_comparison': './results/split_strategy_comparison.csv'
}

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

