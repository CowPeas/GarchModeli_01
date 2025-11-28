"""
GRM (Gravitational Residual Model) - Models paketi.

Bu paket, GRM projesinin tüm model modüllerini içerir.
FAZE 1: Schwarzschild rejimi (kütle bazlı) implementasyonu.
FAZE 2: Kerr rejimi (kütle + dönme) implementasyonu.
FAZE 3: Gerçek veri testleri ve GARCH karşılaştırması.
FAZE 4-6: PIML, GRN, Symbolic Discovery, Unified GRM, Multi-Body GRM.
ENHANCED: İstatistiksel testler, Bootstrap CI, Rejim analizi, Kapsamlı karşılaştırma.
"""

from models.data_generator import SyntheticDataGenerator
from models.baseline_model import BaselineARIMA
from models.baseline_model import BaselineARIMA as BaselineModel  # Alias for compatibility
from models.grm_model import SchwarzschildGRM
from models.kerr_grm_model import KerrGRM
from models.metrics import ModelEvaluator, calculate_rmse, calculate_mae
from models.visualization import ResultVisualizer
from models.real_data_loader import RealDataLoader, load_popular_assets
from models.garch_model import GARCHModel
from models.alternative_data_loader import AlternativeDataLoader, create_manual_download_guide
from models.ablation_study import AblationStudy
from models.cross_validation import TimeSeriesCrossValidator
from models.grn_network import GravitationalResidualNetwork
from models.grn_trainer import GRNTrainer, GRMDataSet
from models.grn_data_preparator import GRNDataPreparator
from models.symbolic_discovery import SymbolicGRM
from models.unified_grm import UnifiedGRM
from models.multi_body_grm import MultiBodyGRM
from models.statistical_tests import StatisticalTests
from models.advanced_metrics import AdvancedMetrics, BootstrapCI
from models.comprehensive_comparison import ComprehensiveComparison, quick_compare
from models.regime_analysis import RegimeAnalyzer, analyze_regime_diversity, recommend_dbscan_params

# ADVANCED ROADMAP MODULES (FAZ 1-5) - PEP8/PEP257
from models.power_analysis import StatisticalPowerAnalyzer, quick_power_check
from models.regime_markov_analysis import RegimeMarkovAnalyzer, analyze_regime_coverage
from models.dbscan_optimizer import DBSCANOptimizer, auto_tune_dbscan
from models.grm_feature_engineering import GRMFeatureEngineer
from models.multi_asset_grm import MultiAssetGRM
from models.asset_selection import AssetSelector
from models.adaptive_windowing import AdaptiveWindowGRM
from models.robust_estimation import RobustGRM

# ANALYSIS & VALIDATION MODULES (ENHANCED)
from models.stratified_split import StratifiedTimeSeriesSplit, compare_split_strategies
from models.regime_coverage_validator import RegimeCoverageValidator, quick_coverage_check
from models.window_stratified_split import WindowStratifiedSplit, quick_window_split
from models.gmm_regime_detector import (
    GMMRegimeDetector,
    auto_select_gmm_components,
    compare_regime_methods
)

# IMPROVED MODULES (Performance Enhancement)
from models.grm_hyperparameter_tuning import GRMGridSearch, quick_tune_grm
from models.ensemble_grm import EnsembleGRM, create_ensemble_from_grid
from models.adaptive_grm import AdaptiveAlphaGRM, MultiRegimeAdaptiveGRM

# VISUALIZATION UTILITIES
from models.visualization_utils import GRMVisualizer

__all__ = [
    'SyntheticDataGenerator',
    'BaselineARIMA',
    'BaselineModel',  # Alias for BaselineARIMA
    'SchwarzschildGRM',
    'KerrGRM',
    'ModelEvaluator',
    'calculate_rmse',
    'calculate_mae',
    'ResultVisualizer',
    'RealDataLoader',
    'load_popular_assets',
    'GARCHModel',
    'AlternativeDataLoader',
    'create_manual_download_guide',
    'AblationStudy',
    'TimeSeriesCrossValidator',
    'GravitationalResidualNetwork',
    'GRNTrainer',
    'GRMDataSet',
    'GRNDataPreparator',
    'SymbolicGRM',
    'UnifiedGRM',
    'MultiBodyGRM',
    'StatisticalTests',
    'AdvancedMetrics',
    'BootstrapCI',
    'ComprehensiveComparison',
    'quick_compare',
    'RegimeAnalyzer',
    'analyze_regime_diversity',
    'recommend_dbscan_params',
    # Advanced Roadmap (FAZ 1-5)
    'StatisticalPowerAnalyzer',
    'quick_power_check',
    'RegimeMarkovAnalyzer',
    'analyze_regime_coverage',
    'DBSCANOptimizer',
    'auto_tune_dbscan',
    'GRMFeatureEngineer',
    'MultiAssetGRM',
    'AssetSelector',
    'AdaptiveWindowGRM',
    'RobustGRM',
    # Enhanced Analysis & Validation
    'StratifiedTimeSeriesSplit',
    'compare_split_strategies',
    'RegimeCoverageValidator',
    'quick_coverage_check',
    'WindowStratifiedSplit',
    'quick_window_split',
    # GMM Alternative
    'GMMRegimeDetector',
    'auto_select_gmm_components',
    'compare_regime_methods',
    # Performance Improvements
    'GRMGridSearch',
    'quick_tune_grm',
    'EnsembleGRM',
    'create_ensemble_from_grid',
    'AdaptiveAlphaGRM',
    'MultiRegimeAdaptiveGRM',
    'GRMVisualizer'
]

__version__ = '4.2.0'  # Multi-Asset + Window Split + GMM (PEP8/PEP257)
__author__ = 'GRM Project Team'

