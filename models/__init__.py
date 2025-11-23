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

__all__ = [
    'SyntheticDataGenerator',
    'BaselineARIMA',
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
    'recommend_dbscan_params'
]

__version__ = '3.1.0'  # Enhanced version
__author__ = 'GRM Project Team'

