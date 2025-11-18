"""
GRM (Gravitational Residual Model) - Models paketi.

Bu paket, GRM projesinin tüm model modüllerini içerir.
FAZE 1: Schwarzschild rejimi (kütle bazlı) implementasyonu.
FAZE 2: Kerr rejimi (kütle + dönme) implementasyonu.
FAZE 3: Gerçek veri testleri ve GARCH karşılaştırması.
"""

from models.data_generator import SyntheticDataGenerator
from models.baseline_model import BaselineARIMA
from models.grm_model import SchwarzschildGRM
from models.kerr_grm_model import KerrGRM
from models.metrics import ModelEvaluator
from models.visualization import ResultVisualizer
from models.real_data_loader import RealDataLoader, load_popular_assets
from models.garch_model import GARCHModel, SimpleVolatilityModel
from models.alternative_data_loader import AlternativeDataLoader, create_manual_download_guide
from models.ablation_study import AblationStudy
from models.cross_validation import TimeSeriesCrossValidator
from models.grn_network import GravitationalResidualNetwork
from models.grn_trainer import GRNTrainer, GRMDataSet
from models.grn_data_preparator import GRNDataPreparator
from models.symbolic_discovery import SymbolicGRM
from models.unified_grm import UnifiedGRM
from models.multi_body_grm import MultiBodyGRM

__all__ = [
    'SyntheticDataGenerator',
    'BaselineARIMA',
    'SchwarzschildGRM',
    'KerrGRM',
    'ModelEvaluator',
    'ResultVisualizer',
    'RealDataLoader',
    'load_popular_assets',
    'GARCHModel',
    'SimpleVolatilityModel',
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
    'MultiBodyGRM'
]

__version__ = '3.0.0'
__author__ = 'GRM Project Team'

