"""
Advanced GRM Features Test Script.

Bu script, yeni eklenen advanced roadmap modüllerini test eder.
FAZ 1-5 özelliklerini demonstrate eder.
"""

import sys
import os
import numpy as np
import pandas as pd

# Encoding fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models import (
    StatisticalPowerAnalyzer,
    quick_power_check,
    RegimeMarkovAnalyzer,
    DBSCANOptimizer,
    auto_tune_dbscan,
    GRMFeatureEngineer,
    AssetSelector
)
from config_phase3 import (
    POWER_ANALYSIS_CONFIG,
    MARKOV_ANALYSIS_CONFIG
)


def test_power_analysis():
    """Test statistical power analysis."""
    print("\n" + "=" * 80)
    print("FAZ 1: STATISTICAL POWER ANALYSIS")
    print("=" * 80)
    
    analyzer = StatisticalPowerAnalyzer(
        alpha=POWER_ANALYSIS_CONFIG['alpha'],
        power=POWER_ANALYSIS_CONFIG['target_power']
    )
    
    # Mevcut durum
    current_n = 110
    observed_delta = 0.000041
    observed_sigma = 0.025
    
    report = analyzer.power_analysis_report(
        n_current=current_n,
        delta_observed=observed_delta,
        sigma_observed=observed_sigma
    )
    
    print(f"\nMevcut Sample Size: {report['current_n']}")
    print(f"Current Power: {report['current_power']:.4f} ({report['current_power']*100:.2f}%)")
    print(f"Gözlemlenen Effect Size: {report['observed_delta']:.6f}")
    print(f"Gerekli Sample Size: {report['required_n_for_target']}")
    print(f"Yeterli güç?: {'✅ EVET' if report['is_adequately_powered'] else '❌ HAYIR'}")
    
    # Öneriler
    print(f"\n{analyzer.interpret_power(report['current_power'])}")
    
    if report['required_n_for_target']:
        increase_factor = report['required_n_for_target'] / current_n
        print(f"\n💡 ÖNERİ: Sample size {increase_factor:.1f}x artırılmalı")
    
    return report


def test_markov_analysis():
    """Test Markov chain regime analysis."""
    print("\n" + "=" * 80)
    print("FAZ 1: MARKOV CHAIN REJİM ANALİZİ")
    print("=" * 80)
    
    # Synthetic regime labels
    np.random.seed(42)
    n = 500
    regime_labels = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    
    # Add persistence
    for i in range(1, n):
        if np.random.rand() < 0.7:  # 70% persistence
            regime_labels[i] = regime_labels[i - 1]
    
    analyzer = RegimeMarkovAnalyzer()
    analyzer.fit(regime_labels)
    
    print(f"\nRejim Sayısı: {analyzer.n_regimes}")
    print(f"Mixing Time: {analyzer.mixing_time:.2f}")
    
    print("\nStationary Distribution:")
    for i, pi in enumerate(analyzer.stationary_dist):
        print(f"  Rejim {i}: π = {pi:.4f} ({pi*100:.2f}%)")
    
    # Recommended test size
    T_rec = analyzer.recommend_test_size(
        coverage_confidence=MARKOV_ANALYSIS_CONFIG['coverage_confidence'],
        min_samples_per_regime=MARKOV_ANALYSIS_CONFIG['min_regime_samples']
    )
    
    print(f"\n💡 ÖNERİ: Minimum test size = {T_rec} gözlem")
    print(f"   ({MARKOV_ANALYSIS_CONFIG['coverage_confidence']*100:.0f}% güvenle tüm rejimleri örneklemek için)")
    
    return analyzer


def test_dbscan_optimizer():
    """Test DBSCAN optimizer."""
    print("\n" + "=" * 80)
    print("FAZ 2: DBSCAN PARAMETRE OPTİMİZASYONU")
    print("=" * 80)
    
    # Synthetic clustered data
    np.random.seed(42)
    n = 200
    
    # 3 clusters
    cluster1 = np.random.randn(70, 5) + [0, 0, 0, 0, 0]
    cluster2 = np.random.randn(70, 5) + [5, 5, 5, 5, 5]
    cluster3 = np.random.randn(60, 5) + [-5, -5, -5, -5, -5]
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Auto-tune
    result = auto_tune_dbscan(X, verbose=True)
    
    print(f"\n[FINAL RESULT]")
    print(f"  Hopkins Statistic: {result['hopkins_statistic']:.4f}")
    print(f"  Clusterable: {'✅ EVET' if result['is_clusterable'] else '❌ HAYIR'}")
    print(f"  Optimal ε: {result['eps']:.4f}")
    print(f"  Optimal minPts: {result['minpts']}")
    print(f"  Cluster sayısı: {result['n_clusters']}")
    print(f"  Silhouette score: {result['silhouette_score']:.4f}")
    
    return result


def test_feature_engineering():
    """Test GRM feature engineering."""
    print("\n" + "=" * 80)
    print("FAZ 2: GRM FEATURE ENGINEERING")
    print("=" * 80)
    
    # Synthetic residuals
    np.random.seed(42)
    n = 500
    residuals = np.random.randn(n) * 0.02
    
    # Add shocks
    shocks = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    residuals += shocks * np.random.randn(n) * 0.1
    
    # Extract features
    features = GRMFeatureEngineer.extract_regime_features(residuals, window=20)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"  (n_samples={features.shape[0]}, n_features={features.shape[1]})")
    
    # Standardize
    features_std, scaler_params = GRMFeatureEngineer.standardize_features(features)
    
    print(f"\nFeature statistics (standardized):")
    print(f"  Mean: {features_std.mean(axis=0)}")
    print(f"  Std: {features_std.std(axis=0)}")
    print(f"  Range: [{features_std.min():.2f}, {features_std.max():.2f}]")
    
    print("\n✅ Feature engineering başarılı!")
    
    return features_std


def test_asset_selection():
    """Test asset selection."""
    print("\n" + "=" * 80)
    print("FAZ 3: ASSET SELECTION")
    print("=" * 80)
    
    selector = AssetSelector()
    portfolio = selector.recommended_portfolio()
    
    print("\nÖnerilen Optimal Portföy:")
    print("-" * 80)
    
    for asset, info in portfolio.items():
        print(f"\n{asset}:")
        print(f"  Tip: {info['type']}")
        print(f"  Volatilite: {info['volatility']}")
        print(f"  Rejim Dinamiği: {info['regime_dynamics']}")
        print(f"  Ağırlık: {info['weight']*100:.0f}%")
    
    total_weight = sum(info['weight'] for info in portfolio.values())
    print(f"\nToplam Ağırlık: {total_weight*100:.0f}%")
    
    return portfolio


def main():
    """Ana test fonksiyonu."""
    print("\n" + "=" * 80)
    print("🚀 ADVANCED GRM FEATURES - COMPREHENSIVE TEST")
    print("=" * 80)
    
    try:
        # FAZ 1
        print("\n\n🎯 FAZ 1 TESTLERI")
        print("=" * 80)
        
        power_report = test_power_analysis()
        markov_analyzer = test_markov_analysis()
        
        # FAZ 2
        print("\n\n🧮 FAZ 2 TESTLERI")
        print("=" * 80)
        
        dbscan_result = test_dbscan_optimizer()
        features = test_feature_engineering()
        
        # FAZ 3
        print("\n\n🌍 FAZ 3 TESTLERI")
        print("=" * 80)
        
        portfolio = test_asset_selection()
        
        # Summary
        print("\n\n" + "=" * 80)
        print("✅ TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
        print("=" * 80)
        
        print("\n📊 ÖZET:")
        print(f"  • Statistical Power Analyzer: ✅")
        print(f"  • Markov Chain Analyzer: ✅")
        print(f"  • DBSCAN Optimizer: ✅")
        print(f"  • Feature Engineering: ✅")
        print(f"  • Asset Selection: ✅")
        
        print("\n💡 SONRAKI ADIMLAR:")
        print("  1. main.py --multi-body ile extended test (5y data)")
        print("  2. Multi-asset implementasyonu")
        print("  3. Adaptive windowing testleri")
        print("  4. Robust estimation uygulaması")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

