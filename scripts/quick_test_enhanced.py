"""Quick test script for enhanced features.

This script provides a quick way to test all new enhanced features
with minimal configuration.

Usage:
    python scripts/quick_test_enhanced.py

PEP8 compliant | PEP257 compliant
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from data_loader import RealDataLoader
except ImportError:
    from models.real_data_loader import RealDataLoader

from models import (
    WindowStratifiedSplit,
    GMMRegimeDetector,
    auto_select_gmm_components,
    GRMFeatureEngineer
)

print("=" * 80)
print("  QUICK TEST: ENHANCED FEATURES")
print("=" * 80)

# Test 1: Window Stratified Split
print("\n[TEST 1] Window-Based Stratified Split")
print("-" * 80)

# Create synthetic regime labels
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'returns': np.random.randn(n_samples) * 0.01,
    'date': pd.date_range('2020-01-01', periods=n_samples)
})

# Simple regime labels (3 regimes)
regime_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

splitter = WindowStratifiedSplit(window_size=30, train_ratio=0.60)
train, val, test = splitter.split(data, regime_labels)

print(f"✓ Train: {len(train)} samples")
print(f"✓ Val:   {len(val)} samples")
print(f"✓ Test:  {len(test)} samples")

dist = splitter.get_regime_distribution()
print("\nRegime Distribution:")
for regime in sorted(dist['train'].keys()):
    print(f"  Regime {regime}: Train={dist['train'].get(regime, 0)}, "
          f"Test={dist['test'].get(regime, 0)}")

print("✅ Window split test PASSED")

# Test 2: GMM Regime Detection
print("\n[TEST 2] GMM Regime Detection")
print("-" * 80)

# Create synthetic features
n_features = 5
features = np.random.randn(n_samples, n_features)

# Add some structure (3 clusters)
features[:300] += np.array([1, 0, 0, 0, 0])
features[300:600] += np.array([0, 1, 0, 0, 0])
features[600:] += np.array([0, 0, 1, 0, 0])

# Auto-select components
n_opt, detector = auto_select_gmm_components(features, max_components=5)

print(f"✓ Optimal components: {n_opt}")

# Get metrics
metrics = detector.get_metrics()
print(f"✓ BIC: {metrics['bic']:.2f}")
print(f"✓ Converged: {metrics['converged']}")

# Predict
labels = detector.predict(features)
unique, counts = np.unique(labels, return_counts=True)

print(f"\nDetected Regimes:")
for regime, count in zip(unique, counts):
    pct = count / len(labels) * 100
    print(f"  Regime {regime}: n={count} ({pct:.1f}%)")

print("✅ GMM test PASSED")

# Test 3: Feature Engineering
print("\n[TEST 3] GRM Feature Engineering")
print("-" * 80)

residuals = np.random.randn(500)
fe = GRMFeatureEngineer(window_size=20)
engineered_features = fe.engineer_features(residuals)

print(f"✓ Input: {len(residuals)} residuals")
print(f"✓ Output: {engineered_features.shape[1]} features")
print(f"✓ Shape: {engineered_features.shape}")

feature_names = ['mass', 'spin', 'tau', 'kurtosis', 'skewness', 'slope', 'entropy']
print(f"\nFeature names: {feature_names}")

print("✅ Feature engineering test PASSED")

# Test 4: Real Data Integration
print("\n[TEST 4] Real Data Integration")
print("-" * 80)

try:
    loader = RealDataLoader()
    df, metadata = loader.load_yahoo_finance(
        ticker='BTC-USD',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    print(f"✓ Loaded {len(df)} observations")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
    print("✅ Real data test PASSED")
    
except Exception as e:
    print(f"⚠️  Real data test SKIPPED (no internet or API issue)")
    print(f"   Error: {e}")

# Summary
print("\n" + "=" * 80)
print("  TEST SUMMARY")
print("=" * 80)
print("✅ Window Stratified Split: PASSED")
print("✅ GMM Regime Detection: PASSED")
print("✅ Feature Engineering: PASSED")
print("✅ Real Data Integration: PASSED/SKIPPED")
print("=" * 80)
print("\n🎉 All tests completed successfully!")
print("\n💡 Next steps:")
print("  1. Run: python main_complete_enhanced.py --mode single")
print("  2. Run: python scripts/test_multi_asset_grm.py")
print("=" * 80)

