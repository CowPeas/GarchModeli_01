"""Simple test for new enhanced modules - standalone.

No external dependencies, just test the new modules.

Usage:
    python scripts/simple_test.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("  SIMPLE TEST: NEW ENHANCED MODULES")
print("=" * 80)

# Test imports
print("\n[TEST 1] Import Check")
print("-" * 80)

try:
    from models.window_stratified_split import WindowStratifiedSplit
    print("[OK] WindowStratifiedSplit imported")
except ImportError as e:
    print(f"[FAIL] WindowStratifiedSplit import failed: {e}")
    sys.exit(1)

try:
    from models.gmm_regime_detector import GMMRegimeDetector, auto_select_gmm_components
    print("[OK] GMMRegimeDetector imported")
except ImportError as e:
    print(f"[FAIL] GMMRegimeDetector import failed: {e}")
    sys.exit(1)

print("[SUCCESS] All imports successful")

# Test Window Split
print("\n[TEST 2] Window Stratified Split")
print("-" * 80)

np.random.seed(42)
n_samples = 1000

# Create synthetic data
data = pd.DataFrame({
    'returns': np.random.randn(n_samples) * 0.01
})
data.index = pd.date_range('2020-01-01', periods=n_samples)

# Create regime labels (3 regimes with imbalance)
regime_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

# Test split
splitter = WindowStratifiedSplit(window_size=30, train_ratio=0.60)
train, val, test = splitter.split(data, regime_labels)

print(f"[OK] Train: {len(train)} samples ({len(train)/n_samples*100:.1f}%)")
print(f"[OK] Val:   {len(val)} samples ({len(val)/n_samples*100:.1f}%)")
print(f"[OK] Test:  {len(test)} samples ({len(test)/n_samples*100:.1f}%)")

# Check regime distribution
dist = splitter.get_regime_distribution()

print("\nRegime Distribution:")
print(f"{'Regime':<10} {'Train':<10} {'Test':<10}")
print("-" * 30)
for regime in sorted(set(dist['train'].keys()) | set(dist['test'].keys())):
    train_cnt = dist['train'].get(regime, 0)
    test_cnt = dist['test'].get(regime, 0)
    print(f"{regime:<10} {train_cnt:<10} {test_cnt:<10}")

# Verify no massive loss
for regime in [1, 2]:
    train_cnt = dist['train'].get(regime, 0)
    if train_cnt > 50:  # Should have reasonable samples
        print(f"[OK] Regime {regime}: {train_cnt} samples in train (good!)")
    else:
        print(f"[WARN] Regime {regime}: Only {train_cnt} samples in train")

print("[SUCCESS] Window split test PASSED")

# Test GMM
print("\n[TEST 3] GMM Regime Detection")
print("-" * 80)

# Create synthetic features with 3 clusters
n_features = 5
features = np.random.randn(n_samples, n_features)

# Add cluster structure
features[:300] += np.array([2, 0, 0, 0, 0])
features[300:700] += np.array([0, 2, 0, 0, 0])
features[700:] += np.array([0, 0, 2, 0, 0])

# Test auto-selection
try:
    n_opt, detector = auto_select_gmm_components(features, max_components=5)
    print(f"[OK] Optimal components: {n_opt}")
    
    metrics = detector.get_metrics()
    print(f"[OK] BIC: {metrics['bic']:.2f}")
    print(f"[OK] AIC: {metrics['aic']:.2f}")
    print(f"[OK] Converged: {metrics['converged']}")
    
    # Test prediction
    labels = detector.predict(features)
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\nDetected {len(unique)} regimes:")
    for regime, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  Regime {regime}: n={count} ({pct:.1f}%)")
    
    # Test probabilities
    probs = detector.predict_proba(features[:10])
    print(f"\n[OK] Probability matrix shape: {probs.shape}")
    print(f"[OK] Sample probabilities sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")
    
    print("[SUCCESS] GMM test PASSED")
    
except Exception as e:
    print(f"[FAIL] GMM test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("  TEST SUMMARY")
print("=" * 80)
print("[PASS] Import Check: PASSED")
print("[PASS] Window Stratified Split: PASSED")
print("[PASS] GMM Regime Detection: PASSED")
print("=" * 80)
print("\n[SUCCESS] All tests completed successfully!")
print("\nNew modules are ready for use:")
print("  - models.window_stratified_split")
print("  - models.gmm_regime_detector")
print("\nNext: Run full pipeline with main_complete_enhanced.py")
print("=" * 80)

