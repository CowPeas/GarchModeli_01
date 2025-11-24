"""
Rejim coverage validation script.

Bu script, mevcut test sonuçlarını analiz eder ve
rejim coverage sorunlarını tespit edip çözüm önerir.
"""

import sys
import os

# Windows encoding fix
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from models import (
    RealDataLoader,
    AlternativeDataLoader,
    BaselineARIMA,
    MultiBodyGRM,
    quick_coverage_check,
    RegimeCoverageValidator
)
from config_phase3 import REAL_DATA_CONFIG, SPLIT_CONFIG, SCHWARZSCHILD_CONFIG


def main():
    """Ana validation fonksiyonu."""
    print("\n" + "=" * 80)
    print("🔍 REGIME COVERAGE VALIDATION")
    print("=" * 80)
    print("")
    
    # 1. Load data
    print("[1/4] Veri yükleniyor...")
    loader = RealDataLoader()
    
    # Load with yahoo finance
    df, metadata = loader.load_yahoo_finance(
        ticker=REAL_DATA_CONFIG['ticker'],
        start_date=REAL_DATA_CONFIG['start_date'],
        end_date=REAL_DATA_CONFIG['end_date']
    )
    
    # Use returns column (already computed)
    if REAL_DATA_CONFIG.get('use_returns', True):
        df['y'] = df['returns']
    else:
        df['y'] = df['price']
    
    print(f"  ✅ {len(df)} gözlem yüklendi")
    
    # 2. Split data
    print("\n[2/4] Veri bölünüyor...")
    n = len(df)
    train_end = int(n * SPLIT_CONFIG['train_ratio'])
    val_end = int(n * (SPLIT_CONFIG['train_ratio'] + SPLIT_CONFIG['val_ratio']))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"  Train: {len(train_df)} ({SPLIT_CONFIG['train_ratio']:.0%})")
    print(f"  Val:   {len(val_df)} ({SPLIT_CONFIG['val_ratio']:.0%})")
    print(f"  Test:  {len(test_df)} ({SPLIT_CONFIG['test_ratio']:.0%})")
    
    # 3. Train baseline & Multi-Body GRM
    print("\n[3/4] Model eğitiliyor...")
    
    baseline = BaselineARIMA()
    best_order = baseline.grid_search(
        train_df['y'], val_df['y'],
        p_range=[0, 1, 2],
        d_range=[0, 1],
        q_range=[0, 1, 2],
        verbose=False
    )
    baseline.fit(train_df['y'], order=best_order)
    train_residuals = baseline.get_residuals()
    
    multi_body_grm = MultiBodyGRM(window_size=SCHWARZSCHILD_CONFIG['window_size'])
    multi_body_grm.fit(train_residuals)
    
    train_regime_labels = multi_body_grm.regime_labels
    n_train_regimes = len(set(train_regime_labels[train_regime_labels != -1]))
    
    print(f"  ✅ Baseline: ARIMA{best_order}")
    print(f"  ✅ Multi-Body GRM: {n_train_regimes} rejim (train)")
    
    # 4. Predict test regimes
    print("\n[4/4] Test rejimleri tahmin ediliyor...")
    
    test_regime_labels = []
    for i in range(len(test_df)):
        try:
            regime = multi_body_grm.predict_regime(test_df['y'].iloc[i:i+1].values)
            test_regime_labels.append(regime)
        except:
            test_regime_labels.append(-1)
    
    test_regime_labels = np.array(test_regime_labels)
    n_test_regimes = len(set(test_regime_labels[test_regime_labels != -1]))
    
    print(f"  ✅ Test rejimleri: {n_test_regimes} unique")
    
    # 5. Validate coverage
    print("\n" + "=" * 80)
    print("📊 COVERAGE ANALYSIS")
    print("=" * 80)
    
    result = quick_coverage_check(train_regime_labels, test_regime_labels, verbose=True)
    
    # 6. Detailed report
    print("\n[DETAILED REPORT] Detaylı rapor oluşturuluyor...")
    
    validator = RegimeCoverageValidator(train_regime_labels, test_regime_labels)
    report = validator.generate_report(
        output_file='./results/regime_coverage_validation.txt'
    )
    
    print("  ✅ Rapor kaydedildi: ./results/regime_coverage_validation.txt")
    
    # 7. Recommendations
    if not result['is_adequate']:
        print("\n" + "=" * 80)
        print("💡 ÖNERİLER")
        print("=" * 80)
        
        recommendations = result['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{rec['priority']} Öneri {i}:")
            print(f"  Problem:  {rec['problem']}")
            print(f"  Çözüm:    {rec['solution']}")
            print(f"  Beklenen: {rec['expected_improvement']}")
        
        print("\n" + "=" * 80)
        print("🚀 SONRAKI ADIMLAR")
        print("=" * 80)
        print("1. python main_multi_body_grm_enhanced.py  # Stratified split dene")
        print("2. python scripts/compare_split_strategies.py  # Split'leri karşılaştır")
        print("3. python main.py --multi-body --asset ^GSPC  # Farklı varlık dene")
        print("=" * 80)
    else:
        print("\n✅ Test coverage yeterli! Multi-Body GRM avantajı kullanılabilir.")
    
    return result


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result['is_adequate'] else 1)
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

