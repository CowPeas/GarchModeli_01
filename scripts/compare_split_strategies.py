"""
Split stratejilerini karşılaştırma script'i.

Standard temporal split vs Stratified split'i karşılaştırır.
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from models import (
    RealDataLoader,
    BaselineARIMA,
    MultiBodyGRM,
    StratifiedTimeSeriesSplit,
    compare_split_strategies
)
from config_phase3 import REAL_DATA_CONFIG, SPLIT_CONFIG, SCHWARZSCHILD_CONFIG


def main():
    """Ana karşılaştırma fonksiyonu."""
    print("\n" + "=" * 80)
    print("🔬 SPLIT STRATEGY COMPARISON")
    print("=" * 80)
    print("")
    
    # Load data
    print("[1/3] Veri yükleniyor...")
    loader = RealDataLoader()
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
    
    print(f"  ✅ {len(df)} gözlem")
    
    # Train baseline & get regimes
    print("\n[2/3] Rejimler tespit ediliyor...")
    
    n = len(df)
    train_end = int(n * 0.7)
    train_temp = df.iloc[:train_end]
    
    baseline = BaselineARIMA()
    baseline.fit(train_temp['y'], order=(2, 0, 2))
    residuals = baseline.get_residuals()
    
    multi_body = MultiBodyGRM(window=SCHWARZSCHILD_CONFIG['window_size'])
    multi_body.fit(residuals)
    
    # Extend regime labels to full data
    full_regime_labels = np.full(len(df), -1)
    full_regime_labels[:len(multi_body.regime_labels)] = multi_body.regime_labels
    
    n_regimes = len(set(multi_body.regime_labels[multi_body.regime_labels != -1]))
    print(f"  ✅ {n_regimes} rejim tespit edildi")
    
    # Compare strategies
    print("\n[3/3] Stratejiler karşılaştırılıyor...")
    
    strategies = [
        {
            'name': 'Standard (70-15-15)',
            'params': {
                'train_ratio': 0.70,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            }
        },
        {
            'name': 'Extended Test (50-15-35)',
            'params': {
                'train_ratio': 0.50,
                'val_ratio': 0.15,
                'test_ratio': 0.35
            }
        },
        {
            'name': 'Balanced (60-20-20)',
            'params': {
                'train_ratio': 0.60,
                'val_ratio': 0.20,
                'test_ratio': 0.20
            }
        }
    ]
    
    comparison_df = compare_split_strategies(
        df['y'],
        full_regime_labels,
        strategies
    )
    
    print("\n" + "=" * 80)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Best strategy
    print("\n" + "=" * 80)
    print("💡 ÖNERİ")
    print("=" * 80)
    
    best_idx = comparison_df['Test_Regimes'].idxmax()
    best = comparison_df.iloc[best_idx]
    
    print(f"\n✅ EN İYİ STRATEJİ: {best['Strategy']}")
    print(f"  • Test Size: {best['Test_Size']}")
    print(f"  • Test Regimes: {best['Test_Regimes']}")
    print(f"  • Coverage: {best['Coverage']:.1%}")
    print(f"  • Valid: {'✅ Evet' if best['Valid'] else '❌ Hayır'}")
    
    # Save
    comparison_df.to_csv('./results/split_strategy_comparison.csv', index=False)
    print(f"\n[OK] Sonuçlar kaydedildi: ./results/split_strategy_comparison.csv")
    
    print("\n" + "=" * 80)
    
    return comparison_df


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

