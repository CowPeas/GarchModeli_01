# -*- coding: utf-8 -*-
"""
GRM Ablasyon Çalışması - FAZE 4.

Bu script, GRM modelinin farklı bileşenlerinin performansa katkısını
sistematik olarak ölçer.

FAZE 4: ZENGİNLEŞTİRME
"""

import numpy as np
import pandas as pd
import os
import warnings
import sys
from datetime import datetime

# Windows encoding fix
if sys.platform == 'win32':
    import codecs
    # Check if stdout/stderr already have buffer attribute (not wrapped yet)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Local imports
from models import (
    RealDataLoader,
    AlternativeDataLoader,
    create_manual_download_guide
)
from models.ablation_study import AblationStudy
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> tuple:
    """
    Veriyi train/val/test olarak böler (time-series aware).
    
    Parameters
    ----------
    df : pd.DataFrame
        Zaman serisi verisi
    train_ratio : float
        Eğitim seti oranı
    val_ratio : float
        Doğrulama seti oranı
    test_ratio : float
        Test seti oranı
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def run_ablation_study():
    """
    Ablasyon çalışmasını çalıştırır.
    """
    print("\n" + "=" * 80)
    print("GRM ABLASYON ÇALIŞMASI - FAZE 4")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Dizinleri oluştur
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ========================================================================
    # ADIM 1: VERİ YÜKLEME
    # ========================================================================
    print("[VERI] ADIM 1: Veri Yükleme")
    print("-" * 80)
    
    loader = RealDataLoader()
    alt_loader = AlternativeDataLoader()
    df = None
    metadata = None
    
    # Manuel CSV kontrol
    csv_path = os.path.join(OUTPUT_PATHS['data'], f"{REAL_DATA_CONFIG['ticker']}.csv")
    
    if os.path.exists(csv_path):
        print(f"[OK] MANUEL CSV BULUNDU: {csv_path}\n")
        try:
            df = alt_loader.load_from_csv(
                filepath=csv_path,
                date_column='Date',
                price_column='Close'
            )
            metadata = {
                'asset': REAL_DATA_CONFIG['ticker'],
                'data_type': 'manual_csv'
            }
            print(f"[OK] CSV'DEN YÜKLEME BAŞARILI! ({len(df)} gözlem)\n")
        except Exception as e:
            print(f"[HATA] CSV okuma hatası: {str(e)}\n")
    
    # Otomatik indirme
    if df is None:
        print("[DOWNLOAD] OTOMATIK İNDİRME BAŞLATILIYOR...\n")
        try:
            df, metadata = loader.load_yahoo_finance(
                ticker=REAL_DATA_CONFIG['ticker'],
                start_date=REAL_DATA_CONFIG['start_date'],
                end_date=REAL_DATA_CONFIG['end_date'],
                column='Close',
                verify_ssl=False
            )
            print(f"[OK] Otomatik indirme başarılı!\n")
        except Exception as e:
            print(f"[HATA] Otomatik indirme başarısız\n")
            print("[FALLBACK] Gerçekçi sentetik veri oluşturuluyor...\n")
            
            create_manual_download_guide()
            
            import time
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            
            df = alt_loader.generate_realistic_crypto_data(
                days=730,
                initial_price=30000.0 if 'BTC' in REAL_DATA_CONFIG['ticker'] else 100.0,
                volatility=0.03
            )
            
            metadata = {
                'asset': f"REALISTIC_{REAL_DATA_CONFIG['ticker']}_SYNTHETIC",
                'data_type': 'realistic_synthetic'
            }
            print(f"[OK] Sentetik veri hazır! ({len(df)} gözlem)\n")
    
    # Veri formatını düzelt
    if 'y' not in df.columns and 'returns' in df.columns:
        df['y'] = df['returns']
    elif 'y' not in df.columns and 'price' in df.columns:
        df['y'] = df['price'].pct_change()
        df = df.dropna()
    
    # ========================================================================
    # ADIM 2: VERİ BÖLME
    # ========================================================================
    print("[SPLIT] ADIM 2: Veri Bölme (Train/Val/Test)")
    print("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    print(f"[OK] Train: {len(train_df)} (%{SPLIT_CONFIG['train_ratio']*100:.0f})")
    print(f"[OK] Val:   {len(val_df)} (%{SPLIT_CONFIG['val_ratio']*100:.0f})")
    print(f"[OK] Test:  {len(test_df)} (%{SPLIT_CONFIG['test_ratio']*100:.0f})\n")
    
    # ========================================================================
    # ADIM 3: ABLASYON ÇALIŞMASI
    # ========================================================================
    print("[ABLASYON] ADIM 3: Ablasyon Çalışması")
    print("-" * 80)
    
    study = AblationStudy(
        train_df['y'],
        val_df['y'],
        test_df['y']
    )
    
    # Baseline
    print("\n[1/6] Baseline modeli çalıştırılıyor...")
    study.run_baseline()
    
    # Ablasyonlar
    print("\n[2/6] Mass Only (sadece kütle, decay yok)...")
    study.test_mass_only()
    
    print("\n[3/6] Mass + Decay...")
    study.test_mass_with_decay()
    
    print("\n[4/6] Kerr Full (M + a + decay + tanh)...")
    study.test_kerr_full()
    
    print("\n[5/6] Kerr No Decay (M + a + tanh, decay yok)...")
    study.test_kerr_no_decay()
    
    print("\n[6/6] Kerr Linear (M + a + decay, tanh yok)...")
    study.test_kerr_linear()
    
    # Hassasiyet analizleri
    print("\n[HASSASİYET] Pencere boyutu analizi...")
    study.test_window_sizes([10, 20, 30, 50, 100])
    
    # ========================================================================
    # ADIM 4: RAPOR VE GÖRSELLEŞTİRME
    # ========================================================================
    print("\n[RAPOR] ADIM 4: Sonuç Raporu")
    print("-" * 80)
    
    results_df = study.generate_report()
    study.plot_results()
    
    # Kaydet
    results_file = os.path.join(OUTPUT_PATHS['results'], 'ablation_results.csv')
    results_df.to_csv(results_file, encoding='utf-8')
    print(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    print("=" * 80)
    print("[SUCCESS] ABLASYON ÇALIŞMASI TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return results_df


if __name__ == '__main__':
    results = run_ablation_study()

