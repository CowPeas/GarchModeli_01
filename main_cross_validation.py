# -*- coding: utf-8 -*-
"""
GRM Time-Series Cross-Validation - FAZE 4.

Bu script, GRM modellerini rolling window validation ile değerlendirir.

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
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
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
from models.cross_validation import TimeSeriesCrossValidator
from models.baseline_model import BaselineARIMA
from models.grm_model import SchwarzschildGRM
from models.kerr_grm_model import KerrGRM
from models.garch_model import GARCHModel
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS,
    SCHWARZSCHILD_CONFIG,
    KERR_CONFIG
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


def run_cross_validation():
    """
    Time-series cross-validation çalıştırır.
    """
    print("\n" + "=" * 80)
    print("GRM TIME-SERIES CROSS-VALIDATION - FAZE 4")
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
            
            df = alt_loader.generate_realistic_crypto_data(
                days=730,
                initial_price=30000.0 if 'BTC' in REAL_DATA_CONFIG['ticker'] else 100.0,
                volatility=0.03
            )
            print(f"[OK] Sentetik veri hazır! ({len(df)} gözlem)\n")
    
    # Veri formatını düzelt
    if 'y' not in df.columns and 'returns' in df.columns:
        df['y'] = df['returns']
    elif 'y' not in df.columns and 'price' in df.columns:
        df['y'] = df['price'].pct_change()
        df = df.dropna()
    
    # ========================================================================
    # ADIM 2: CV OLUŞTUR
    # ========================================================================
    print("[CV] ADIM 2: Cross-Validation Oluşturma")
    print("-" * 80)
    
    cv = TimeSeriesCrossValidator(
        initial_train_size=300,
        val_size=50,
        test_size=50,
        step_size=50
    )
    
    folds = cv.split(df['y'].values)
    print(f"[OK] Toplam {len(folds)} fold oluşturuldu\n")
    
    # ========================================================================
    # ADIM 3: MODELLERİ TANIMLA
    # ========================================================================
    print("[MODEL] ADIM 3: Modelleri Tanımlama")
    print("-" * 80)
    
    models = {
        'Baseline': (BaselineARIMA, {}),
        'Schwarzschild': (SchwarzschildGRM, {
            'window_size': SCHWARZSCHILD_CONFIG['window_size'],
            'use_decay': SCHWARZSCHILD_CONFIG.get('use_decay', True)
        }),
        'Kerr': (KerrGRM, {
            'window_size': KERR_CONFIG['window_size'],
            'use_decay': KERR_CONFIG.get('use_decay', True),
            'use_tanh': KERR_CONFIG.get('use_tanh', True)
        })
    }
    
    print(f"[OK] {len(models)} model tanımlandı\n")
    
    # ========================================================================
    # ADIM 4: KARŞILAŞTIRMA
    # ========================================================================
    print("[KARSILASTIRMA] ADIM 4: Modelleri Karşılaştırma")
    print("-" * 80)
    
    comparison_df, detailed_results = cv.compare_models(
        models, df['y'].values
    )
    
    # ========================================================================
    # ADIM 5: RAPOR
    # ========================================================================
    print("\n[RAPOR] ADIM 5: Sonuç Raporu")
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TIME-SERIES CROSS-VALIDATION SONUÇLARI")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80 + "\n")
    
    # Kaydet
    results_file = os.path.join(OUTPUT_PATHS['results'], 'cv_results.csv')
    comparison_df.to_csv(results_file, encoding='utf-8')
    print(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    print("=" * 80)
    print("[SUCCESS] TIME-SERIES CROSS-VALIDATION TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return comparison_df, detailed_results


if __name__ == '__main__':
    comparison_df, detailed_results = run_cross_validation()

