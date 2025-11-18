# -*- coding: utf-8 -*-
"""
Multi-Body GRM Test Script - FAZE 6.

Bu script, Multi-Body GRM modelini eğitir ve test eder.

FAZE 6: PIML İLERİ SEVİYE
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
    BaselineARIMA,
    SchwarzschildGRM
)
from models.multi_body_grm import MultiBodyGRM
from models.metrics import calculate_rmse, calculate_mae
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS,
    SCHWARZSCHILD_CONFIG
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


def walk_forward_predict_multi_body(
    baseline_model: BaselineARIMA,
    multi_body_model: MultiBodyGRM,
    test_data: pd.Series,
    verbose: bool = False
) -> tuple:
    """
    Walk-forward validation ile Multi-Body GRM tahminleri.
    
    Parameters
    ----------
    baseline_model : BaselineARIMA
        Eğitilmiş baseline model
    multi_body_model : MultiBodyGRM
        Eğitilmiş Multi-Body GRM modeli
    test_data : pd.Series
        Test verisi
    verbose : bool
        İlerleme göster
        
    Returns
    -------
    tuple
        (baseline_predictions, grm_corrections, final_predictions, regime_ids)
    """
    baseline_preds = []
    grm_corrections = []
    final_preds = []
    regime_ids = []
    
    all_residuals = list(baseline_model.get_residuals())
    
    for i in range(len(test_data)):
        # Baseline tahmin
        baseline_pred = baseline_model.predict(1)[0]
        baseline_preds.append(baseline_pred)
        
        # Multi-Body GRM correction
        current_time = len(all_residuals)
        residuals_array = np.array(all_residuals)
        
        baseline_pred_wf, grm_correction, final_pred, regime_id = \
            multi_body_model.predict(
                residuals_array, current_time, baseline_pred
            )
        
        grm_corrections.append(grm_correction)
        final_preds.append(final_pred)
        regime_ids.append(regime_id)
        
        if verbose and (i % 20 == 0):
            print(f"   Walk-forward Multi-Body: {i+1}/{len(test_data)} "
                  f"(Regime: {regime_id})")
        
        # Gerçek değeri gözlemle
        actual = test_data.iloc[i]
        residual = actual - baseline_pred
        all_residuals.append(residual)
        
        # Baseline'ı güncelle
        if i < len(test_data) - 1:
            try:
                baseline_model.fitted_model = baseline_model.fitted_model.append(
                    [actual], refit=False
                )
            except:
                pass
    
    return (
        np.array(baseline_preds),
        np.array(grm_corrections),
        np.array(final_preds),
        np.array(regime_ids)
    )


def run_multi_body_grm_test():
    """
    Multi-Body GRM test sürecini çalıştırır.
    """
    print("\n" + "=" * 80)
    print("MULTI-BODY GRM TEST - FAZE 6")
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
    # ADIM 2: VERİ BÖLME
    # ========================================================================
    print("[SPLIT] ADIM 2: Veri Bölme (Train/Val/Test)")
    print("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    print(f"[OK] Train: {len(train_df)} (%{SPLIT_CONFIG['train_ratio']*100:.0f})")
    print(f"[OK] Val:   {len(val_df)} (%{SPLIT_CONFIG['val_ratio']*100:.0f})")
    print(f"[OK] Test:  {len(test_df)} (%{SPLIT_CONFIG['test_ratio']*100:.0f})\n")
    
    # ========================================================================
    # ADIM 3: BASELINE MODEL VE REZİDÜELLER
    # ========================================================================
    print("[BASELINE] ADIM 3: Baseline Model ve Rezidüeller")
    print("-" * 80)
    
    baseline = BaselineARIMA()
    
    # Grid search
    best_order = baseline.grid_search(
        train_df['y'], val_df['y'],
        p_range=[0, 1, 2],
        d_range=[0, 1],
        q_range=[0, 1, 2],
        verbose=False
    )
    
    # Fit
    baseline.fit(train_df['y'], order=best_order)
    train_residuals = baseline.get_residuals()
    
    print(f"[OK] Baseline: ARIMA{best_order}")
    print(f"[OK] Train rezidüelleri: {len(train_residuals)} gözlem\n")
    
    # ========================================================================
    # ADIM 4: MULTI-BODY GRM EĞİTİMİ
    # ========================================================================
    print("[MULTI-BODY] ADIM 4: Multi-Body GRM Eğitimi")
    print("-" * 80)
    
    window_size = SCHWARZSCHILD_CONFIG['window_size']
    
    multi_body_model = MultiBodyGRM(
        window_size=window_size,
        eps=0.5,
        min_samples=10,
        use_decay=True
    )
    
    multi_body_model.fit(train_residuals)
    
    print(f"[OK] Multi-Body GRM eğitildi!\n")
    
    # ========================================================================
    # ADIM 5: TEST VE KARŞILAŞTIRMA
    # ========================================================================
    print("[TEST] ADIM 5: Test ve Karşılaştırma")
    print("-" * 80)
    
    # Manuel fonksiyon (Schwarzschild)
    print("   Manuel fonksiyon (Schwarzschild) test ediliyor...")
    manual_model = SchwarzschildGRM(
        window_size=window_size,
        use_decay=True
    )
    manual_model.fit(train_residuals)
    
    # Manuel fonksiyon tahminleri (walk-forward)
    def walk_forward_predict_grm_simple(baseline_model, grm_model, test_data):
        """Basit walk-forward GRM tahmini."""
        predictions = []
        all_residuals = list(baseline_model.get_residuals())
        
        shock_times = None
        if len(all_residuals) > 0:
            shock_times = grm_model.detect_shocks(np.array(all_residuals))
        
        for i in range(len(test_data)):
            baseline_pred = baseline_model.predict(1)[0]
            
            current_time = len(all_residuals)
            tau = grm_model.compute_time_since_shock(current_time, shock_times)
            
            recent_residuals = np.array(all_residuals[-grm_model.window_size:])
            
            if len(recent_residuals) > 0:
                mass = grm_model.compute_mass(recent_residuals)[-1]
                correction = grm_model.compute_curvature_single(
                    recent_residuals[-1], mass, time_since_shock=tau
                )
            else:
                correction = 0.0
            
            final_pred = baseline_pred + correction
            predictions.append(final_pred)
            
            actual = test_data.iloc[i]
            residual = actual - baseline_pred
            all_residuals.append(residual)
            
            if len(all_residuals) > grm_model.window_size:
                shock_times = grm_model.detect_shocks(np.array(all_residuals))
            
            if i < len(test_data) - 1:
                try:
                    baseline_model.fitted_model = baseline_model.fitted_model.append(
                        [actual], refit=False
                    )
                except:
                    pass
        
        return np.array(predictions)
    
    manual_predictions = walk_forward_predict_grm_simple(
        baseline, manual_model, test_df['y']
    )
    manual_rmse = calculate_rmse(test_df['y'].values, manual_predictions)
    
    print(f"   Manuel fonksiyon RMSE: {manual_rmse:.6f}\n")
    
    # Multi-Body tahminleri
    print("   Multi-Body GRM tahminleri yapılıyor...")
    baseline_preds, grm_corrections, multi_body_predictions, regime_ids = \
        walk_forward_predict_multi_body(
            baseline, multi_body_model, test_df['y'], verbose=True
        )
    multi_body_rmse = calculate_rmse(test_df['y'].values, multi_body_predictions)
    
    print(f"   Multi-Body GRM RMSE: {multi_body_rmse:.6f}\n")
    
    # Rejim analizi
    unique_regimes, regime_counts = np.unique(regime_ids, return_counts=True)
    print("   Rejim Dağılımı:")
    for regime_id, count in zip(unique_regimes, regime_counts):
        print(f"     Rejim {regime_id}: {count} gözlem (%{count/len(regime_ids)*100:.1f})")
    print()
    
    # Karşılaştırma
    improvement = (manual_rmse - multi_body_rmse) / manual_rmse * 100
    
    print("=" * 80)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 80)
    print(f"Manuel Fonksiyon RMSE: {manual_rmse:.6f}")
    print(f"Multi-Body GRM RMSE:   {multi_body_rmse:.6f}")
    print(f"İyileşme:              {improvement:+.2f}%")
    print("=" * 80 + "\n")
    
    # Sonuçları kaydet
    results_file = os.path.join(OUTPUT_PATHS['results'], 'multi_body_grm_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-BODY GRM TEST SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Manuel Fonksiyon RMSE: {manual_rmse:.6f}\n")
        f.write(f"  Multi-Body GRM RMSE:   {multi_body_rmse:.6f}\n")
        f.write(f"  İyileşme:              {improvement:+.2f}%\n\n")
        f.write("REJİM ANALİZİ:\n")
        for regime_id, count in zip(unique_regimes, regime_counts):
            f.write(f"  Rejim {regime_id}: {count} gözlem (%{count/len(regime_ids)*100:.1f})\n")
        f.write("\nBODY PARAMETRELERİ:\n")
        for params in multi_body_model.body_params:
            f.write(f"  Body {params['body_id']}: "
                   f"α={params['alpha']:.4f}, "
                   f"β={params['beta']:.4f if params['beta'] else 'N/A'}, "
                   f"n={params['n_samples']}\n")
    
    print(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    print("=" * 80)
    print("[SUCCESS] MULTI-BODY GRM TEST TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'manual_rmse': manual_rmse,
        'multi_body_rmse': multi_body_rmse,
        'improvement': improvement,
        'regime_ids': regime_ids,
        'body_params': multi_body_model.body_params
    }


if __name__ == '__main__':
    results = run_multi_body_grm_test()

