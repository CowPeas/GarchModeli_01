# -*- coding: utf-8 -*-
"""
GRN (Gravitational Residual Network) Eğitim Scripti - FAZE 5.

Bu script, GRN modelini eğitir ve manuel fonksiyon ile karşılaştırır.

FAZE 5: PIML TEMEL ENTEGRASYONU
"""

import numpy as np
import pandas as pd
import os
import warnings
import sys
from datetime import datetime
from torch.utils.data import DataLoader

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
    BaselineARIMA,
    SchwarzschildGRM,
    create_manual_download_guide
)
from models.grn_network import GravitationalResidualNetwork
from models.grn_trainer import GRNTrainer, GRMDataSet
from models.grn_data_preparator import GRNDataPreparator
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


def walk_forward_predict_grn(
    baseline_model: BaselineARIMA,
    grn_model: GravitationalResidualNetwork,
    grn_preparator: GRNDataPreparator,
    test_data: pd.Series,
    window_size: int = 20,
    verbose: bool = False
) -> np.ndarray:
    """
    Walk-forward validation ile GRN tahminleri.
    
    Parameters
    ----------
    baseline_model : BaselineARIMA
        Eğitilmiş baseline model
    grn_model : GravitationalResidualNetwork
        Eğitilmiş GRN modeli
    grn_preparator : GRNDataPreparator
        Veri hazırlayıcı
    test_data : pd.Series
        Test verisi
    window_size : int
        Pencere boyutu
    verbose : bool
        İlerleme göster
        
    Returns
    -------
    np.ndarray
        Final tahminler
    """
    import torch
    
    predictions = []
    all_residuals = list(baseline_model.get_residuals())
    
    # Şok eşiği
    abs_residuals = np.abs(all_residuals)
    shock_threshold = np.quantile(abs_residuals, 0.95)
    
    grn_model.eval()
    
    for i in range(len(test_data)):
        # Baseline tahmin
        baseline_pred = baseline_model.predict(1)[0]
        
        # GRN features hazırla
        if len(all_residuals) >= window_size:
            recent_residuals = np.array(all_residuals[-window_size:])
            
            # Features
            mass = np.var(recent_residuals)
            if len(recent_residuals) > 1 and np.std(recent_residuals) > 1e-8:
                spin = np.corrcoef(recent_residuals[1:], recent_residuals[:-1])[0, 1]
                spin = np.clip(spin, -1, 1)
            else:
                spin = 0.0
            
            tau = grn_preparator.compute_tau(
                np.array(all_residuals),
                shock_threshold
            )
            
            # Tensor'a çevir
            mass_t = torch.FloatTensor([[mass]])
            spin_t = torch.FloatTensor([[spin]])
            tau_t = torch.FloatTensor([[tau]])
            residuals_history_t = torch.FloatTensor([recent_residuals])
            
            # GRN tahmin
            with torch.no_grad():
                grn_correction = grn_model(
                    mass_t, spin_t, tau_t, residuals_history_t
                ).item()
        else:
            grn_correction = 0.0
        
        final_pred = baseline_pred + grn_correction
        predictions.append(final_pred)
        
        if verbose and (i % 20 == 0):
            print(f"   Walk-forward GRN: {i+1}/{len(test_data)}")
        
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
    
    return np.array(predictions)


def run_grn_training():
    """
    GRN eğitim sürecini çalıştırır.
    """
    print("\n" + "=" * 80)
    print("GRN (GRAVITATIONAL RESIDUAL NETWORK) EĞİTİMİ - FAZE 5")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Dizinleri oluştur
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
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
    
    # Val rezidüelleri (baseline val üzerinde tahmin yaparak)
    val_predictions = baseline.predict(len(val_df))
    val_residuals = val_df['y'].values - val_predictions
    
    print(f"[OK] Baseline: ARIMA{best_order}")
    print(f"[OK] Train rezidüelleri: {len(train_residuals)} gözlem")
    print(f"[OK] Val rezidüelleri: {len(val_residuals)} gözlem\n")
    
    # ========================================================================
    # ADIM 4: GRN VERİ HAZIRLAMA
    # ========================================================================
    print("[GRN] ADIM 4: GRN Veri Hazırlama")
    print("-" * 80)
    
    preparator = GRNDataPreparator(
        shock_threshold_quantile=0.95
    )
    
    window_size = SCHWARZSCHILD_CONFIG['window_size']
    
    print(f"   Train features hazırlanıyor (window_size={window_size})...")
    train_features = preparator.prepare_features(
        train_residuals, window_size=window_size
    )
    
    print(f"   Val features hazırlanıyor...")
    val_features = preparator.prepare_features(
        val_residuals, window_size=window_size
    )
    
    print(f"[OK] Train samples: {len(train_features[0])}")
    print(f"[OK] Val samples: {len(val_features[0])}\n")
    
    # Dataset oluştur
    train_dataset = GRMDataSet(*train_features)
    val_dataset = GRMDataSet(*val_features)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # ========================================================================
    # ADIM 5: GRN MODEL OLUŞTURMA
    # ========================================================================
    print("[GRN] ADIM 5: GRN Model Oluşturma")
    print("-" * 80)
    
    model = GravitationalResidualNetwork(
        input_size=4,  # M, a, τ, ε
        hidden_sizes=[64, 32, 16],
        output_size=1,
        use_monotonicity=True,
        use_energy_conservation=True
    )
    
    print(f"[OK] Model oluşturuldu")
    print(f"  - Input size: 4")
    print(f"  - Hidden sizes: [64, 32, 16]")
    print(f"  - Output size: 1")
    print(f"  - Physics constraints: Monotonicity + Energy Conservation\n")
    
    # ========================================================================
    # ADIM 6: GRN EĞİTİMİ
    # ========================================================================
    print("[GRN] ADIM 6: GRN Eğitimi")
    print("-" * 80)
    
    trainer = GRNTrainer(model, learning_rate=0.001)
    
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=100,
        early_stopping=10,
        save_path='models/grn_best.pth'
    )
    
    print(f"[OK] Eğitim tamamlandı!\n")
    
    # ========================================================================
    # ADIM 7: TEST VE KARŞILAŞTIRMA
    # ========================================================================
    print("[TEST] ADIM 7: Test ve Karşılaştırma")
    print("-" * 80)
    
    # Manuel fonksiyon (Schwarzschild)
    print("   Manuel fonksiyon (Schwarzschild) test ediliyor...")
    manual_model = SchwarzschildGRM(
        window_size=window_size,
        use_decay=True
    )
    manual_model.fit(train_residuals)
    
    # Manuel fonksiyon tahminleri (walk-forward)
    # Walk-forward fonksiyonunu burada tanımla
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
    
    manual_predictions = walk_forward_predict_grm_simple(baseline, manual_model, test_df['y'])
    manual_rmse = calculate_rmse(test_df['y'].values, manual_predictions)
    
    print(f"   Manuel fonksiyon RMSE: {manual_rmse:.6f}\n")
    
    # GRN tahminleri
    print("   GRN tahminleri yapılıyor...")
    grn_predictions = walk_forward_predict_grn(
        baseline, model, preparator, test_df['y'],
        window_size=window_size, verbose=True
    )
    grn_rmse = calculate_rmse(test_df['y'].values, grn_predictions)
    
    print(f"   GRN RMSE: {grn_rmse:.6f}\n")
    
    # Karşılaştırma
    improvement = (manual_rmse - grn_rmse) / manual_rmse * 100
    
    print("=" * 80)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 80)
    print(f"Manuel Fonksiyon RMSE: {manual_rmse:.6f}")
    print(f"GRN RMSE:              {grn_rmse:.6f}")
    print(f"İyileşme:              {improvement:+.2f}%")
    print("=" * 80 + "\n")
    
    # Sonuçları kaydet
    results_file = os.path.join(OUTPUT_PATHS['results'], 'grn_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRN EĞİTİM SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Manuel Fonksiyon RMSE: {manual_rmse:.6f}\n")
        f.write(f"  GRN RMSE:              {grn_rmse:.6f}\n")
        f.write(f"  İyileşme:              {improvement:+.2f}%\n\n")
        f.write("EĞİTİM PARAMETRELERİ:\n")
        f.write(f"  Epochs: {len(history['train_loss'])}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"  Best Val Loss: {min(history['val_loss']):.6f}\n")
    
    print(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    print("=" * 80)
    print("[SUCCESS] GRN EĞİTİMİ TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'manual_rmse': manual_rmse,
        'grn_rmse': grn_rmse,
        'improvement': improvement,
        'history': history
    }


if __name__ == '__main__':
    results = run_grn_training()

