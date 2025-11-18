# -*- coding: utf-8 -*-
"""
Unified End-to-End GRM Test Script - FAZE 6.

Bu script, Unified GRM modelini eğitir ve test eder.

FAZE 6: PIML İLERİ SEVİYE
"""

import numpy as np
import pandas as pd
import os
import warnings
import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    BaselineARIMA
)
from models.unified_grm import UnifiedGRM
from models.metrics import calculate_rmse, calculate_mae
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TimeSeriesDataset(Dataset):
    """Time series dataset for Unified GRM."""
    
    def __init__(self, data: np.ndarray, seq_len: int = 20):
        """
        TimeSeriesDataset sınıfını başlatır.
        
        Parameters
        ----------
        data : np.ndarray
            Zaman serisi verisi
        seq_len : int, optional
            Sequence length (varsayılan: 20)
        """
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        """Dataset boyutu."""
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Veri öğesi alır.
        
        Returns
        -------
        tuple
            (x_history, target)
        """
        x_history = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len]
        
        # Reshape for LSTM: (seq_len, features)
        x_history = x_history.reshape(self.seq_len, 1)
        
        return torch.FloatTensor(x_history), torch.FloatTensor([target])


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


def train_unified_grm(
    model: UnifiedGRM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    Unified GRM modelini eğitir.
    
    Parameters
    ----------
    model : UnifiedGRM
        Unified GRM modeli
    train_loader : DataLoader
        Eğitim veri yükleyici
    val_loader : DataLoader
        Validation veri yükleyici
    epochs : int, optional
        Epoch sayısı (varsayılan: 50)
    learning_rate : float, optional
        Öğrenme hızı (varsayılan: 0.001)
    device : str, optional
        Cihaz (varsayılan: 'cpu')
        
    Returns
    -------
    dict
        Eğitim geçmişi
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'loss_final': [],
        'loss_baseline': [],
        'loss_physics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping = 10
    
    print("\n[UnifiedGRM] Eğitim başlatılıyor...")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_total = 0.0
        train_loss_final = 0.0
        train_loss_baseline = 0.0
        train_loss_physics = 0.0
        n_batches = 0
        
        for x_history, targets in train_loader:
            x_history = x_history.to(device)
            targets = targets.to(device)
            
            # Forward pass
            baseline_pred, grm_correction, final_pred = model(x_history)
            
            # Compute mass for physics loss
            residuals = x_history[:, :, 0] - baseline_pred.detach()
            mass = torch.var(residuals, dim=1, keepdim=True)
            
            # Loss
            total_loss, loss_final, loss_baseline, loss_physics = \
                model.combined_loss(
                    baseline_pred, grm_correction, final_pred, targets, mass
                )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss_total += total_loss.item()
            train_loss_final += loss_final.item()
            train_loss_baseline += loss_baseline.item()
            train_loss_physics += loss_physics.item()
            n_batches += 1
        
        avg_train_loss = train_loss_total / n_batches if n_batches > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for x_history, targets in val_loader:
                x_history = x_history.to(device)
                targets = targets.to(device)
                
                baseline_pred, grm_correction, final_pred = model(x_history)
                
                residuals = x_history[:, :, 0] - baseline_pred.detach()
                mass = torch.var(residuals, dim=1, keepdim=True)
                
                total_loss, _, _, _ = model.combined_loss(
                    baseline_pred, grm_correction, final_pred, targets, mass
                )
                
                val_loss_total += total_loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss_total / n_val_batches if n_val_batches > 0 else 0.0
        
        # History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['loss_final'].append(train_loss_final / n_batches if n_batches > 0 else 0.0)
        history['loss_baseline'].append(train_loss_baseline / n_batches if n_batches > 0 else 0.0)
        history['loss_physics'].append(train_loss_physics / n_batches if n_batches > 0 else 0.0)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {avg_train_loss:.6f}, "
              f"Val Loss = {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/unified_grm_best.pth')
            print(f"  → Yeni en iyi model kaydedildi (Val Loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f"\n[UnifiedGRM] Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/unified_grm_best.pth'))
    print(f"\n[UnifiedGRM] En iyi model yüklendi (Val Loss: {best_val_loss:.6f})")
    print("-" * 80)
    
    return history


def run_unified_grm_test():
    """
    Unified GRM test sürecini çalıştırır.
    """
    print("\n" + "=" * 80)
    print("UNIFIED END-TO-END GRM TEST - FAZE 6")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Dizinleri oluştur
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] {device} kullanılıyor\n")
    
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
    # ADIM 3: DATASET OLUŞTURMA
    # ========================================================================
    print("[DATASET] ADIM 3: Dataset Oluşturma")
    print("-" * 80)
    
    seq_len = 20
    train_data = train_df['y'].values
    val_data = val_df['y'].values
    test_data = test_df['y'].values
    
    train_dataset = TimeSeriesDataset(train_data, seq_len=seq_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"[OK] Train samples: {len(train_dataset)}")
    print(f"[OK] Val samples: {len(val_dataset)}")
    print(f"[OK] Sequence length: {seq_len}\n")
    
    # ========================================================================
    # ADIM 4: MODEL OLUŞTURMA
    # ========================================================================
    print("[MODEL] ADIM 4: Unified GRM Model Oluşturma")
    print("-" * 80)
    
    model = UnifiedGRM(
        input_size=1,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        grn_hidden_sizes=[64, 32, 16],
        use_monotonicity=True,
        use_energy_conservation=True
    )
    
    print(f"[OK] Model oluşturuldu")
    print(f"  - LSTM hidden size: 64")
    print(f"  - LSTM layers: 2")
    print(f"  - GRN hidden sizes: [64, 32, 16]\n")
    
    # ========================================================================
    # ADIM 5: EĞİTİM
    # ========================================================================
    print("[TRAIN] ADIM 5: Unified GRM Eğitimi")
    print("-" * 80)
    
    history = train_unified_grm(
        model, train_loader, val_loader,
        epochs=50, learning_rate=0.001, device=device
    )
    
    print(f"[OK] Eğitim tamamlandı!\n")
    
    # ========================================================================
    # ADIM 6: TEST
    # ========================================================================
    print("[TEST] ADIM 6: Test ve Karşılaştırma")
    print("-" * 80)
    
    # Baseline ARIMA
    print("   Baseline ARIMA test ediliyor...")
    baseline = BaselineARIMA()
    best_order = baseline.grid_search(
        train_df['y'], val_df['y'],
        p_range=[0, 1, 2],
        d_range=[0, 1],
        q_range=[0, 1, 2],
        verbose=False
    )
    baseline.fit(train_df['y'], order=best_order)
    
    # Baseline tahminleri (walk-forward)
    baseline_predictions = []
    all_train = list(train_df['y'].values)
    
    for i in range(len(test_df)):
        baseline_pred = baseline.predict(1)[0]
        baseline_predictions.append(baseline_pred)
        
        actual = test_df['y'].iloc[i]
        all_train.append(actual)
        
        if i < len(test_df) - 1:
            try:
                baseline.fitted_model = baseline.fitted_model.append(
                    [actual], refit=False
                )
            except:
                pass
    
    baseline_rmse = calculate_rmse(test_df['y'].values, np.array(baseline_predictions))
    print(f"   Baseline ARIMA RMSE: {baseline_rmse:.6f}\n")
    
    # Unified GRM tahminleri
    print("   Unified GRM tahminleri yapılıyor...")
    unified_predictions = []
    
    model.eval()
    all_data = np.concatenate([train_data, val_data, test_data])
    
    with torch.no_grad():
        for i in range(len(test_df)):
            # Son seq_len kadar veriyi al
            start_idx = len(train_data) + len(val_data) + i - seq_len
            if start_idx < 0:
                start_idx = 0
            
            x_history = all_data[start_idx:start_idx + seq_len]
            x_history = x_history.reshape(1, seq_len, 1)
            
            _, _, final_pred = model.predict(x_history, device=device)
            unified_predictions.append(final_pred)
            
            if (i + 1) % 20 == 0:
                print(f"   Unified GRM: {i+1}/{len(test_df)}")
    
    unified_rmse = calculate_rmse(test_df['y'].values, np.array(unified_predictions))
    print(f"   Unified GRM RMSE: {unified_rmse:.6f}\n")
    
    # Karşılaştırma
    improvement = (baseline_rmse - unified_rmse) / baseline_rmse * 100
    
    print("=" * 80)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 80)
    print(f"Baseline ARIMA RMSE: {baseline_rmse:.6f}")
    print(f"Unified GRM RMSE:    {unified_rmse:.6f}")
    print(f"İyileşme:            {improvement:+.2f}%")
    print("=" * 80 + "\n")
    
    # Sonuçları kaydet
    results_file = os.path.join(OUTPUT_PATHS['results'], 'unified_grm_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("UNIFIED GRM TEST SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Baseline ARIMA RMSE: {baseline_rmse:.6f}\n")
        f.write(f"  Unified GRM RMSE:    {unified_rmse:.6f}\n")
        f.write(f"  İyileşme:            {improvement:+.2f}%\n\n")
        f.write("EĞİTİM PARAMETRELERİ:\n")
        f.write(f"  Epochs: {len(history['train_loss'])}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"  Best Val Loss: {min(history['val_loss']):.6f}\n")
    
    print(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    print("=" * 80)
    print("[SUCCESS] UNIFIED GRM TEST TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'baseline_rmse': baseline_rmse,
        'unified_rmse': unified_rmse,
        'improvement': improvement,
        'history': history
    }


if __name__ == '__main__':
    results = run_unified_grm_test()

