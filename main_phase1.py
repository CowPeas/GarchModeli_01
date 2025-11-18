"""
GRM (Gravitational Residual Model) - FAZE 1 Ana Simülasyon.

Bu script, Schwarzschild rejimi (sadece kütle parametresi) kullanarak
basit GRM simülasyonunu çalıştırır.

FAZE 1 Özellikleri:
- Sentetik veri üretimi
- ARIMA baseline model
- Schwarzschild GRM (kütle bazlı)
- Basit lineer bükülme fonksiyonu
"""

import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime

# Matplotlib backend'ini Agg'ye ayarla (GUI gerektirmez)
import matplotlib
matplotlib.use('Agg')

# Local imports
from models import (
    SyntheticDataGenerator,
    BaselineARIMA,
    SchwarzschildGRM,
    ModelEvaluator,
    ResultVisualizer
)
from config import (
    DATA_CONFIG,
    SHOCK_CONFIG,
    SPLIT_CONFIG,
    ARIMA_CONFIG,
    GRM_CONFIG,
    VIS_CONFIG,
    OUTPUT_PATHS
)

# Tüm uyarıları filtrele
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> tuple:
    """
    Veriyi train, validation ve test setlerine böler.
    
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


def run_phase1_simulation():
    """
    FAZE 1 simülasyonunu çalıştırır.
    
    Bu fonksiyon tüm simülasyon adımlarını içerir:
    1. Sentetik veri oluşturma
    2. Veri bölme
    3. Baseline ARIMA modeli eğitimi
    4. Artık analizi
    5. GRM modeli eğitimi
    6. Model değerlendirme
    7. Görselleştirme
    """
    print("\n" + "=" * 80)
    print("GRM (GRAVITATIONAL RESIDUAL MODEL) - FAZE 1 SİMÜLASYONU")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # ADIM 1: SENTETIK VERİ OLUŞTURMA
    # ========================================================================
    print("📊 ADIM 1: Sentetik Veri Oluşturma")
    print("-" * 80)
    
    data_gen = SyntheticDataGenerator(**DATA_CONFIG)
    df, metadata = data_gen.generate(**SHOCK_CONFIG)
    
    print(f"✓ Toplam gözlem sayısı: {len(df)}")
    print(f"✓ Şok sayısı: {metadata['n_shocks']}")
    print(f"✓ Şok pozisyonları: {metadata['shock_positions']}")
    print(f"✓ Seri istatistikleri:")
    print(f"  - Ortalama: {df['y'].mean():.2f}")
    print(f"  - Std Sapma: {df['y'].std():.2f}")
    print(f"  - Min: {df['y'].min():.2f}")
    print(f"  - Max: {df['y'].max():.2f}")
    
    # Veriyi kaydet
    data_path = os.path.join(OUTPUT_PATHS['data'], 'synthetic_data_phase1.csv')
    df.to_csv(data_path, index=False)
    print(f"✓ Veri kaydedildi: {data_path}\n")
    
    # ========================================================================
    # ADIM 2: VERİ BÖLME
    # ========================================================================
    print("📂 ADIM 2: Veri Bölme (Train/Val/Test)")
    print("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    
    print(f"✓ Train set: {len(train_df)} gözlem")
    print(f"✓ Validation set: {len(val_df)} gözlem")
    print(f"✓ Test set: {len(test_df)} gözlem\n")
    
    # ========================================================================
    # ADIM 3: BASELINE ARIMA MODELİ
    # ========================================================================
    print("🎯 ADIM 3: Baseline ARIMA Modeli")
    print("-" * 80)
    
    baseline_model = BaselineARIMA()
    
    # Grid search ile optimal parametreleri bul
    print("Grid Search ile optimal parametreler bulunuyor...")
    best_order = baseline_model.grid_search(
        train_df['y'],
        val_df['y'],
        p_range=ARIMA_CONFIG['p_range'],
        d_range=ARIMA_CONFIG['d_range'],
        q_range=ARIMA_CONFIG['q_range'],
        verbose=True
    )
    
    # Train + Val ile final modeli eğit
    combined_train = pd.concat([train_df['y'], val_df['y']])
    baseline_model.fit(combined_train, order=best_order)
    
    # Test seti üzerinde tahmin
    baseline_predictions = []
    for i in range(len(test_df)):
        pred = baseline_model.predict(steps=1)[0]
        baseline_predictions.append(pred)
        
        # Modeli gerçek değerle güncelle (one-step-ahead forecasting)
        if i < len(test_df) - 1:
            actual_value = test_df['y'].iloc[i]
            baseline_model.fitted_model = baseline_model.fitted_model.append(
                [actual_value], refit=False
            )
    
    baseline_predictions = np.array(baseline_predictions)
    
    # Artıkları hesapla
    train_residuals = baseline_model.get_residuals()
    test_residuals = test_df['y'].values - baseline_predictions
    
    # Artık diagnostics
    print("\n📊 Artık Analizi:")
    diagnostics = baseline_model.diagnose_residuals()
    print(f"  - Ljung-Box p-değeri: {diagnostics['ljung_box_pvalue']:.4f}")
    print(f"  - ARCH-LM p-değeri: {diagnostics['arch_lm_pvalue']:.4f}")
    print(f"  - Otokorelasyon tespit edildi: {diagnostics['autocorr_detected']}")
    print(f"  - Heteroskedastisite tespit edildi: "
          f"{diagnostics['heteroscedasticity_detected']}\n")
    
    # ========================================================================
    # ADIM 4: GRM MODELİ (SCHWARZSCHILD)
    # ========================================================================
    print("🌀 ADIM 4: GRM (Schwarzschild) Modeli")
    print("-" * 80)
    
    grm_model = SchwarzschildGRM(window_size=GRM_CONFIG['window_size'])
    
    # Parametreleri optimize et
    grm_model.fit(
        train_residuals,
        alpha_range=GRM_CONFIG['alpha_range'],
        beta_range=GRM_CONFIG['beta_range'],
        val_residuals=None  # Train üzerinde optimize et (basit versiyon)
    )
    
    # Test seti için GRM tahminleri
    # Train artıklarıyla kütle hesapla ve eşik belirle
    train_mass = grm_model.compute_mass(train_residuals)
    grm_model.compute_event_horizon(
        train_mass,
        quantile=GRM_CONFIG['shock_threshold_quantile']
    )
    
    # Test seti için kütle ve bükülme hesapla
    # Test artıklarını kullanarak (gerçekte bunlar bilinmez, ama FAZE 1 için basitleştirme)
    all_residuals = np.concatenate([train_residuals, test_residuals])
    test_mass = grm_model.compute_mass(all_residuals)[len(train_residuals):]
    test_curvature = grm_model.compute_curvature(test_residuals, test_mass)
    
    # GRM hibrit tahminleri
    grm_predictions = baseline_predictions + test_curvature
    
    # GRM diagnostics
    grm_diagnostics = grm_model.get_diagnostics()
    print(f"📊 GRM Model Bilgileri:")
    print(f"  - Optimal α: {grm_diagnostics['alpha']:.3f}")
    print(f"  - Optimal β: {grm_diagnostics['beta']:.3f}")
    print(f"  - Pencere boyutu: {grm_diagnostics['window_size']}")
    print(f"  - Olay ufku eşiği: {grm_diagnostics['shock_threshold']:.4f}")
    print(f"  - Algılanan şok sayısı: {grm_diagnostics['n_shocks_detected']}\n")
    
    # ========================================================================
    # ADIM 5: MODEL DEĞERLENDİRME
    # ========================================================================
    print("📈 ADIM 5: Model Değerlendirme ve Karşılaştırma")
    print("-" * 80)
    
    evaluator = ModelEvaluator()
    
    y_true = test_df['y'].values
    comparison = evaluator.compare_models(
        y_true,
        baseline_predictions,
        grm_predictions
    )
    
    evaluator.print_comparison(comparison)
    
    # Sonuçları kaydet
    results_path = os.path.join(OUTPUT_PATHS['results'], 'phase1_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRM FAZE 1 SİMÜLASYON SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"VERİ KONFIGÜRASYONU:\n")
        f.write(f"  - Toplam gözlem: {len(df)}\n")
        f.write(f"  - Şok sayısı: {metadata['n_shocks']}\n")
        f.write(f"  - Best ARIMA order: {best_order}\n\n")
        f.write(f"GRM PARAMETRELERI:\n")
        f.write(f"  - α: {grm_diagnostics['alpha']:.3f}\n")
        f.write(f"  - β: {grm_diagnostics['beta']:.3f}\n\n")
        f.write(f"PERFORMANS:\n")
        f.write(f"  Baseline RMSE: {comparison['baseline_metrics']['rmse']:.4f}\n")
        f.write(f"  GRM RMSE: {comparison['grm_metrics']['rmse']:.4f}\n")
        f.write(f"  İyileşme: {comparison['rmse_improvement_pct']:.2f}%\n")
        f.write(f"  DM p-değeri: {comparison['diebold_mariano_pvalue']:.4f}\n\n")
        f.write(f"SONUÇ: {'HİPOTEZ DESTEKLENDI' if comparison['grm_is_better'] else 'HİPOTEZ DESTEKLENMEDİ'}\n")
    
    print(f"✓ Sonuçlar kaydedildi: {results_path}\n")
    
    # ========================================================================
    # ADIM 6: GÖRSELLEŞTİRME
    # ========================================================================
    print("🎨 ADIM 6: Görselleştirme")
    print("-" * 80)
    
    visualizer = ResultVisualizer(
        style=VIS_CONFIG['style'],
        figsize=VIS_CONFIG['figure_size'],
        dpi=VIS_CONFIG['dpi']
    )
    
    # Tüm seriler için tam uzunlukta tahminler oluştur
    full_time = df['time'].values
    full_actual = df['y'].values
    
    # Train bölümü için baseline tahminler
    train_predictions = baseline_model.fitted_model.fittedvalues[:len(train_df)]
    
    # Tüm tahminleri birleştir
    full_baseline = np.concatenate([
        train_predictions,
        np.full(len(val_df), np.nan),
        baseline_predictions
    ])
    
    full_grm = np.concatenate([
        train_predictions,
        np.full(len(val_df), np.nan),
        grm_predictions
    ])
    
    # Grafik 1: Zaman serisi karşılaştırması
    vis_path1 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'time_series_comparison.png')
    visualizer.plot_time_series_comparison(
        full_time,
        full_actual,
        full_baseline,
        full_grm,
        shock_positions=metadata['shock_positions'],
        train_end=len(train_df) + len(val_df),
        save_path=vis_path1
    )
    
    # Grafik 2: Artıklar karşılaştırması
    test_time = test_df['time'].values
    grm_residuals = y_true - grm_predictions
    
    vis_path2 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'residuals_comparison.png')
    visualizer.plot_residuals_comparison(
        test_time,
        test_residuals,
        grm_residuals,
        save_path=vis_path2
    )
    
    # Grafik 3: Kütle evrimi
    full_mass = grm_model.compute_mass(all_residuals)
    test_mass_full = full_mass[len(train_residuals):]
    
    vis_path3 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'mass_evolution.png')
    visualizer.plot_mass_evolution(
        test_time,
        test_mass_full,
        grm_diagnostics['shock_threshold'],
        shock_positions=[sp for sp in metadata['shock_positions']
                        if sp >= len(train_df) + len(val_df)],
        detected_shocks=[st - len(train_residuals)
                        for st in grm_diagnostics['shock_times']],
        save_path=vis_path3
    )
    
    # Grafik 4: Performans karşılaştırması
    vis_path4 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'performance_comparison.png')
    visualizer.plot_performance_comparison(
        comparison,
        save_path=vis_path4
    )
    
    print("\n" + "=" * 80)
    print("✅ FAZE 1 SİMÜLASYONU TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Çıktılar:")
    print(f"  - Veri: {OUTPUT_PATHS['data']}/")
    print(f"  - Sonuçlar: {OUTPUT_PATHS['results']}/")
    print(f"  - Grafikler: {OUTPUT_PATHS['visualizations']}/")
    print("=" * 80 + "\n")
    
    return {
        'data': df,
        'metadata': metadata,
        'baseline_model': baseline_model,
        'grm_model': grm_model,
        'comparison': comparison,
        'diagnostics': grm_diagnostics
    }


if __name__ == "__main__":
    """Ana simülasyonu çalıştır."""
    results = run_phase1_simulation()

