"""
GRM (Gravitational Residual Model) - FAZE 2 Ana Simülasyon.

Bu script, Kerr rejimi (kütle + dönme parametresi) kullanarak
gelişmiş GRM simülasyonunu çalıştırır.

FAZE 2 Özellikleri:
- Schwarzschild (FAZE 1) + Kerr (FAZE 2) karşılaştırması
- Dönme parametresi a(t) (otokorelasyon)
- Non-linear bükülme fonksiyonu (tanh)
- Adaptif rejim seçimi
- Kapsamlı ablasyon çalışması
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
    KerrGRM,
    ModelEvaluator,
    ResultVisualizer
)
from config_phase2 import (
    DATA_CONFIG,
    SHOCK_CONFIG,
    SPLIT_CONFIG,
    ARIMA_CONFIG,
    SCHWARZSCHILD_CONFIG,
    KERR_CONFIG,
    VIS_CONFIG,
    OUTPUT_PATHS,
    COMPARISON_CONFIG,
    ABLATION_CONFIG
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


def run_phase2_simulation():
    """
    FAZE 2 simülasyonunu çalıştırır.
    
    Bu fonksiyon tüm simülasyon adımlarını içerir:
    1. Sentetik veri oluşturma
    2. Veri bölme
    3. Baseline ARIMA modeli
    4. Schwarzschild GRM (FAZE 1 - karşılaştırma)
    5. Kerr GRM (FAZE 2 - yeni)
    6. Üç model karşılaştırma
    7. Ablasyon çalışması
    8. Kapsamlı görselleştirme
    """
    print("\n" + "=" * 80)
    print("GRM (GRAVITATIONAL RESIDUAL MODEL) - FAZE 2 SİMÜLASYONU")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Özellikler: Kerr Rejimi + Non-linear Aktivasyon + Ablasyon")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # ADIM 1: SENTETIK VERİ OLUŞTURMA
    # ========================================================================
    print("📊 ADIM 1: Sentetik Veri Oluşturma (FAZE 2)")
    print("-" * 80)
    
    data_gen = SyntheticDataGenerator(**DATA_CONFIG)
    df, metadata = data_gen.generate(**SHOCK_CONFIG)
    
    print(f"✓ Toplam gözlem sayısı: {len(df)}")
    print(f"✓ Şok sayısı: {metadata['n_shocks']}")
    print(f"✓ Şok pozisyonları: {metadata['shock_positions']}")
    print(f"✓ Seri istatistikleri:")
    print(f"  - Ortalama: {df['y'].mean():.2f}")
    print(f"  - Std Sapma: {df['y'].std():.2f}")
    
    # Veriyi kaydet
    data_path = os.path.join(OUTPUT_PATHS['data'], 'synthetic_data_phase2.csv')
    df.to_csv(data_path, index=False)
    print(f"✓ Veri kaydedildi: {data_path}\n")
    
    # ========================================================================
    # ADIM 2: VERİ BÖLME
    # ========================================================================
    print("📂 ADIM 2: Veri Bölme (Train/Val/Test)")
    print("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # ========================================================================
    # ADIM 3: BASELINE ARIMA MODELİ
    # ========================================================================
    print("🎯 ADIM 3: Baseline ARIMA Modeli")
    print("-" * 80)
    
    baseline_model = BaselineARIMA()
    best_order = baseline_model.grid_search(
        train_df['y'], val_df['y'],
        p_range=ARIMA_CONFIG['p_range'],
        d_range=ARIMA_CONFIG['d_range'],
        q_range=ARIMA_CONFIG['q_range'],
        verbose=True
    )
    
    combined_train = pd.concat([train_df['y'], val_df['y']])
    baseline_model.fit(combined_train, order=best_order)
    
    # Test tahminleri
    baseline_predictions = []
    for i in range(len(test_df)):
        pred = baseline_model.predict(steps=1)[0]
        baseline_predictions.append(pred)
        if i < len(test_df) - 1:
            baseline_model.fitted_model = baseline_model.fitted_model.append(
                [test_df['y'].iloc[i]], refit=False
            )
    
    baseline_predictions = np.array(baseline_predictions)
    train_residuals = baseline_model.get_residuals()
    test_residuals = test_df['y'].values - baseline_predictions
    
    print(f"\n✓ Baseline model eğitildi: ARIMA{best_order}\n")
    
    # ========================================================================
    # ADIM 4: SCHWARZSCHILD GRM (FAZE 1 - Karşılaştırma)
    # ========================================================================
    print("🌑 ADIM 4: Schwarzschild GRM (FAZE 1 - Karşılaştırma)")
    print("-" * 80)
    
    schwarzschild_model = SchwarzschildGRM(
        window_size=SCHWARZSCHILD_CONFIG['window_size']
    )
    
    schwarzschild_model.fit(
        train_residuals,
        alpha_range=SCHWARZSCHILD_CONFIG['alpha_range'],
        beta_range=SCHWARZSCHILD_CONFIG['beta_range']
    )
    
    # Test için bükülme hesapla
    all_residuals_s = np.concatenate([train_residuals, test_residuals])
    test_mass_s = schwarzschild_model.compute_mass(all_residuals_s)[len(train_residuals):]
    schwarzschild_model.compute_event_horizon(
        schwarzschild_model.compute_mass(train_residuals),
        quantile=SCHWARZSCHILD_CONFIG['shock_threshold_quantile']
    )
    test_curvature_s = schwarzschild_model.compute_curvature(
        test_residuals, test_mass_s
    )
    
    schwarzschild_predictions = baseline_predictions + test_curvature_s
    
    schwarz_diag = schwarzschild_model.get_diagnostics()
    print(f"✓ Schwarzschild: α={schwarz_diag['alpha']:.3f}, "
          f"β={schwarz_diag['beta']:.3f}\n")
    
    # ========================================================================
    # ADIM 5: KERR GRM (FAZE 2 - Yeni)
    # ========================================================================
    print("🌀 ADIM 5: Kerr GRM (FAZE 2 - Dönme + Non-linear)")
    print("-" * 80)
    
    kerr_model = KerrGRM(
        window_size=KERR_CONFIG['window_size'],
        use_tanh=KERR_CONFIG['use_tanh'],
        regime=KERR_CONFIG['regime']
    )
    
    kerr_model.fit(
        train_residuals,
        alpha_range=KERR_CONFIG['alpha_range'],
        beta_range=KERR_CONFIG['beta_range'],
        gamma_range=KERR_CONFIG['gamma_range'],
        verbose=True
    )
    
    # Test için bükülme hesapla
    all_residuals_k = np.concatenate([train_residuals, test_residuals])
    test_mass_k = kerr_model.compute_mass(all_residuals_k)[len(train_residuals):]
    test_spin_k = kerr_model.compute_spin(all_residuals_k)[len(train_residuals):]
    kerr_model.compute_event_horizon(
        kerr_model.compute_mass(train_residuals),
        quantile=KERR_CONFIG['shock_threshold_quantile']
    )
    test_curvature_k = kerr_model.compute_curvature(
        test_residuals, test_mass_k, test_spin_k, use_detected_regime=False
    )
    
    kerr_predictions = baseline_predictions + test_curvature_k
    
    kerr_diag = kerr_model.get_diagnostics()
    print(f"📊 Kerr Model Bilgileri:")
    print(f"  - Optimal α: {kerr_diag['alpha']:.3f}")
    print(f"  - Optimal β: {kerr_diag['beta']:.3f}")
    print(f"  - Optimal γ: {kerr_diag['gamma']:.3f}")
    print(f"  - Non-linear (tanh): {kerr_diag['use_tanh']}")
    print(f"  - Tespit edilen rejim: {kerr_diag['detected_regime']}")
    print(f"  - Ortalama |dönme|: {kerr_diag['avg_spin']:.3f}\n")
    
    # ========================================================================
    # ADIM 6: ÜÇ MODEL KARŞILAŞTIRMA
    # ========================================================================
    print("📈 ADIM 6: Üç Model Karşılaştırması")
    print("-" * 80)
    
    evaluator = ModelEvaluator()
    y_true = test_df['y'].values
    
    # Baseline vs Schwarzschild
    comp_baseline_schwarz = evaluator.compare_models(
        y_true, baseline_predictions, schwarzschild_predictions
    )
    
    # Baseline vs Kerr
    comp_baseline_kerr = evaluator.compare_models(
        y_true, baseline_predictions, kerr_predictions
    )
    
    # Schwarzschild vs Kerr
    comp_schwarz_kerr = evaluator.compare_models(
        y_true, schwarzschild_predictions, kerr_predictions
    )
    
    print("\n" + "=" * 80)
    print("ÜÇ MODEL PERFORMANS TABLOSU")
    print("=" * 80)
    
    baseline_metrics = comp_baseline_kerr['baseline_metrics']
    schwarz_metrics = evaluator.evaluate_model(
        y_true, schwarzschild_predictions, "Schwarzschild"
    )
    kerr_metrics = comp_baseline_kerr['grm_metrics']
    
    print(f"\n{'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R²':>10}")
    print("-" * 80)
    print(f"{'Baseline':<20} {baseline_metrics['rmse']:>10.4f} "
          f"{baseline_metrics['mae']:>10.4f} {baseline_metrics['mape']:>10.2f} "
          f"{baseline_metrics['r2']:>10.4f}")
    print(f"{'Schwarzschild':<20} {schwarz_metrics['rmse']:>10.4f} "
          f"{schwarz_metrics['mae']:>10.4f} {schwarz_metrics['mape']:>10.2f} "
          f"{schwarz_metrics['r2']:>10.4f}")
    print(f"{'Kerr':<20} {kerr_metrics['rmse']:>10.4f} "
          f"{kerr_metrics['mae']:>10.4f} {kerr_metrics['mape']:>10.2f} "
          f"{kerr_metrics['r2']:>10.4f}")
    
    print("\n" + "=" * 80)
    print("İYİLEŞME YÜZD ELERİ (Baseline'a göre)")
    print("=" * 80)
    schwarz_imp = ((baseline_metrics['rmse'] - schwarz_metrics['rmse']) /
                   baseline_metrics['rmse'] * 100)
    kerr_imp = ((baseline_metrics['rmse'] - kerr_metrics['rmse']) /
                baseline_metrics['rmse'] * 100)
    print(f"Schwarzschild: {schwarz_imp:+.2f}%")
    print(f"Kerr:          {kerr_imp:+.2f}%")
    
    print("\n" + "=" * 80)
    print("DİEBOLD-MARIANO TEST SONUÇLARI")
    print("=" * 80)
    print(f"Schwarzschild vs Baseline: p = {comp_baseline_schwarz['diebold_mariano_pvalue']:.4f}")
    print(f"Kerr vs Baseline:          p = {comp_baseline_kerr['diebold_mariano_pvalue']:.4f}")
    print(f"Kerr vs Schwarzschild:     p = {comp_schwarz_kerr['diebold_mariano_pvalue']:.4f}")
    print("=" * 80 + "\n")
    
    # Sonuçları kaydet
    results_path = os.path.join(OUTPUT_PATHS['results'], 'phase2_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRM FAZE 2 SİMÜLASYON SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Baseline RMSE: {baseline_metrics['rmse']:.4f}\n")
        f.write(f"  Schwarzschild RMSE: {schwarz_metrics['rmse']:.4f} ({schwarz_imp:+.2f}%)\n")
        f.write(f"  Kerr RMSE: {kerr_metrics['rmse']:.4f} ({kerr_imp:+.2f}%)\n\n")
        f.write("KERR PARAMETRELERİ:\n")
        f.write(f"  α: {kerr_diag['alpha']:.3f}\n")
        f.write(f"  β: {kerr_diag['beta']:.3f}\n")
        f.write(f"  γ: {kerr_diag['gamma']:.3f}\n")
        f.write(f"  Rejim: {kerr_diag['detected_regime']}\n\n")
        f.write(f"SONUÇ: Kerr GRM, Schwarzschild'e göre ")
        if comp_schwarz_kerr['diebold_mariano_pvalue'] < 0.05:
            f.write("İSTATİSTİKSEL OLARAK ANLAMLI şekilde daha iyi\n")
        else:
            f.write("anlamlı bir fark göstermedi\n")
    
    print(f"✓ Sonuçlar kaydedildi: {results_path}\n")
    
    # ========================================================================
    # ADIM 7: GÖRSELLEŞTİRME
    # ========================================================================
    print("🎨 ADIM 7: Görselleştirme (FAZE 2)")
    print("-" * 80)
    
    visualizer = ResultVisualizer(
        style=VIS_CONFIG['style'],
        figsize=VIS_CONFIG['figure_size'],
        dpi=VIS_CONFIG['dpi']
    )
    
    # Tam seriler için tahminler
    full_time = df['time'].values
    full_actual = df['y'].values
    train_predictions = baseline_model.fitted_model.fittedvalues[:len(train_df)]
    
    full_baseline = np.concatenate([
        train_predictions, np.full(len(val_df), np.nan), baseline_predictions
    ])
    full_schwarz = np.concatenate([
        train_predictions, np.full(len(val_df), np.nan), schwarzschild_predictions
    ])
    full_kerr = np.concatenate([
        train_predictions, np.full(len(val_df), np.nan), kerr_predictions
    ])
    
    # Grafik 1: Üç model karşılaştırması
    vis_path1 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'three_model_comparison.png')
    visualizer.plot_three_model_comparison(
        full_time, full_actual, full_baseline, full_schwarz, full_kerr,
        shock_positions=metadata['shock_positions'],
        train_end=len(train_df) + len(val_df),
        save_path=vis_path1
    )
    
    # Grafik 2: Dönme evrimi
    test_time = test_df['time'].values
    vis_path2 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'spin_evolution.png')
    visualizer.plot_spin_evolution(
        test_time, test_spin_k, test_mass_k,
        save_path=vis_path2
    )
    
    # Grafik 3: Kütle evrimi (Kerr için)
    vis_path3 = os.path.join(OUTPUT_PATHS['visualizations'],
                             'mass_evolution_kerr.png')
    visualizer.plot_mass_evolution(
        test_time, test_mass_k, kerr_diag['shock_threshold'],
        shock_positions=[sp for sp in metadata['shock_positions']
                        if sp >= len(train_df) + len(val_df)],
        detected_shocks=[st - len(train_residuals)
                        for st in kerr_diag['shock_times']
                        if st >= len(train_residuals)],
        save_path=vis_path3
    )
    
    print("\n" + "=" * 80)
    print("✅ FAZE 2 SİMÜLASYONU TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Çıktılar:")
    print(f"  - Veri: {data_path}")
    print(f"  - Sonuçlar: {results_path}")
    print(f"  - Grafikler: {OUTPUT_PATHS['visualizations']}/")
    print("=" * 80 + "\n")
    
    return {
        'data': df,
        'metadata': metadata,
        'baseline_model': baseline_model,
        'schwarzschild_model': schwarzschild_model,
        'kerr_model': kerr_model,
        'comparisons': {
            'baseline_vs_schwarzschild': comp_baseline_schwarz,
            'baseline_vs_kerr': comp_baseline_kerr,
            'schwarzschild_vs_kerr': comp_schwarz_kerr
        },
        'diagnostics': {
            'schwarzschild': schwarz_diag,
            'kerr': kerr_diag
        }
    }


if __name__ == "__main__":
    """Ana simülasyonu çalıştır."""
    results = run_phase2_simulation()

