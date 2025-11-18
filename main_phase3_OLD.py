# -*- coding: utf-8 -*-
"""
GRM (Gravitational Residual Model) - FAZE 3 Ana Simulasyon.

Bu script, gercek finansal veri uzerinde GRM modellerini test eder ve
GARCH gibi standart volatilite modelleriyle kapsamli karsilastirma yapar.

FAZE 3 Ozellikleri:
- Gercek finansal veri (Bitcoin, S&P 500, vb.)
- GARCH benchmark modeli
- 4 model karsilastirmasi (Baseline, GARCH, Schwarzschild, Kerr)
- Kapsamli istatistiksel testler
- Risk metrikleri (VaR, CVaR, Sharpe)
- Performans analizi
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
    load_popular_assets,
    BaselineARIMA,
    GARCHModel,
    SimpleVolatilityModel,
    SchwarzschildGRM,
    KerrGRM,
    ModelEvaluator,
    ResultVisualizer,
    AlternativeDataLoader,
    create_manual_download_guide
)
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    ARIMA_CONFIG,
    GARCH_CONFIG,
    SCHWARZSCHILD_CONFIG,
    KERR_CONFIG,
    VIS_CONFIG,
    OUTPUT_PATHS,
    COMPARISON_CONFIG,
    STATISTICAL_TEST_CONFIG,
    RISK_METRICS_CONFIG
)

# Uyarıları filtrele
warnings.filterwarnings('ignore')


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
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


def run_phase3_simulation():
    """
    FAZE 3 simülasyonunu çalıştırır.
    
    Bu fonksiyon gerçek veri üzerinde tüm modelleri test eder.
    """
    print("\n" + "=" * 80)
    print("GRM (GRAVITATIONAL RESIDUAL MODEL) - FAZE 3 SİMÜLASYONU")
    print("=" * 80)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Özellikler: Gerçek Veri + GARCH + Kapsamlı Analiz")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # ADIM 1: GERCEK VERI YUKLEME (CSV ÖNCELIGI + OTOMATIK + FALLBACK)
    # ========================================================================
    print("[VERI] ADIM 1: Gercek Finansal Veri Yukleme")
    print("-" * 80)
    
    loader = RealDataLoader()
    alt_loader = AlternativeDataLoader()
    df = None
    metadata = None
    
    # ---- YÖNTEM 1: MANUEL İNDİRİLEN CSV VARMI? ----
    csv_path = os.path.join(OUTPUT_PATHS['data'], f"{REAL_DATA_CONFIG['ticker']}.csv")
    
    if os.path.exists(csv_path):
        print(f"[OK] MANUEL CSV BULUNDU: {csv_path}")
        print("   (Bu en guvenilir yontem - otomatik indirme atlan iyor)\n")
        
        try:
            df = alt_loader.load_from_csv(
                filepath=csv_path,
                date_column='Date',
                price_column='Close'
            )
            
            metadata = {
                'asset': REAL_DATA_CONFIG['ticker'],
                'period': f"{REAL_DATA_CONFIG['start_date']} - {REAL_DATA_CONFIG['end_date']}",
                'n_samples': len(df),
                'start_date': df['date'].min(),
                'end_date': df['date'].max(),
                'mean_return': df['returns'].mean(),
                'std_return': df['returns'].std(),
                'data_type': 'manual_csv'
            }
            
            print(f"[OK] CSV'DEN YUKLEME BASARILI!")
            print(f"   - Varlik: {metadata['asset']}")
            print(f"   - Gozlem: {len(df)}")
            print(f"   - Tarih: {metadata['start_date']} - {metadata['end_date']}\n")
            
        except Exception as csv_error:
            print(f"[HATA] CSV okuma hatasi: {str(csv_error)}")
            print("[DEVAM] Otomatik indirme deneniyor...\n")
    
    # ---- YONTEM 2: OTOMATIK INDIRME ----
    if df is None:
        print("[DOWNLOAD] OTOMATIK INDIRME BASLATILIYOR...")
        print(f"   Ticker: {REAL_DATA_CONFIG['ticker']}")
        print(f"   Tarih: {REAL_DATA_CONFIG['start_date']} - {REAL_DATA_CONFIG['end_date']}\n")
        
        try:
            df, metadata = loader.load_yahoo_finance(
                ticker=REAL_DATA_CONFIG['ticker'],
                start_date=REAL_DATA_CONFIG['start_date'],
                end_date=REAL_DATA_CONFIG['end_date'],
                column='Close',
                verify_ssl=False
            )
            
            # Basarili! Kaydet
            data_path = os.path.join(OUTPUT_PATHS['data'], 'real_data_phase3.csv')
            df.to_csv(data_path, index=False)
            print(f"[OK] Kaydedildi: {data_path}\n")
        
        except Exception as download_error:
            print(f"\n{'='*80}")
            print("[HATA] OTOMATIK INDIRME BASARISIZ!")
            print(f"{'='*80}\n")
            
            # Manuel indirme rehberi olustur
            create_manual_download_guide()
            
            print("[COZUM] 3 SECENEKSINIZ VAR:")
            print("=" * 80)
            
            print("\n[1] MANUEL INDIRME (ONERILEN - %100 CALISIR)")
            print("-" * 80)
            print(f"   a) Tarayicinizda acin:")
            print(f"      https://finance.yahoo.com/quote/{REAL_DATA_CONFIG['ticker']}/history")
            print(f"   b) Tarih secin: {REAL_DATA_CONFIG['start_date']} - {REAL_DATA_CONFIG['end_date']}")
            print(f"   c) 'Download' tiklayin")
            print(f"   d) CSV'yi kaydedin: {csv_path}")
            print(f"   e) Programi tekrar calistirin")
            print(f"\n   [INFO] Detayli rehber: data/MANUAL_DOWNLOAD_GUIDE.md")
            print(f"   [INFO] Veya: GERCEK_VERI_REHBERI.md")
            
            print("\n[2] GERCEKCI SENTETIK VERI (HIZLI TEST)")
            print("-" * 80)
            print("   Program otomatik devam edecek")
            print("   Kripto benzeri gercekci veri olusturulacak")
            print("   (GARCH + volatilite kumelenmesi + trendler)")
            
            print("\n[3] BASIT SENTETIK VERI (FAZE 2)")
            print("-" * 80)
            print("   python main_phase2.py")
            
            print("\n" + "=" * 80)
            print("[SORU] Ne yapmak istersiniz?")
            print("   [ENTER] = Gercekci sentetik veri ile devam")
            print("   [Ctrl+C] = Dur, manuel indireceğim")
            print("=" * 80 + "\n")
            
            # 5 saniye bekle
            import time
            print("[BEKLIYOR] 5 saniye bekleniyor...")
            for i in range(5, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            
            print("\n[DEVAM] GERCEKCI SENTETIK VERI OLUSTURULUYOR...")
            print("-" * 80)
            
            df = alt_loader.generate_realistic_crypto_data(
                days=730,  # 2 yil
                initial_price=30000.0 if 'BTC' in REAL_DATA_CONFIG['ticker'] else 100.0,
                volatility=0.03 if 'BTC' in REAL_DATA_CONFIG['ticker'] else 0.02
            )
            
            metadata = {
                'asset': f"REALISTIC_{REAL_DATA_CONFIG['ticker']}_SYNTHETIC",
                'period': 'simulated_2y',
                'n_samples': len(df),
                'start_date': df['date'].min(),
                'end_date': df['date'].max(),
                'mean_return': df['returns'].mean(),
                'std_return': df['returns'].std(),
                'data_type': 'realistic_synthetic',
                'note': 'Gercek veri benzeri sentetik (GARCH-like volatility clustering + trends)'
            }
            
            # Kaydet
            data_path = os.path.join(OUTPUT_PATHS['data'], 'realistic_synthetic_data.csv')
            df.to_csv(data_path, index=False)
            print(f"\n[OK] SENTETIK VERI HAZIR!")
            print(f"   - Kaydedildi: {data_path}")
            print(f"   - Gozlem: {len(df)}")
            print(f"   - Volatilite: {metadata['std_return']:.6f}\n")
    
    # Veri formatini duzelt (y sutunu ekle)
    if 'y' not in df.columns and 'returns' in df.columns:
        df['y'] = df['returns']
    elif 'y' not in df.columns and 'price' in df.columns:
        df['y'] = df['price'].pct_change()
        df = df.dropna()
    
    # ========================================================================
    # ADIM 2: VERI BOLME
    # ========================================================================
    print("\n[SPLIT] ADIM 2: Veri Bolme (Train/Val/Test)")
    print("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    print(f"✓ Train: {len(train_df)} ({SPLIT_CONFIG['train_ratio']*100:.0f}%)")
    print(f"✓ Val: {len(val_df)} ({SPLIT_CONFIG['val_ratio']*100:.0f}%)")
    print(f"✓ Test: {len(test_df)} ({SPLIT_CONFIG['test_ratio']*100:.0f}%)\n")
    
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
        verbose=False
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
    
    print(f"✓ Baseline: ARIMA{best_order}\n")
    
    # ========================================================================
    # ADIM 4: GARCH MODELİ (YENİ - FAZE 3)
    # ========================================================================
    print("📊 ADIM 4: GARCH Volatilite Modeli")
    print("-" * 80)
    
    try:
        garch_model = GARCHModel(
            p=GARCH_CONFIG['p_range'][0],
            q=GARCH_CONFIG['q_range'][0],
            model_type=GARCH_CONFIG['model_types'][0]
        )
        
        garch_model.fit(
            combined_train,
            mean_model=GARCH_CONFIG['mean_model'],
            ar_lags=GARCH_CONFIG['ar_lags'],
            verbose=True
        )
        
        # GARCH tahminleri (1-step ahead)
        garch_predictions = []
        garch_volatilities = []
        
        # Eğitim volatilitesi
        train_vol = garch_model.get_conditional_volatility()
        
        print(f"\n⚙️ GARCH ile test tahminleri yapılıyor...")
        
        # Basit 1-step ahead (GARCH re-fit gerektirir, burada basitleştirilmiş)
        for i in range(len(test_df)):
            # Son volatiliteyi kullan (basitleştirilmiş)
            last_vol = train_vol[-1] if len(train_vol) > 0 else 0.01
            
            # Basit tahmin: 0 ortalama + volatilite
            garch_predictions.append(0.0)
            garch_volatilities.append(last_vol)
        
        garch_predictions = np.array(garch_predictions)
        test_residuals_garch = test_df['y'].values - garch_predictions
        
        garch_diag = garch_model.get_diagnostics()
        print(f"✓ GARCH({garch_diag['p']},{garch_diag['q']})")
        print(f"  - AIC: {garch_diag.get('aic', 'N/A')}")
        print(f"  - Ortalama volatilite: {garch_diag.get('mean_conditional_vol', 0):.6f}\n")
        
    except Exception as e:
        print(f"⚠️ GARCH modeli başarısız: {str(e)}")
        print(f"   Basit volatilite modeli kullanılıyor...\n")
        
        simple_vol_model = SimpleVolatilityModel(window=20)
        simple_vol_model.fit(combined_train)
        garch_predictions = np.zeros(len(test_df))
        test_residuals_garch = test_df['y'].values
        garch_diag = {'model_type': 'Simple', 'fitted': False}
    
    # ========================================================================
    # ADIM 5: SCHWARZSCHILD GRM
    # ========================================================================
    print("🌑 ADIM 5: Schwarzschild GRM (FAZE 1)")
    print("-" * 80)
    
    schwarzschild_model = SchwarzschildGRM(
        window_size=SCHWARZSCHILD_CONFIG['window_size']
    )
    
    schwarzschild_model.fit(
        train_residuals,
        alpha_range=SCHWARZSCHILD_CONFIG['alpha_range'],
        beta_range=SCHWARZSCHILD_CONFIG['beta_range']
    )
    
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
    # ADIM 6: KERR GRM
    # ========================================================================
    print("🌀 ADIM 6: Kerr GRM (FAZE 2)")
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
    
    print(f"✓ Kerr: α={kerr_diag['alpha']:.3f}, β={kerr_diag['beta']:.3f}, "
          f"γ={kerr_diag['gamma']:.3f}")
    print(f"  - Rejim: {kerr_diag['detected_regime']}\n")
    
    # ========================================================================
    # ADIM 7: KAPSAMLI MODEL KARŞILAŞTIRMASI
    # ========================================================================
    print("📈 ADIM 7: Dört Model Kapsamlı Karşılaştırma")
    print("=" * 80)
    
    evaluator = ModelEvaluator()
    y_true = test_df['y'].values
    
    # Her model için metrikler
    baseline_metrics = evaluator.evaluate_model(y_true, baseline_predictions, "Baseline")
    garch_metrics = evaluator.evaluate_model(y_true, garch_predictions, "GARCH")
    schwarz_metrics = evaluator.evaluate_model(y_true, schwarzschild_predictions, "Schwarzschild")
    kerr_metrics = evaluator.evaluate_model(y_true, kerr_predictions, "Kerr")
    
    # Performans tablosu
    print(f"\n{'Model':<20} {'RMSE':>12} {'MAE':>12} {'MAPE':>12} {'R²':>12}")
    print("-" * 80)
    for metrics in [baseline_metrics, garch_metrics, schwarz_metrics, kerr_metrics]:
        print(f"{metrics['model_name']:<20} {metrics['rmse']:>12.6f} "
              f"{metrics['mae']:>12.6f} {metrics['mape']:>12.2f} "
              f"{metrics['r2']:>12.4f}")
    
    # İyileşme yüzdeleri
    print("\n" + "=" * 80)
    print("İYİLEŞME YÜZDE LERİ (Baseline'a göre)")
    print("=" * 80)
    
    for name, metrics in [("GARCH", garch_metrics),
                          ("Schwarzschild", schwarz_metrics),
                          ("Kerr", kerr_metrics)]:
        imp = ((baseline_metrics['rmse'] - metrics['rmse']) /
               baseline_metrics['rmse'] * 100)
        print(f"{name:<20} {imp:>+8.2f}%")
    
    # İstatistiksel testler
    print("\n" + "=" * 80)
    print("DIEBOLD-MARIANO TEST SONUÇLARI")
    print("=" * 80)
    
    comparisons = {}
    model_pairs = [
        ("GARCH vs Baseline", garch_predictions),
        ("Schwarzschild vs Baseline", schwarzschild_predictions),
        ("Kerr vs Baseline", kerr_predictions),
        ("Kerr vs GARCH", kerr_predictions),
        ("Kerr vs Schwarzschild", kerr_predictions)
    ]
    
    for name, pred2 in model_pairs:
        if "GARCH" in name and "vs Baseline" in name:
            comp = evaluator.compare_models(y_true, baseline_predictions, pred2)
        elif "Schwarzschild" in name and "vs Baseline" in name:
            comp = evaluator.compare_models(y_true, baseline_predictions, pred2)
        elif "Kerr vs Baseline" in name:
            comp = evaluator.compare_models(y_true, baseline_predictions, pred2)
        elif "Kerr vs GARCH" in name:
            comp = evaluator.compare_models(y_true, garch_predictions, pred2)
        elif "Kerr vs Schwarzschild" in name:
            comp = evaluator.compare_models(y_true, schwarzschild_predictions, pred2)
        
        comparisons[name] = comp
        print(f"{name:<30} p = {comp['diebold_mariano_pvalue']:.4f}")
    
    print("=" * 80 + "\n")
    
    # Sonuçları kaydet
    results_path = os.path.join(OUTPUT_PATHS['results'], 'phase3_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        if metadata.get('data_type') == 'synthetic_fallback':
            f.write("GRM FAZE 3 - SENTETİK VERİ TEST SONUÇLARI (FALLBACK)\n")
        else:
            f.write("GRM FAZE 3 - GERÇEK VERİ TEST SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        
        f.write(f"Varlık: {metadata['asset']}\n")
        f.write(f"Veri Tipi: {metadata.get('data_type', 'real')}\n")
        f.write(f"Periyot: {metadata['period']}\n")
        f.write(f"Test gözlem sayısı: {len(test_df)}\n\n")
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Baseline RMSE: {baseline_metrics['rmse']:.6f}\n")
        f.write(f"  GARCH RMSE: {garch_metrics['rmse']:.6f}\n")
        f.write(f"  Schwarzschild RMSE: {schwarz_metrics['rmse']:.6f}\n")
        f.write(f"  Kerr RMSE: {kerr_metrics['rmse']:.6f}\n\n")
        f.write("İSTATİSTİKSEL TEST SONUÇLARI:\n")
        for name, comp in comparisons.items():
            f.write(f"  {name}: p = {comp['diebold_mariano_pvalue']:.4f}\n")
        
        if metadata.get('data_type') == 'synthetic_fallback':
            f.write("\n" + "=" * 80 + "\n")
            f.write("NOT: Gerçek veri indirilemediği için sentetik veri kullanıldı.\n")
            f.write("Gerçek veri testi için SSL sorununu çözün veya CSV kullanın.\n")
            f.write("=" * 80 + "\n")
    
    print(f"✓ Sonuçlar kaydedildi: {results_path}\n")
    
    print("\n" + "=" * 80)
    print("✅ FAZE 3 SİMÜLASYONU TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if metadata.get('data_type') == 'synthetic_fallback':
        print("\n⚠️  NOT: Sentetik veri kullanıldı (gerçek veri indirilemedi)")
        print("   ✅ Ancak tüm modeller başarıyla test edildi!")
        print("   💡 Gerçek veri için: SSL sorununu çözün veya CSV kullanın")
    
    print(f"\n📁 Çıktılar: {OUTPUT_PATHS['results']}/")
    print("=" * 80 + "\n")
    
    return {
        'data': df,
        'metadata': metadata,
        'models': {
            'baseline': baseline_model,
            'garch': garch_model if 'garch_model' in locals() else None,
            'schwarzschild': schwarzschild_model,
            'kerr': kerr_model
        },
        'metrics': {
            'baseline': baseline_metrics,
            'garch': garch_metrics,
            'schwarzschild': schwarz_metrics,
            'kerr': kerr_metrics
        },
        'comparisons': comparisons
    }


if __name__ == "__main__":
    """Ana simülasyonu çalıştır."""
    results = run_phase3_simulation()

