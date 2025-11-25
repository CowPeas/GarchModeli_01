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
    SchwarzschildGRM
)
from models.multi_body_grm import MultiBodyGRM
from models.metrics import calculate_rmse, calculate_mae
from models.statistical_tests import StatisticalTests
from models.advanced_metrics import AdvancedMetrics, BootstrapCI
from models.comprehensive_comparison import ComprehensiveComparison
from models.regime_analysis import RegimeAnalyzer, analyze_regime_diversity
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS,
    SCHWARZSCHILD_CONFIG,
    STATISTICAL_TEST_CONFIG,
    REGIME_CONFIG
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
            # Progress logging handled by main logger
            pass
        
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
    import logging
    
    # Logger oluştur
    logger = logging.getLogger('MultiBodyGRM')
    logger.setLevel(logging.INFO)
    
    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-BODY GRM TEST - FAZE 6")
    logger.info("=" * 80)
    logger.info(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")
    
    # Dizinleri oluştur
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ========================================================================
    # ADIM 1: VERİ YÜKLEME
    # ========================================================================
    logger.info("[ADIM 1] VERİ YÜKLEME")
    logger.info("-" * 80)
    
    loader = RealDataLoader()
    alt_loader = AlternativeDataLoader()
    df = None
    
    # Manuel CSV kontrol
    csv_path = os.path.join(OUTPUT_PATHS['data'], f"{REAL_DATA_CONFIG['ticker']}.csv")
    
    if os.path.exists(csv_path):
        logger.info(f"[OK] MANUEL CSV BULUNDU: {csv_path}")
        try:
            df = alt_loader.load_from_csv(
                filepath=csv_path,
                date_column='Date',
                price_column='Close'
            )
            logger.info(f"[OK] CSV'DEN YÜKLEME BAŞARILI! ({len(df)} gözlem)\n")
        except Exception as e:
            logger.error(f"[HATA] CSV okuma hatası: {str(e)}\n")
    
    # Otomatik indirme
    if df is None:
        logger.info("[DOWNLOAD] OTOMATIK İNDİRME BAŞLATILIYOR...")
        try:
            df, metadata = loader.load_yahoo_finance(
                ticker=REAL_DATA_CONFIG['ticker'],
                start_date=REAL_DATA_CONFIG['start_date'],
                end_date=REAL_DATA_CONFIG['end_date'],
                column='Close',
                verify_ssl=False
            )
            logger.info(f"[OK] Otomatik indirme başarılı! ({len(df)} gözlem)\n")
        except Exception as e:
            logger.warning(f"[HATA] Otomatik indirme başarısız: {str(e)}")
            logger.info("[FALLBACK] Gerçekçi sentetik veri oluşturuluyor...")
            
            df = alt_loader.generate_realistic_crypto_data(
                days=730,
                initial_price=30000.0 if 'BTC' in REAL_DATA_CONFIG['ticker'] else 100.0,
                volatility=0.03
            )
            logger.info(f"[OK] Sentetik veri hazır! ({len(df)} gözlem)\n")
    
    # Veri formatını düzelt
    if 'y' not in df.columns and 'returns' in df.columns:
        df['y'] = df['returns']
    elif 'y' not in df.columns and 'price' in df.columns:
        df['y'] = df['price'].pct_change()
        df = df.dropna()
    
    # ========================================================================
    # ADIM 2: VERİ BÖLME
    # ========================================================================
    logger.info("[ADIM 2] VERİ BÖLME (Train/Val/Test)")
    logger.info("-" * 80)
    
    train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)
    logger.info(f"[OK] Train: {len(train_df)} gözlem (%{SPLIT_CONFIG['train_ratio']*100:.0f})")
    logger.info(f"[OK] Val:   {len(val_df)} gözlem (%{SPLIT_CONFIG['val_ratio']*100:.0f})")
    logger.info(f"[OK] Test:  {len(test_df)} gözlem (%{SPLIT_CONFIG['test_ratio']*100:.0f})\n")
    
    # ========================================================================
    # ADIM 3: BASELINE MODEL VE REZİDÜELLER
    # ========================================================================
    logger.info("[ADIM 3] BASELINE MODEL VE REZİDÜELLER")
    logger.info("-" * 80)
    
    logger.info("Baseline ARIMA modeli oluşturuluyor...")
    baseline = BaselineARIMA()
    
    # Grid search
    logger.info("Grid search başlatılıyor (p, d, q parametreleri)...")
    best_order = baseline.grid_search(
        train_df['y'], val_df['y'],
        p_range=[0, 1, 2],
        d_range=[0, 1],
        q_range=[0, 1, 2],
        verbose=False
    )
    logger.info(f"[OK] En iyi ARIMA parametreleri: {best_order}")
    
    # Fit
    logger.info("Baseline model eğitiliyor...")
    baseline.fit(train_df['y'], order=best_order)
    train_residuals = baseline.get_residuals()
    
    logger.info(f"[OK] Baseline: ARIMA{best_order}")
    logger.info(f"[OK] Train rezidüelleri: {len(train_residuals)} gözlem\n")
    
    # ========================================================================
    # ADIM 4: MULTI-BODY GRM EĞİTİMİ
    # ========================================================================
    logger.info("[ADIM 4] MULTI-BODY GRM EĞİTİMİ")
    logger.info("-" * 80)
    
    window_size = SCHWARZSCHILD_CONFIG['window_size']
    logger.info(f"Pencere boyutu: {window_size}")
    logger.info("Multi-Body GRM modeli oluşturuluyor...")
    
    multi_body_model = MultiBodyGRM(
        window_size=window_size,
        eps=0.5,
        min_samples=10,
        use_decay=True
    )
    
    logger.info("Rejim tespiti ve GRM eğitimi başlatılıyor...")
    multi_body_model.fit(train_residuals)
    
    logger.info(f"[OK] Multi-Body GRM eğitildi!")
    logger.info(f"[OK] Tespit edilen rejim sayısı: {len(multi_body_model.body_params)}\n")
    
    # ========================================================================
    # ADIM 5: TEST VE KARŞILAŞTIRMA
    # ========================================================================
    logger.info("[ADIM 5] TEST VE KARŞILAŞTIRMA")
    logger.info("-" * 80)
    
    # Manuel fonksiyon (Schwarzschild)
    logger.info("Manuel fonksiyon (Schwarzschild) test ediliyor...")
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
                # NaN temizle
                recent_clean = recent_residuals[~np.isnan(recent_residuals)]
                if len(recent_clean) > 0:
                    mass = grm_model.compute_mass(recent_clean)[-1]
                correction = grm_model.compute_curvature_single(
                        recent_clean[-1], mass, time_since_shock=tau
                )
            else:
                correction = 0.0
        else:
                correction = 0.0
            
            # NaN kontrolü
        if np.isnan(baseline_pred) or np.isnan(correction):
                correction = 0.0
        if np.isnan(baseline_pred):
                baseline_pred = 0.0
            
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
    
    logger.info("Walk-forward validation ile tahminler yapılıyor...")
    manual_predictions = walk_forward_predict_grm_simple(
        baseline, manual_model, test_df['y']
    )
    manual_rmse = calculate_rmse(test_df['y'].values, manual_predictions)
    
    logger.info(f"[OK] Manuel fonksiyon RMSE: {manual_rmse:.6f}\n")
    
    # Multi-Body tahminleri
    logger.info("Multi-Body GRM tahminleri yapılıyor...")
    baseline_preds, grm_corrections, multi_body_predictions, regime_ids = \
        walk_forward_predict_multi_body(
            baseline, multi_body_model, test_df['y'], verbose=True
        )
    multi_body_rmse = calculate_rmse(test_df['y'].values, multi_body_predictions)
    
    logger.info(f"[OK] Multi-Body GRM RMSE: {multi_body_rmse:.6f}\n")
    
    # ========================================================================
    # ADIM 6: GELİŞMİŞ ANALİZLER (YENİ)
    # ========================================================================
    logger.info("[ADIM 6] GELİŞMİŞ İSTATİSTİKSEL ANALİZLER")
    logger.info("-" * 80)
    
    # 6.1 Rejim Analizi (Detaylı)
    logger.info("\n[6.1] DETAYLI REJİM ANALİZİ")
    logger.info("-" * 80)
    
    if REGIME_CONFIG['enable_regime_analysis']:
        regime_analyzer = RegimeAnalyzer()
        regime_analyzer.fit(test_df['y'].values, regime_ids)
        
        regime_summary = regime_analyzer.get_regime_summary()
        logger.info("\nRejim Özellikleri:")
        logger.info(regime_summary.to_string(index=False))
        
        # Rejim geçişleri
        transitions = regime_analyzer.get_regime_transitions()
        logger.info("\nRejim Geçişleri (Top 5):")
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
        for trans, count in sorted_transitions:
            logger.info(f"  {trans}: {count} kez")
        
        # Dataset karakterizasyonu
        dataset_char = regime_analyzer.characterize_dataset()
        logger.info(f"\nToplam Rejim Sayısı: {dataset_char['n_regimes']}")
        logger.info(f"Outlier Oranı: {dataset_char['outlier_ratio']*100:.1f}%")
        logger.info(f"Dominant Rejim: {dataset_char['dominant_regime']}")
        
        # Rejim analiz raporunu kaydet
        regime_report_file = os.path.join(OUTPUT_PATHS['results'], 'regime_analysis_report.txt')
        regime_analyzer.generate_report(output_file=regime_report_file)
        logger.info(f"\n[OK] Rejim analizi raporu kaydedildi: {regime_report_file}")
    
    # 6.2 İstatistiksel Anlamlılık Testleri
    logger.info("\n[6.2] İSTATİSTİKSEL ANLAMLILIK TESTLERİ")
    logger.info("-" * 80)
    
    # Hata serileri
    manual_errors = test_df['y'].values - manual_predictions
    multi_body_errors = test_df['y'].values - multi_body_predictions
    
    # Temizlik
    mask = ~(np.isnan(manual_errors) | np.isnan(multi_body_errors) | 
             np.isinf(manual_errors) | np.isinf(multi_body_errors))
    manual_errors_clean = manual_errors[mask]
    multi_body_errors_clean = multi_body_errors[mask]
    
    # Diebold-Mariano Test
    try:
        dm_stat, dm_pval = StatisticalTests.diebold_mariano_test(
            manual_errors_clean,
            multi_body_errors_clean,
            alternative=STATISTICAL_TEST_CONFIG['diebold_mariano_alternative']
        )
        logger.info(f"\nDiebold-Mariano Test:")
        logger.info(f"  Test İstatistiği: {dm_stat:.4f}")
        logger.info(f"  P-Değeri: {dm_pval:.4f}")
        if dm_pval < STATISTICAL_TEST_CONFIG['significance_level']:
            logger.info(f"  ✅ Multi-Body GRM, Manuel modelden İSTATİSTİKSEL OLARAK ANLAMLI şekilde farklı (α={STATISTICAL_TEST_CONFIG['significance_level']})")
        else:
            logger.info(f"  ⚠️  İki model arasında istatistiksel olarak ANLAMLI fark YOK (α={STATISTICAL_TEST_CONFIG['significance_level']})")
    except Exception as e:
        logger.warning(f"  Diebold-Mariano test hatası: {str(e)}")
    
    # ARCH-LM Test (Multi-Body residuals)
    try:
        arch_lm, arch_pval = StatisticalTests.arch_lm_test(
            multi_body_errors_clean,
            lags=STATISTICAL_TEST_CONFIG['arch_lm_lags']
        )
        logger.info(f"\nARCH-LM Test (Multi-Body Residuals):")
        logger.info(f"  LM İstatistiği: {arch_lm:.4f}")
        logger.info(f"  P-Değeri: {arch_pval:.4f}")
        if arch_pval < STATISTICAL_TEST_CONFIG['significance_level']:
            logger.info(f"  ⚠️  ARCH etkileri tespit edildi (heteroskedasticity var)")
        else:
            logger.info(f"  ✅ ARCH etkileri tespit EDİLEMEDİ (homoskedastic)")
    except Exception as e:
        logger.warning(f"  ARCH-LM test hatası: {str(e)}")
    
    # Ljung-Box Test (Multi-Body residuals)
    try:
        lb_stats, lb_pvals = StatisticalTests.ljung_box_test(
            multi_body_errors_clean,
            lags=STATISTICAL_TEST_CONFIG['ljung_box_lags']
        )
        logger.info(f"\nLjung-Box Test (Multi-Body Residuals, Lag {STATISTICAL_TEST_CONFIG['ljung_box_lags']}):")
        logger.info(f"  LB İstatistiği: {lb_stats[-1]:.4f}")
        logger.info(f"  P-Değeri: {lb_pvals[-1]:.4f}")
        if lb_pvals[-1] < STATISTICAL_TEST_CONFIG['significance_level']:
            logger.info(f"  ⚠️  Otokorelasyon tespit edildi (beyaz gürültü DEĞİL)")
        else:
            logger.info(f"  ✅ Otokorelasyon tespit EDİLEMEDİ (beyaz gürültü)")
    except Exception as e:
        logger.warning(f"  Ljung-Box test hatası: {str(e)}")
    
    # 6.3 Bootstrap Güven Aralıkları
    logger.info("\n[6.3] BOOTSTRAP GÜVEN ARALIKLARI")
    logger.info("-" * 80)
    
    try:
        boot = BootstrapCI(
            n_bootstrap=STATISTICAL_TEST_CONFIG['bootstrap_n_iterations'],
            confidence_level=STATISTICAL_TEST_CONFIG['bootstrap_confidence_level']
        )
        
        # RMSE farkı CI
        ci_result = boot.performance_difference_ci(
            test_df['y'].values[mask],
            manual_predictions[mask],
            multi_body_predictions[mask],
            metric='rmse'
        )
        
        logger.info(f"\nRMSE Farkı (Manuel - Multi-Body):")
        logger.info(f"  Ortalama Fark: {ci_result['mean_difference']:.6f}")
        logger.info(f"  %{STATISTICAL_TEST_CONFIG['bootstrap_confidence_level']*100:.0f} CI: [{ci_result['ci_lower']:.6f}, {ci_result['ci_upper']:.6f}]")
        logger.info(f"  İstatistiksel Anlamlı: {'Evet' if ci_result['is_significant'] else 'Hayır'}")
        logger.info(f"\n  Yorum: {ci_result['interpretation']}")
    except Exception as e:
        logger.warning(f"  Bootstrap CI hesaplama hatası: {str(e)}")
    
    # 6.4 Gelişmiş Metrikler
    logger.info("\n[6.4] GELİŞMİŞ PERFORMANS METRİKLERİ")
    logger.info("-" * 80)
    
    try:
        # Manuel model metrikleri
        manual_metrics = AdvancedMetrics.calculate_all_metrics(
            test_df['y'].values[mask],
            manual_predictions[mask]
        )
        
        # Multi-Body metrikleri
        multi_body_metrics = AdvancedMetrics.calculate_all_metrics(
            test_df['y'].values[mask],
            multi_body_predictions[mask]
        )
        
        logger.info("\nManuel GRM:")
        for key, value in manual_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        logger.info("\nMulti-Body GRM:")
        for key, value in multi_body_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
    except Exception as e:
        logger.warning(f"  Gelişmiş metrikler hatası: {str(e)}")
    
    logger.info("\n" + "=" * 80)
    
    # Karşılaştırma
    improvement = (manual_rmse - multi_body_rmse) / manual_rmse * 100
    
    logger.info("=" * 80)
    logger.info("KARŞILAŞTIRMA SONUÇLARI")
    logger.info("=" * 80)
    logger.info(f"Manuel Fonksiyon RMSE: {manual_rmse:.6f}")
    logger.info(f"Multi-Body GRM RMSE:   {multi_body_rmse:.6f}")
    logger.info(f"İyileşme:              {improvement:+.2f}%")
    logger.info("=" * 80 + "\n")
    
    # ========================================================================
    # ADIM 7: KAPSAMLI RAPOR OLUŞTURMA
    # ========================================================================
    logger.info("[ADIM 7] KAPSAMLI RAPOR OLUŞTURMA")
    logger.info("-" * 80)
    
    # Comprehensive Comparison kullanarak detaylı rapor
    comp = ComprehensiveComparison(baseline_name='Manuel_GRM')
    
    # Model sonuçlarını ekle
    comp.add_model_results('Manuel_GRM', test_df['y'].values, manual_predictions)
    comp.add_model_results('Multi_Body_GRM', test_df['y'].values, multi_body_predictions)
    
    # Comprehensive rapor oluştur
    comprehensive_report_file = os.path.join(OUTPUT_PATHS['results'], 'comprehensive_comparison_report.txt')
    comp.generate_comprehensive_report(output_file=comprehensive_report_file)
    logger.info(f"[OK] Kapsamlı karşılaştırma raporu kaydedildi: {comprehensive_report_file}\n")
    
    # Sonuçları kaydet (eski format)
    results_file = os.path.join(OUTPUT_PATHS['results'], 'multi_body_grm_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-BODY GRM TEST SONUÇLARI - ENHANCED\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PERFORMANS KARŞILAŞTIRMASI:\n")
        f.write(f"  Manuel Fonksiyon RMSE: {manual_rmse:.6f}\n")
        f.write(f"  Multi-Body GRM RMSE:   {multi_body_rmse:.6f}\n")
        f.write(f"  İyileşme:              {improvement:+.2f}%\n\n")
        
        # İstatistiksel testler
        f.write("İSTATİSTİKSEL ANLAMLILIK:\n")
        try:
            f.write(f"  Diebold-Mariano p-değeri: {dm_pval:.4f}\n")
            f.write(f"  İstatistiksel Anlamlı: {'Evet' if dm_pval < STATISTICAL_TEST_CONFIG['significance_level'] else 'Hayır'}\n\n")
        except:
            f.write("  Diebold-Mariano testi hesaplanamadı\n\n")
        
        # Bootstrap CI
        try:
            f.write("BOOTSTRAP GÜVEN ARALIKLARI:\n")
            f.write(f"  RMSE Farkı: {ci_result['mean_difference']:.6f}\n")
            f.write(f"  %{STATISTICAL_TEST_CONFIG['bootstrap_confidence_level']*100:.0f} CI: [{ci_result['ci_lower']:.6f}, {ci_result['ci_upper']:.6f}]\n")
            f.write(f"  Anlamlı: {'Evet' if ci_result['is_significant'] else 'Hayır'}\n\n")
        except:
            f.write("  Bootstrap CI hesaplanamadı\n\n")
        
        f.write("REJİM ANALİZİ:\n")
        unique_regimes, regime_counts = np.unique(regime_ids, return_counts=True)
        for regime_id, count in zip(unique_regimes, regime_counts):
            f.write(f"  Rejim {regime_id}: {count} gözlem (%{count/len(regime_ids)*100:.1f})\n")
        
        f.write("\nBODY PARAMETRELERİ:\n")
        for params in multi_body_model.body_params:
            beta_str = f"{params['beta']:.4f}" if params['beta'] else "N/A"
            f.write(f"  Body {params['body_id']}: "
                   f"α={params['alpha']:.4f}, "
                   f"β={beta_str}, "
                   f"n={params['n_samples']}\n")
        
        # Gelişmiş metrikler
        f.write("\nGELİŞMİŞ METRİKLER:\n")
        f.write("Manuel GRM:\n")
        try:
            for key, value in manual_metrics.items():
                f.write(f"  {key}: {value:.6f}\n")
        except:
            f.write("  Hesaplanamadı\n")
        
        f.write("\nMulti-Body GRM:\n")
        try:
            for key, value in multi_body_metrics.items():
                f.write(f"  {key}: {value:.6f}\n")
        except:
            f.write("  Hesaplanamadı\n")
    
    logger.info(f"[OK] Sonuçlar kaydedildi: {results_file}\n")
    
    logger.info("=" * 80)
    logger.info("[SUCCESS] MULTI-BODY GRM TEST TAMAMLANDI!")
    logger.info("=" * 80)
    logger.info(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'manual_rmse': manual_rmse,
        'multi_body_rmse': multi_body_rmse,
        'improvement': improvement,
        'regime_ids': regime_ids,
        'body_params': multi_body_model.body_params
    }


if __name__ == '__main__':
    results = run_multi_body_grm_test()

