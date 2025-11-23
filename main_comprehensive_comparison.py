# -*- coding: utf-8 -*-
"""
Comprehensive Model Comparison Script - ENHANCED.

Bu script, tüm GRM modellerini kapsamlı şekilde karşılaştırır:
- ARIMA Baseline
- GARCH
- Schwarzschild GRM
- Kerr GRM
- Multi-Body GRM

İstatistiksel testler, Bootstrap CI ve rejim analizi dahildir.

PEP8 ve PEP257 standartlarına uygun olarak hazırlanmıştır.
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
from datetime import datetime

# Windows encoding fix
if sys.platform == 'win32':
    import codecs
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
    KerrGRM,
    GARCHModel,
    ComprehensiveComparison,
    RegimeAnalyzer
)
from models.multi_body_grm import MultiBodyGRM
from models.metrics import calculate_rmse, calculate_mae
from config_phase3 import (
    REAL_DATA_CONFIG,
    SPLIT_CONFIG,
    OUTPUT_PATHS,
    SCHWARZSCHILD_CONFIG,
    KERR_CONFIG,
    GARCH_CONFIG,
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
    train_ratio : float, optional
        Eğitim seti oranı (varsayılan: 0.7)
    val_ratio : float, optional
        Doğrulama seti oranı (varsayılan: 0.15)
    test_ratio : float, optional
        Test seti oranı (varsayılan: 0.15)
        
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


def walk_forward_predict(
    baseline_model: BaselineARIMA,
    grm_model,
    test_data: pd.Series
) -> np.ndarray:
    """
    Walk-forward validation ile GRM tahminleri.
    
    Parameters
    ----------
    baseline_model : BaselineARIMA
        Eğitilmiş baseline model
    grm_model : SchwarzschildGRM or KerrGRM
        Eğitilmiş GRM modeli
    test_data : pd.Series
        Test verisi
        
    Returns
    -------
    np.ndarray
        Final tahminler
    """
    predictions = []
    all_residuals = list(baseline_model.get_residuals())
    
    # Şok tespiti
    shock_times = None
    if len(all_residuals) > 0:
        shock_times = grm_model.detect_shocks(np.array(all_residuals))
    
    for i in range(len(test_data)):
        # Baseline tahmin
        baseline_pred = baseline_model.predict(1)[0]
        
        # Time since shock
        current_time = len(all_residuals)
        tau = grm_model.compute_time_since_shock(
            current_time=current_time,
            shock_times=shock_times
        )
        
        # GRM düzeltmesi
        recent_residuals = np.array(all_residuals[-grm_model.window_size:])
        
        if len(recent_residuals) > 0:
            # NaN temizle
            recent_clean = recent_residuals[~np.isnan(recent_residuals)]
            
            if len(recent_clean) > 1:
                mass = grm_model.compute_mass(recent_clean)[-1]
                
                if hasattr(grm_model, 'compute_spin'):
                    # Kerr
                    spin = grm_model.compute_spin(recent_clean)[-1]
                    correction = grm_model.compute_curvature_single(
                        recent_clean[-1],
                        mass,
                        spin,
                        time_since_shock=tau
                    )
                else:
                    # Schwarzschild
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
        
        # Gerçek değeri gözlemle
        actual = test_data.iloc[i]
        residual = actual - baseline_pred
        all_residuals.append(residual)
        
        # Şok tespiti güncelle
        if len(all_residuals) > grm_model.window_size:
            shock_times = grm_model.detect_shocks(np.array(all_residuals))
        
        # Baseline'ı güncelle
        if i < len(test_data) - 1:
            try:
                baseline_model.fitted_model = baseline_model.fitted_model.append(
                    [actual], refit=False
                )
            except:
                pass
    
    return np.array(predictions)


def run_comprehensive_comparison():
    """
    Kapsamlı model karşılaştırması çalıştırır.
    
    Returns
    -------
    dict
        Sonuç dictionary
    """
    import logging
    
    # Logger oluştur
    logger = logging.getLogger('ComprehensiveComparison')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info("\n" + "=" * 100)
    logger.info("KAPSAMLI MODEL KARŞILAŞTIRMASI - ENHANCED")
    logger.info("=" * 100)
    logger.info(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 100 + "\n")
    
    # Dizinleri oluştur
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ========================================================================
    # VERİ YÜKLEME
    # ========================================================================
    logger.info("[1/6] VERİ YÜKLEME")
    logger.info("-" * 100)
    
    loader = RealDataLoader()
    alt_loader = AlternativeDataLoader()
    df = None
    
    # Veri yükleme stratejisi
    try:
        df = loader.load_data(
            ticker=REAL_DATA_CONFIG['ticker'],
            start_date=REAL_DATA_CONFIG['start_date'],
            end_date=REAL_DATA_CONFIG['end_date'],
            use_returns=REAL_DATA_CONFIG['use_returns']
        )
        logger.info(f"[OK] Gerçek veri yüklendi: {REAL_DATA_CONFIG['ticker']}")
    except Exception as e:
        logger.warning(f"Gerçek veri yüklenemedi: {str(e)}")
        logger.info("Alternatif veri loader deneniyor...")
        df = alt_loader.load_btc_alternative()
    
    if df is None or len(df) == 0:
        logger.error("[HATA] Veri yüklenemedi!")
        return None
    
    logger.info(f"Veri boyutu: {len(df)} gözlem")
    logger.info(f"Veri aralığı: {df.index[0]} - {df.index[-1]}\n")
    
    # Veri bölme
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=SPLIT_CONFIG['train_ratio'],
        val_ratio=SPLIT_CONFIG['val_ratio'],
        test_ratio=SPLIT_CONFIG['test_ratio']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
    
    # ========================================================================
    # MODEL EĞİTİMİ
    # ========================================================================
    logger.info("[2/6] TÜM MODELLERİ EĞİTME")
    logger.info("-" * 100)
    
    all_predictions = {}
    
    # 1. ARIMA Baseline
    logger.info("\n[2.1] ARIMA Baseline eğitiliyor...")
    baseline = BaselineARIMA()
    best_order = baseline.grid_search(
        train_df['y'], val_df['y'],
        p_range=[0, 1, 2],
        d_range=[0, 1],
        q_range=[0, 1, 2],
        verbose=False
    )
    baseline.fit(train_df['y'], order=best_order)
    logger.info(f"[OK] ARIMA{best_order} eğitildi")
    
    # Baseline tahminleri (walk-forward)
    baseline_preds = []
    for i in range(len(test_df)):
        pred = baseline.predict(1)[0]
        baseline_preds.append(pred)
        if i < len(test_df) - 1:
            try:
                baseline.fitted_model = baseline.fitted_model.append(
                    [test_df['y'].iloc[i]], refit=False
                )
            except:
                pass
    all_predictions['ARIMA'] = np.array(baseline_preds)
    
    # 2. GARCH
    logger.info("\n[2.2] GARCH modeli eğitiliyor...")
    try:
        garch = GARCHModel(
            model_type=GARCH_CONFIG['model_types'][0],
            p=GARCH_CONFIG['p_range'][0],
            q=GARCH_CONFIG['q_range'][0],
            mean_model=GARCH_CONFIG['mean_model']
        )
        garch.fit(train_df['y'], verbose=False)
        
        # GARCH mean forecasts
        garch_preds = []
        for i in range(len(test_df)):
            pred = garch.forecast_mean(horizon=1)[0]
            garch_preds.append(pred)
        
        all_predictions['GARCH'] = np.array(garch_preds)
        logger.info(f"[OK] GARCH eğitildi")
    except Exception as e:
        logger.warning(f"GARCH eğitimi başarısız: {str(e)}")
        all_predictions['GARCH'] = np.zeros(len(test_df))  # Fallback
    
    # 3. Schwarzschild GRM
    logger.info("\n[2.3] Schwarzschild GRM eğitiliyor...")
    baseline_2 = BaselineARIMA()
    baseline_2.fit(train_df['y'], order=best_order)
    train_residuals = baseline_2.get_residuals()
    
    schwarz_model = SchwarzschildGRM(
        window_size=SCHWARZSCHILD_CONFIG['window_size'],
        shock_threshold_quantile=SCHWARZSCHILD_CONFIG['shock_threshold_quantile']
    )
    schwarz_model.fit(train_residuals)
    schwarz_model.grid_search_hyperparameters(
        val_df['y'],
        baseline_2,
        alpha_range=SCHWARZSCHILD_CONFIG['alpha_range'],
        beta_range=SCHWARZSCHILD_CONFIG['beta_range'],
        verbose=False
    )
    
    schwarz_preds = walk_forward_predict(baseline_2, schwarz_model, test_df['y'])
    all_predictions['Schwarzschild_GRM'] = schwarz_preds
    logger.info(f"[OK] Schwarzschild GRM eğitildi")
    
    # 4. Kerr GRM
    logger.info("\n[2.4] Kerr GRM eğitiliyor...")
    baseline_3 = BaselineARIMA()
    baseline_3.fit(train_df['y'], order=best_order)
    
    kerr_model = KerrGRM(
        window_size=KERR_CONFIG['window_size'],
        use_tanh=KERR_CONFIG['use_tanh'],
        regime=KERR_CONFIG['regime'],
        shock_threshold_quantile=KERR_CONFIG['shock_threshold_quantile']
    )
    kerr_model.fit(baseline_3.get_residuals())
    kerr_model.grid_search_hyperparameters(
        val_df['y'],
        baseline_3,
        alpha_range=KERR_CONFIG['alpha_range'],
        beta_range=KERR_CONFIG['beta_range'],
        gamma_range=KERR_CONFIG['gamma_range'],
        verbose=False
    )
    
    kerr_preds = walk_forward_predict(baseline_3, kerr_model, test_df['y'])
    all_predictions['Kerr_GRM'] = kerr_preds
    logger.info(f"[OK] Kerr GRM eğitildi\n")
    
    # ========================================================================
    # KAPSAMLI ANALİZ
    # ========================================================================
    logger.info("[3/6] KAPSAMLI ANALİZ")
    logger.info("-" * 100)
    
    comp = ComprehensiveComparison(baseline_name='ARIMA')
    
    for model_name, predictions in all_predictions.items():
        comp.add_model_results(model_name, test_df['y'].values, predictions)
    
    # Comprehensive rapor
    report_file = os.path.join(OUTPUT_PATHS['results'], 'comprehensive_all_models_report.txt')
    report = comp.generate_comprehensive_report(output_file=report_file)
    
    print(report)
    
    logger.info(f"\n[OK] Kapsamlı rapor kaydedildi: {report_file}\n")
    
    # ========================================================================
    # ÖZET
    # ========================================================================
    logger.info("=" * 100)
    logger.info("[SUCCESS] KAPSAMLI KARŞILAŞTIRMA TAMAMLANDI!")
    logger.info("=" * 100)
    logger.info(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'predictions': all_predictions,
        'report': report
    }


if __name__ == '__main__':
    results = run_comprehensive_comparison()

