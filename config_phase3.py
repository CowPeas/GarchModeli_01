"""
Konfigürasyon dosyası - GRM Projesi FAZE 3.

Bu modül, gerçek veri testleri için gerekli tüm parametreleri içerir.
PEP8 ve PEP257 standartlarına uygun olarak hazırlanmıştır.
"""

# Gercek veri parametreleri (FAZE 3 - ENHANCED)
REAL_DATA_CONFIG = {
    'ticker': 'BTC-USD',         # Ticker: 'BTC-USD', 'ETH-USD', '^GSPC', '^DJI', 'AAPL'
    'asset': 'BTC-USD',          # Varlik (uyumluluk icin)
    'start_date': '2018-01-01',  # Baslangic tarihi (YYYY-MM-DD) - 5 YIL!
    'end_date': '2025-11-09',    # Bitis tarihi (YYYY-MM-DD)
    'period': '5y',              # Periyot: '6mo', '1y', '2y', '5y' - EXTENDED!
    'use_returns': True,         # Getiri kullan (fiyat yerine)
    'detect_volatility': True    # Volatilite analizi yap
}

# Alternatif varlıklar (test için)
AVAILABLE_ASSETS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    'EURUSD=X': 'EUR/USD',
    'GC=F': 'Gold Futures'
}

# Veri bölme parametreleri (FAZE 3 - OPTIMIZED FOR REGIME COVERAGE)
SPLIT_CONFIG = {
    'train_ratio': 0.50,         # 0.70 → 0.50 (daha fazla test için)
    'val_ratio': 0.15,           # Doğrulama
    'test_ratio': 0.35           # 0.15 → 0.35 (CRITICAL! Rejim coverage için)
}

# ARIMA model parametreleri (optimize edilmiş)
ARIMA_CONFIG = {
    'p_range': [0, 1, 2],        # Daha dar aralık (gerçek veri için)
    'd_range': [0, 1],
    'q_range': [0, 1, 2],
    'max_iterations': 200,       # Daha fazla iterasyon
    'method': 'lbfgs'
}

# GARCH model parametreleri (FAZE 3 - YENİ)
GARCH_CONFIG = {
    'p_range': [1],              # ARCH sırası
    'q_range': [1],              # GARCH sırası
    'model_types': ['GARCH'],    # Model tipleri: 'GARCH', 'EGARCH', 'GJR-GARCH'
    'mean_model': 'AR',          # Ortalama model: 'Constant', 'Zero', 'AR'
    'ar_lags': 1                 # AR gecikmeleri
}

# GRM Schwarzschild parametreleri (FAZE 1 + FAZE 4)
SCHWARZSCHILD_CONFIG = {
    'window_size': 20,
    'alpha_range': [0.1, 0.5, 1.0, 2.0],
    'beta_range': [0.01, 0.05, 0.1],
    'use_decay': True,
    'decay_beta_range': [0.01, 0.05, 0.1, 0.2],
    'shock_threshold_quantile': 0.95,
    'shock_detection_method': 'quantile'  # 'quantile' veya 'statistical'
}

# GRM Kerr parametreleri (FAZE 2 + FAZE 4)
KERR_CONFIG = {
    'window_size': 20,
    'alpha_range': [0.1, 0.5, 1.0, 2.0],
    'beta_range': [0.01, 0.05, 0.1],
    'gamma_range': [0, 0.5, 1.0],
    'use_tanh': True,
    'regime': 'adaptive',
    'use_decay': True,
    'decay_beta_range': [0.01, 0.05, 0.1, 0.2],
    'shock_threshold_quantile': 0.95,
    'shock_detection_method': 'quantile'  # 'quantile' veya 'statistical'
}

# Görselleştirme parametreleri
VIS_CONFIG = {
    'figure_size': (18, 12),     # Daha büyük (5 model için)
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'color_actual': '#2E86AB',
    'color_baseline': '#A23B72',
    'color_garch': '#6A4C93',    # GARCH rengi (YENİ)
    'color_schwarzschild': '#C06C84',
    'color_kerr': '#F18F01',
    'color_shock': '#C73E1D'
}

# Çıktı klasörleri
OUTPUT_PATHS = {
    'data': './data',
    'models': './models',
    'results': './results',
    'visualizations': './visualizations'
}

# Karşılaştırma ayarları (FAZE 3 - genişletilmiş)
COMPARISON_CONFIG = {
    'models_to_compare': [
        'baseline',           # ARIMA baseline
        'garch',             # GARCH (YENİ)
        'schwarzschild',     # GRM Schwarzschild (FAZE 1)
        'kerr'               # GRM Kerr (FAZE 2)
    ],
    'metrics': ['rmse', 'mae', 'mape', 'r2', 'mda'],  # MDA eklendi
    'statistical_tests': [
        'diebold_mariano',
        'arch_lm',
        'ljung_box'
    ],
    'rolling_window_analysis': True,   # Yuvarlanan pencere analizi
    'rolling_window_size': 50          # Pencere boyutu
}

# İstatistiksel test parametreleri (FAZE 3 - YENİ + ENHANCED)
STATISTICAL_TEST_CONFIG = {
    'significance_level': 0.05,       # Anlamlılık seviyesi
    'diebold_mariano_alternative': 'two-sided',
    'ljung_box_lags': 10,
    'arch_lm_lags': 5,
    'white_noise_test': True,         # Beyaz gürültü testi
    'normality_test': True,           # Normallik testi
    'stationarity_test': True,        # Durağanlık testi
    'bootstrap_n_iterations': 1000,   # Bootstrap iterasyon sayısı
    'bootstrap_confidence_level': 0.95  # Bootstrap güven seviyesi
}

# Cross-Validation Parametreleri (YENİ)
CV_CONFIG = {
    'method': 'expanding',  # 'expanding', 'walk-forward', 'blocked'
    'n_splits': 5,
    'test_size': 100,  # Her fold için test boyutu
    'gap': 0  # Train-test arası boşluk
}

# Advanced Metrics Config (YENİ)
METRICS_CONFIG = {
    'calculate_financial_metrics': False,  # Finansal metrikler (getiri için)
    'calculate_volatility_metrics': True,  # Volatilite metrikleri
    'calculate_directional_accuracy': True  # Yön doğruluğu
}

# Regime Analysis Config (YENİ)
REGIME_CONFIG = {
    'enable_regime_analysis': True,
    'dbscan_eps': 0.5,
    'dbscan_min_samples': 5,
    'auto_tune_dbscan': True  # Otomatik parametre ayarlama
}

# Markov analysis config (YENİ - FAZ 1)
MARKOV_ANALYSIS_CONFIG = {
    'enable': True,
    'coverage_confidence': 0.95,      # %95 güvenle tüm rejimleri örneklemek
    'min_regime_samples': 20,         # Her rejimde en az 20 gözlem
    'auto_adjust_split': True         # T_min'e göre otomatik split ayarlama
}

# Power analysis config (YENİ - FAZ 1)
POWER_ANALYSIS_CONFIG = {
    'enable': True,
    'target_power': 0.80,             # Hedef statistical power
    'alpha': 0.05,                    # Significance level
    'report_scenarios': True          # Farklı senaryolar için rapor
}

# Risk metrikleri (FAZE 3 - YENİ)
RISK_METRICS_CONFIG = {
    'calculate_var': True,            # Value at Risk
    'var_confidence': 0.95,           # VaR güven seviyesi
    'calculate_cvar': True,           # Conditional VaR
    'calculate_sharpe': True,         # Sharpe oranı
    'risk_free_rate': 0.02            # Risksiz faiz oranı (yıllık)
}

# Performans analizi (FAZE 3 - YENİ)
PERFORMANCE_ANALYSIS_CONFIG = {
    'compute_cumulative_returns': True,
    'compute_drawdown': True,         # Maksimum düşüş
    'compute_hit_ratio': True,        # İsabet oranı
    'direction_accuracy': True,       # Yön doğruluğu
    'volatility_forecast_accuracy': True  # Volatilite tahmin doğruluğu
}

