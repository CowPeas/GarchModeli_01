"""
Konfigürasyon dosyası - GRM Projesi FAZE 3.

Bu modül, gerçek veri testleri için gerekli tüm parametreleri içerir.
PEP8 ve PEP257 standartlarına uygun olarak hazırlanmıştır.
"""

# Gercek veri parametreleri (FAZE 3 - YENI)
REAL_DATA_CONFIG = {
    'ticker': 'BTC-USD',         # Ticker: 'BTC-USD', 'ETH-USD', '^GSPC', '^DJI', 'AAPL'
    'asset': 'BTC-USD',          # Varlik (uyumluluk icin)
    'start_date': '2023-11-10',  # Baslangic tarihi (YYYY-MM-DD)
    'end_date': '2025-11-09',    # Bitis tarihi (YYYY-MM-DD)
    'period': '2y',              # Periyot: '6mo', '1y', '2y', '5y'
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

# Veri bölme parametreleri (FAZE 3)
SPLIT_CONFIG = {
    'train_ratio': 0.7,          # Daha fazla eğitim (gerçek veri için)
    'val_ratio': 0.15,           # Doğrulama
    'test_ratio': 0.15           # Test
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

# GRM Schwarzschild parametreleri (FAZE 1)
SCHWARZSCHILD_CONFIG = {
    'window_size': 20,
    'alpha_range': [0.1, 0.5, 1.0, 2.0],
    'beta_range': [0.01, 0.05, 0.1],
    'shock_threshold_quantile': 0.99
}

# GRM Kerr parametreleri (FAZE 2)
KERR_CONFIG = {
    'window_size': 20,
    'alpha_range': [0.1, 0.5, 1.0, 2.0],
    'beta_range': [0.01, 0.05, 0.1],
    'gamma_range': [0, 0.5, 1.0],
    'use_tanh': True,
    'regime': 'adaptive',
    'shock_threshold_quantile': 0.99
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

# İstatistiksel test parametreleri (FAZE 3 - YENİ)
STATISTICAL_TEST_CONFIG = {
    'significance_level': 0.05,       # Anlamlılık seviyesi
    'diebold_mariano_alternative': 'two-sided',
    'ljung_box_lags': 10,
    'arch_lm_lags': 5,
    'white_noise_test': True,         # Beyaz gürültü testi
    'normality_test': True,           # Normallik testi
    'stationarity_test': True         # Durağanlık testi
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

