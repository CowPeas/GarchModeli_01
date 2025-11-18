"""
Konfigürasyon dosyası - GRM Projesi FAZE 2.

Bu modül, Kerr rejimi için gerekli tüm parametreleri içerir.
PEP8 ve PEP257 standartlarına uygun olarak hazırlanmıştır.
"""

# Veri parametreleri (FAZE 1'den devam)
DATA_CONFIG = {
    'n_samples': 500,           # Toplam gözlem sayısı
    'trend_coef': 0.05,          # Trend katsayısı (β₁)
    'trend_intercept': 100.0,    # Trend kesişim noktası (β₀)
    'seasonal_amplitude': 5.0,   # Mevsimsel genlik
    'seasonal_period': 50,       # Mevsimsel periyot
    'noise_std': 2.0,            # Beyaz gürültü standart sapması
    'random_seed': 42            # Tekrarlanabilirlik için seed
}

# Şok parametreleri - daha karmaşık (FAZE 2)
SHOCK_CONFIG = {
    'n_shocks': 4,               # Şok sayısı (FAZE 1: 3)
    'shock_std': 25.0,           # Şok büyüklüğü (FAZE 1: 20.0)
    'decay_rate': 0.08,          # Sönümleme oranı (FAZE 1: 0.1)
    'shock_positions': [120, 220, 320, 420]  # Şok pozisyonları
}

# Veri bölme parametreleri
SPLIT_CONFIG = {
    'train_ratio': 0.6,          # Eğitim seti oranı
    'val_ratio': 0.2,            # Doğrulama seti oranı
    'test_ratio': 0.2            # Test seti oranı
}

# ARIMA model parametreleri
ARIMA_CONFIG = {
    'p_range': [0, 1, 2, 3],     # AR sırası arama aralığı
    'd_range': [0, 1],           # Fark alma sırası
    'q_range': [0, 1, 2, 3],     # MA sırası arama aralığı
    'max_iterations': 100,       # Maksimum iterasyon
    'method': 'lbfgs'            # Optimizasyon yöntemi
}

# GRM Schwarzschild parametreleri (karşılaştırma için)
SCHWARZSCHILD_CONFIG = {
    'window_size': 20,
    'alpha_range': [0.1, 0.5, 1.0, 2.0],
    'beta_range': [0.01, 0.05, 0.1],
    'shock_threshold_quantile': 0.99
}

# GRM Kerr parametreleri (FAZE 2 - YENİ)
KERR_CONFIG = {
    'window_size': 20,                              # Pencere boyutu
    'alpha_range': [0.1, 0.5, 1.0, 2.0, 5.0],      # Kütleçekimsel etkileşim (genişletilmiş)
    'beta_range': [0.01, 0.05, 0.1, 0.2],          # Sönümleme hızı (genişletilmiş)
    'gamma_range': [0, 0.5, 1.0, 1.5],             # Dönme etkisi (YENİ)
    'use_tanh': True,                               # Non-linear aktivasyon (YENİ)
    'regime': 'adaptive',                           # 'schwarzschild', 'kerr', veya 'adaptive' (YENİ)
    'shock_threshold_quantile': 0.99
}

# Görselleştirme parametreleri
VIS_CONFIG = {
    'figure_size': (16, 12),     # Figür boyutu (FAZE 2 için biraz daha büyük)
    'dpi': 100,                  # Çözünürlük
    'style': 'seaborn-v0_8-darkgrid',  # Stil
    'color_actual': '#2E86AB',   # Gerçek veri rengi
    'color_baseline': '#A23B72', # Baseline rengi
    'color_schwarzschild': '#C06C84',  # Schwarzschild rengi (YENİ)
    'color_kerr': '#F18F01',     # Kerr rengi (YENİ)
    'color_shock': '#C73E1D'     # Şok rengi
}

# Çıktı klasörleri
OUTPUT_PATHS = {
    'data': './data',
    'models': './models',
    'results': './results',
    'visualizations': './visualizations'
}

# Karşılaştırma ayarları (FAZE 2 - YENİ)
COMPARISON_CONFIG = {
    'models_to_compare': [
        'baseline',           # ARIMA baseline
        'schwarzschild',     # GRM Schwarzschild (FAZE 1)
        'kerr'               # GRM Kerr (FAZE 2)
    ],
    'metrics': ['rmse', 'mae', 'mape', 'r2'],
    'statistical_tests': ['diebold_mariano', 'arch_lm'],
    'ablation_study': True    # Ablasyon çalışması yap
}

# Ablasyon çalışması ayarları (FAZE 2 - YENİ)
ABLATION_CONFIG = {
    'test_variants': [
        {'name': 'Schwarzschild (Sadece Kütle)', 'gamma': 0, 'use_tanh': False},
        {'name': 'Kerr Linear (Kütle+Dönme)', 'gamma': 1.0, 'use_tanh': False},
        {'name': 'Schwarzschild Non-linear', 'gamma': 0, 'use_tanh': True},
        {'name': 'Kerr Non-linear (Tam)', 'gamma': 1.0, 'use_tanh': True}
    ]
}

