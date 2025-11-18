"""
Konfigürasyon dosyası - GRM Projesi FAZE 1.

Bu modül, simülasyon için gerekli tüm parametreleri içerir.
PEP8 ve PEP257 standartlarına uygun olarak hazırlanmıştır.
"""

# Veri parametreleri
DATA_CONFIG = {
    'n_samples': 500,           # Toplam gözlem sayısı
    'trend_coef': 0.05,          # Trend katsayısı (β₁)
    'trend_intercept': 100.0,    # Trend kesişim noktası (β₀)
    'seasonal_amplitude': 5.0,   # Mevsimsel genlik
    'seasonal_period': 50,       # Mevsimsel periyot
    'noise_std': 2.0,            # Beyaz gürültü standart sapması
    'random_seed': 42            # Tekrarlanabilirlik için seed
}

# Şok parametreleri
SHOCK_CONFIG = {
    'n_shocks': 3,               # Toplam şok sayısı
    'shock_std': 20.0,           # Şok büyüklüğü standart sapması
    'decay_rate': 0.1,           # Sönümleme oranı (τ)
    'shock_positions': [150, 250, 400]  # Şok pozisyonları (None ise rastgele)
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

# GRM (Schwarzschild) parametreleri
GRM_CONFIG = {
    'window_size': 20,           # Volatilite hesaplama pencere boyutu
    'alpha_range': [0.1, 0.5, 1.0, 2.0],     # Kütleçekimsel etkileşim katsayısı
    'beta_range': [0.01, 0.05, 0.1],         # Sönümleme hızı
    'shock_threshold_quantile': 0.99         # Olay ufku eşiği (99. yüzdelik)
}

# Görselleştirme parametreleri
VIS_CONFIG = {
    'figure_size': (15, 10),     # Figür boyutu
    'dpi': 100,                  # Çözünürlük
    'style': 'seaborn-v0_8-darkgrid',  # Stil
    'color_actual': '#2E86AB',   # Gerçek veri rengi
    'color_baseline': '#A23B72', # Baseline rengi
    'color_grm': '#F18F01',      # GRM rengi
    'color_shock': '#C73E1D'     # Şok rengi
}

# Çıktı klasörleri
OUTPUT_PATHS = {
    'data': './data',
    'models': './models',
    'results': './results',
    'visualizations': './visualizations'
}

