# GRM (Gravitational Residual Model) - FAZE 1

## 📖 Proje Hakkında

Bu proje, **Kütleçekimsel Artık Modeli (Gravitational Residual Model - GRM)** hipotezini test etmek için geliştirilmiş bir simülasyon çerçevesidir. GRM, zaman serisi analizinde artıkları "gürültü" olarak değil, Genel Görelilik teorisindeki kütleçekimsel etkilerden ilham alan bir yaklaşımla modellenebilir "yapısal bilgi" olarak ele alır.

**FAZE 1**, Schwarzschild metriğinden ilham alan basit bir implementasyon içerir:
- Sentetik veri oluşturma (kontrollü şoklar)
- ARIMA baseline model
- Schwarzschild GRM (sadece kütle parametresi)
- Basit lineer bükülme fonksiyonu

## 🏗️ Proje Yapısı

```
.
├── data/                     # Veri dosyaları
├── models/                   # Model modülleri
│   ├── __init__.py
│   ├── data_generator.py     # Sentetik veri üretici
│   ├── baseline_model.py     # ARIMA baseline
│   ├── grm_model.py          # Schwarzschild GRM
│   ├── metrics.py            # Performans metrikleri
│   └── visualization.py      # Görselleştirme
├── results/                  # Simülasyon sonuçları
├── visualizations/           # Grafikler
├── config.py                 # Konfigürasyon
├── main_phase1.py            # Ana simülasyon scripti
├── requirements.txt          # Bağımlılıklar
└── README.md                 # Bu dosya
```

## 🚀 Kurulum

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Simülasyonu Çalıştırın

```bash
python main_phase1.py
```

## 📊 Simülasyon Adımları

1. **Sentetik Veri Oluşturma**: Trend, mevsimsellik ve kontrollü şoklar içeren zaman serisi
2. **Veri Bölme**: Train (60%), Validation (20%), Test (20%)
3. **Baseline ARIMA**: Grid search ile optimal parametre bulma
4. **Artık Analizi**: Ljung-Box ve ARCH-LM testleri
5. **GRM Modeli**: Schwarzschild bükülme fonksiyonu
6. **Model Değerlendirme**: RMSE, MAE, MAPE, R², Diebold-Mariano testi
7. **Görselleştirme**: Zaman serileri, artıklar, kütle evrimi, performans

## 📈 Çıktılar

Simülasyon tamamlandığında şu çıktılar üretilir:

### Veri
- `data/synthetic_data_phase1.csv`: Sentetik zaman serisi

### Sonuçlar
- `results/phase1_results.txt`: Detaylı simülasyon sonuçları

### Görselleştirmeler
- `visualizations/time_series_comparison.png`: Tahmin karşılaştırması
- `visualizations/residuals_comparison.png`: Artık analizi
- `visualizations/mass_evolution.png`: Kütle evrimi ve olay ufku
- `visualizations/performance_comparison.png`: Performans metrikleri

## ⚙️ Konfigürasyon

`config.py` dosyasında şu parametreler ayarlanabilir:

### Veri Parametreleri
- `n_samples`: Toplam gözlem sayısı (varsayılan: 500)
- `trend_coef`: Trend katsayısı (varsayılan: 0.05)
- `seasonal_period`: Mevsimsel periyot (varsayılan: 50)
- `noise_std`: Beyaz gürültü std sapması (varsayılan: 2.0)

### Şok Parametreleri
- `n_shocks`: Şok sayısı (varsayılan: 3)
- `shock_std`: Şok büyüklüğü (varsayılan: 20.0)
- `decay_rate`: Sönümleme oranı (varsayılan: 0.1)

### GRM Parametreleri
- `window_size`: Volatilite pencere boyutu (varsayılan: 20)
- `alpha_range`: Kütleçekimsel etkileşim katsayısı aralığı
- `beta_range`: Sönümleme hızı aralığı

## 🔬 Metodoloji

### Schwarzschild Bükülme Fonksiyonu

```
Γ(t+1) = α * M(t) * sign(ε(t)) * decay(τ)
```

Burada:
- `Γ(t)`: Bükülme etkisi
- `α`: Kütleçekimsel etkileşim katsayısı
- `M(t)`: Kütle (yerel volatilite) = variance(ε[t-w:t])
- `sign(ε(t))`: Şok yönü
- `decay(τ)`: Sönümleme = 1 / (1 + β*τ)

### Hibrit Model

```
Y_GRM(t) = Y_baseline(t) + Γ(t)
```

### Olay Ufku

Kritik kütle eşiği:
```
σ²_critical = quantile(M(t), 0.99)
```

## 📊 Değerlendirme Metrikleri

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Diebold-Mariano Testi**: İstatistiksel karşılaştırma

## 🎯 Hipotez

**H1**: GRM, baseline modele göre istatistiksel olarak anlamlı şekilde (p < 0.05) daha iyi tahmin performansı gösterir.

**H0**: GRM ve baseline model arasında anlamlı bir fark yoktur.

## 🔜 Gelecek Fazlar

### FAZE 2
- Kerr rejimi (dönme parametresi)
- Non-linear bükülme (tanh)
- Sönümleme optimizasyonu

### FAZE 3
- Gerçek veri testleri
- GARCH ile karşılaştırma
- Kapsamlı istatistiksel analizler

## 📝 Kod Standartları

Proje, **PEP 8** (kod stili) ve **PEP 257** (docstring) standartlarına uygun olarak yazılmıştır.

## 👥 Katkıda Bulunma

Bu proje, akademik bir hipotez testi projesidir. Sorularınız veya önerileriniz için lütfen iletişime geçin.

## 📄 Lisans

Bu proje, akademik araştırma amaçlı geliştirilmiştir.

---

**Not**: Bu FAZE 1 implementasyonudur. Basit Schwarzschild rejimi kullanır ve kontrollü test amaçlıdır.
