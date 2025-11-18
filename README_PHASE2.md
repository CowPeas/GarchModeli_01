# GRM (Gravitational Residual Model) - FAZE 2

## 📖 Proje Hakkında

**FAZE 2**, GRM projesinin gelişmiş versiyonudur. Schwarzschild metriğine (FAZE 1) ek olarak **Kerr metriğinden** ilham alan dönme parametresi ve non-linear aktivasyon fonksiyonları içerir.

### 🆕 FAZE 2 Yenilikleri

1. **Kerr Rejimi**: Dönme parametresi `a(t)` (otokorelasyon)
2. **Non-linear Bükülme**: `tanh` aktivasyon fonksiyonu
3. **Adaptif Rejim Seçimi**: Veri özelliklerine göre otomatik Schwarzschild/Kerr seçimi
4. **Ablasyon Çalışması**: Her bileşenin katkısını ayrı ayrı test etme
5. **Üç Model Karşılaştırması**: Baseline vs Schwarzschild vs Kerr

## 🌀 Kerr vs Schwarzschild

### Schwarzschild (FAZE 1)
```
Γ(t) = α * M(t) * sign(ε(t)) * decay(τ)
```
- Sadece **kütle** (volatilite) parametresi
- Şok büyüklüğünü modellerler
- Basit lineer bükülme

### Kerr (FAZE 2)
```
Γ(t) = tanh(α * M(t) * [1 + γ*a(t)]) * decay(τ)
```
- **Kütle** `M(t)` = variance(ε[t-w:t])
- **Dönme** `a(t)` = ACF(ε[t-w:t], lag=1)
- **Non-linear** aktivasyon (tanh)
- Şok sonrası momentum ve otokorelasyonu yakalar

## 🏗️ Proje Yapısı (FAZE 2)

```
.
├── models/
│   ├── kerr_grm_model.py     # 🆕 Kerr GRM implementasyonu
│   ├── grm_model.py           # Schwarzschild (FAZE 1)
│   ├── baseline_model.py      # ARIMA baseline
│   ├── data_generator.py      # Sentetik veri
│   ├── metrics.py             # Performans metrikleri
│   └── visualization.py       # Görselleştirme (genişletilmiş)
├── config_phase2.py           # 🆕 FAZE 2 konfigürasyonu
├── main_phase2.py             # 🆕 FAZE 2 ana script
├── README_PHASE2.md           # Bu dosya
└── results/, visualizations/  # Çıktılar
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin (FAZE 1 ile aynı)

```bash
pip install -r requirements.txt
```

### 2. FAZE 2 Simülasyonunu Çalıştırın

```bash
python main_phase2.py
```

## 🎯 Kerr Model Parametreleri

### Kütle (M) - Yerel Volatilite
```python
M(t) = variance(ε[t-w:t])
```
- Artıkların hareketli pencere varyansı
- Şok büyüklüğünün bir ölçüsü

### Dönme (a) - Otokorelasyon
```python
a(t) = ACF(ε[t-w:t], lag=1)
```
- Artıkların birinci otokorelasyonu
- Şok sonrası momentum göstergesi
- Değer aralığı: `a(t) ∈ [-1, 1]`

### Non-linear Aktivasyon
```python
Γ(t) = tanh(α * M(t) * [1 + γ*a(t)]) * decay(τ)
```
- `tanh`: Aşırı tahminleri sınırlar
- Çıktı aralığı: `[-1, 1]`

### Adaptif Rejim Seçimi
```python
IF Ljung-Box test p < 0.05:
    regime = 'kerr'      # Otokorelasyon var
ELSE:
    regime = 'schwarzschild'  # Otokorelasyon yok
```

## 📊 Beklenen Çıktılar

### Grafikler (visualizations/)
1. **three_model_comparison.png** - Baseline vs Schwarzschild vs Kerr
2. **spin_evolution.png** - Dönme parametresi a(t) evrimi
3. **mass_evolution_kerr.png** - Kütle M(t) ve olay ufku

### Sonuçlar (results/)
- **phase2_results.txt** - Detaylı karşılaştırma raporu

## 📈 Örnek Çıktı

```
================================================================================
ÜÇ MODEL PERFORMANS TABLOSU
================================================================================

Model                    RMSE        MAE       MAPE         R²
--------------------------------------------------------------------------------
Baseline             12.3456     9.8765      8.45     0.8234
Schwarzschild        10.7890     8.5432      7.23     0.8567
Kerr                  9.8765     7.8901      6.54     0.8891

================================================================================
İYİLEŞME YÜZDE LERİ (Baseline'a göre)
================================================================================
Schwarzschild: +12.64%
Kerr:          +20.05%

================================================================================
DİEBOLD-MARIANO TEST SONUÇLARI
================================================================================
Schwarzschild vs Baseline: p = 0.0234
Kerr vs Baseline:          p = 0.0089
Kerr vs Schwarzschild:     p = 0.0456
================================================================================
```

## 🔬 Hipotez (FAZE 2)

**H1 (Kerr)**: Dönme parametresi eklenen Kerr GRM, sadece kütle kullanan Schwarzschild GRM'ye göre istatistiksel olarak anlamlı şekilde (p < 0.05) daha iyi tahmin performansı gösterir, özellikle otokorelasyon içeren artıklarda.

**H0**: Kerr ve Schwarzschild arasında anlamlı bir fark yoktur.

## ⚙️ Özelleştirme

### config_phase2.py

#### Kerr Parametreleri
```python
KERR_CONFIG = {
    'window_size': 20,                          # Pencere boyutu
    'alpha_range': [0.1, 0.5, 1.0, 2.0, 5.0],  # Kütle etkisi
    'beta_range': [0.01, 0.05, 0.1, 0.2],      # Sönümleme
    'gamma_range': [0, 0.5, 1.0, 1.5],         # Dönme etkisi
    'use_tanh': True,                           # Non-linear
    'regime': 'adaptive',                       # Otomatik seçim
}
```

#### Şok Parametreleri (Daha Karmaşık)
```python
SHOCK_CONFIG = {
    'n_shocks': 4,           # Daha fazla şok
    'shock_std': 25.0,       # Daha güçlü şoklar
    'decay_rate': 0.08,      # Yavaş sönümleme
}
```

## 🧪 Test Senaryoları

### Düşük Otokorelasyon (Schwarzschild Avantajlı)
```python
# Beyaz gürültüye yakın
DATA_CONFIG = {'noise_std': 5.0}
SHOCK_CONFIG = {'decay_rate': 0.5}  # Hızlı sönümleme
```
**Beklenti**: Schwarzschild ve Kerr benzer performans

### Yüksek Otokorelasyon (Kerr Avantajlı)
```python
# Momentum efekti
SHOCK_CONFIG = {
    'decay_rate': 0.02,         # Çok yavaş sönümleme
    'shock_autocorr': 0.5        # Yüksek otokorelasyon
}
```
**Beklenti**: Kerr belirgin şekilde daha iyi

## 📊 Model Karşılaştırması

| Özellik | Schwarzschild | Kerr |
|---------|---------------|------|
| Kütle (M) | ✅ | ✅ |
| Dönme (a) | ❌ | ✅ |
| Non-linear | ❌ | ✅ |
| Otokorelasyon | ❌ | ✅ |
| Parametre Sayısı | 2 (α, β) | 3 (α, β, γ) |
| Hesaplama Maliyeti | Düşük | Orta |

## 🎓 Bilimsel Katkı

FAZE 2'nin yenilikleri:

1. **Otokorelasyon Modelleme**: Şok sonrası momentum efektlerini yakalar
2. **Non-linear Sınırlama**: Aşırı tahminleri önler
3. **Adaptif Yaklaşım**: Veri özelliklerine göre model seçer
4. **Karşılaştırmalı Analiz**: İki metriğin de güçlü/zayıf yönlerini ortaya koyar

## 🔜 Sonraki Adımlar (FAZE 3)

- [ ] Gerçek finansal veri testleri
- [ ] GARCH/LSTM ile karşılaştırma
- [ ] Çoklu kara delik modeli
- [ ] Online learning ve adaptif parametreler

## 📝 Kod Standartları

- ✅ **PEP 8** uyumlu
- ✅ **PEP 257** docstring'ler
- ✅ Type hints
- ✅ Kapsamlı dokümantasyon

---

**🎉 FAZE 2 hazır! Kerr GRM ile daha gelişmiş modelleme.**

Sorularınız için `README.md` (FAZE 1) ve bu dosyaya başvurabilirsiniz.

