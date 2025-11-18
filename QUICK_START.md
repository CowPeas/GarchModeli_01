# 🚀 GRM FAZE 1 - Hızlı Başlangıç Kılavuzu

## ⚡ Hızlı Kurulum ve Çalıştırma

### 1️⃣ Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2️⃣ Simülasyonu Çalıştırın

```bash
python main_phase1.py
```

### 3️⃣ Sonuçları İnceleyin

Simülasyon tamamlandığında şu klasörlerde çıktılar oluşacak:
- `data/` - Sentetik veri
- `results/` - Detaylı sonuçlar
- `visualizations/` - Grafikler

## 📋 Gerekli Kütüphaneler Listesi

Ana bağımlılıklar:
- `numpy` - Sayısal hesaplamalar
- `pandas` - Veri manipülasyonu
- `statsmodels` - ARIMA modeli
- `scipy` - İstatistiksel testler
- `matplotlib` - Görselleştirme
- `seaborn` - Gelişmiş grafikler
- `scikit-learn` - Yardımcı metrikler

## 🎮 Özelleştirme

`config.py` dosyasını düzenleyerek parametreleri değiştirebilirsiniz:

### Veri Boyutu Değiştirme

```python
DATA_CONFIG = {
    'n_samples': 1000,  # 500'den 1000'e çıkar
    ...
}
```

### Daha Fazla Şok Ekleme

```python
SHOCK_CONFIG = {
    'n_shocks': 5,  # 3'ten 5'e çıkar
    'shock_std': 30.0,  # Şokları güçlendir
    ...
}
```

### GRM Parametrelerini Genişletme

```python
GRM_CONFIG = {
    'window_size': 30,  # Pencereyi büyüt
    'alpha_range': [0.1, 0.5, 1.0, 2.0, 5.0],  # Daha fazla α dene
    ...
}
```

## 🔍 Çıktıları Anlama

### Performans Metrikleri

- **RMSE düşükse**: Model tahminleri gerçeğe yakın
- **R² yüksekse**: Model varyansın büyük kısmını açıklıyor
- **DM p-değeri < 0.05**: GRM, baseline'dan istatistiksel olarak daha iyi

### Grafikler

1. **time_series_comparison.png**: Gerçek vs Tahminler
   - Mavi: Gerçek veri
   - Mor: Baseline ARIMA
   - Turuncu: GRM
   - Kırmızı kesikli çizgiler: Şoklar

2. **residuals_comparison.png**: Artık analizi
   - Üst: Baseline artıkları
   - Alt: GRM artıkları
   - GRM artıkları daha az yapısal olmalı

3. **mass_evolution.png**: Kütle (volatilite) evrimi
   - Turuncu: Kütle M(t)
   - Kırmızı kesikli: Olay ufku eşiği
   - X işaretleri: Algılanan şoklar

4. **performance_comparison.png**: Bar grafikleri
   - Her metrik için baseline vs GRM karşılaştırması

## 🧪 Test Senaryoları

### Senaryo 1: Düşük Volatilite

```python
DATA_CONFIG = {
    'noise_std': 1.0,  # Düşük gürültü
    ...
}
SHOCK_CONFIG = {
    'shock_std': 10.0,  # Küçük şoklar
    ...
}
```

**Beklenti**: GRM ve baseline arasında küçük fark

### Senaryo 2: Yüksek Volatilite

```python
DATA_CONFIG = {
    'noise_std': 5.0,  # Yüksek gürültü
    ...
}
SHOCK_CONFIG = {
    'shock_std': 50.0,  # Büyük şoklar
    ...
}
```

**Beklenti**: GRM, baseline'dan belirgin şekilde daha iyi

### Senaryo 3: Çok Sayıda Şok

```python
SHOCK_CONFIG = {
    'n_shocks': 10,  # Çok sayıda şok
    'shock_positions': None,  # Rastgele yerleştir
    ...
}
```

**Beklenti**: GRM'nin şok algılama mekanizması aktif çalışır

## ❓ Sık Karşılaşılan Sorunlar

### Sorun: ModuleNotFoundError

**Çözüm**: requirements.txt'i yükleyin
```bash
pip install -r requirements.txt
```

### Sorun: Grafikler görünmüyor

**Çözüm**: `main_phase1.py` içinde `plt.show()` satırlarını kontrol edin veya sadece kaydedilen PNG dosyalarına bakın.

### Sorun: ARIMA convergence hatası

**Çözüm**: `config.py` içinde ARIMA parametre aralıklarını daraltın:
```python
ARIMA_CONFIG = {
    'p_range': [0, 1, 2],  # 3'ü kaldır
    'd_range': [0, 1],
    'q_range': [0, 1, 2],  # 3'ü kaldır
}
```

## 📊 Örnek Çıktı

```
================================================================================
MODEL KARŞILAŞTIRMA SONUÇLARI
================================================================================

📊 BASELINE MODEL:
   RMSE  : 12.3456
   MAE   : 9.8765
   MAPE  : 8.45%
   R²    : 0.8234

🌀 GRM MODEL:
   RMSE  : 10.2345
   MAE   : 8.1234
   MAPE  : 7.12%
   R²    : 0.8756

📈 İYİLEŞME:
   RMSE  : +17.13%
   MAE   : +17.76%

📊 DIEBOLD-MARIANO TESTİ:
   İstatistik: -2.3456
   P-değeri  : 0.0189

🎯 SONUÇ:
   ✓ GRM, baseline modele göre İSTATİSTİKSEL OLARAK ANLAMLI
     şekilde daha iyi performans gösterdi (p < 0.05)
================================================================================
```

## 🎯 Başarı Kriterleri

Hipotezin desteklendiği kabul edilir eğer:
1. ✅ RMSE iyileşmesi > %5
2. ✅ Diebold-Mariano p-değeri < 0.05
3. ✅ GRM artıklarında yapısal bilgi azalmış (ARCH test p > 0.05)

## 🆘 Yardım

Sorun yaşarsanız:
1. `config.py` parametrelerini varsayılana döndürün
2. `data/`, `results/`, `visualizations/` klasörlerini temizleyin
3. Simülasyonu yeniden çalıştırın

---

**İyi Şanslar! 🚀**

