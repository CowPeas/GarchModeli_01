# 🚀 GRM FAZE 2 - Hızlı Başlangıç

## ⚡ 3 Adımda Başlayın

### 1️⃣ Simülasyonu Çalıştırın

```bash
python main_phase2.py
```

### 2️⃣ Sonuçları İnceleyin

```bash
# Detaylı sonuçlar
cat results/phase2_results.txt

# Grafikler
ls visualizations/
```

### 3️⃣ Parametreleri Değiştirin

`config_phase2.py` dosyasını düzenleyin:

```python
# Daha fazla şok
SHOCK_CONFIG = {
    'n_shocks': 6,
    'shock_std': 30.0,
}

# Daha geniş parametre arama
KERR_CONFIG = {
    'gamma_range': [0, 0.5, 1.0, 1.5, 2.0],  # Daha fazla dönme değeri
}
```

## 🎯 Ne Beklemeli?

### ✅ İdeal Senaryo (Kerr Üstün)

```
📊 SONUÇLAR:
- Baseline RMSE:      12.34
- Schwarzschild RMSE: 10.78 (+12.6%)
- Kerr RMSE:           9.23 (+25.2%)

📈 Kerr vs Schwarzschild:
- İyileşme: +14.4%
- DM p-değeri: 0.018 ✅ (p < 0.05)

🎯 SONUÇ: Kerr, Schwarzschild'i GEÇTİ!
```

### 🔄 Benzer Performans Senaryosu

```
📊 SONUÇLAR:
- Baseline RMSE:      12.34
- Schwarzschild RMSE: 10.89 (+11.8%)
- Kerr RMSE:          10.67 (+13.5%)

📈 Kerr vs Schwarzschild:
- İyileşme: +2.0%
- DM p-değeri: 0.234 ❌ (p > 0.05)

🎯 SONUÇ: İki model benzer performans
```

## 📊 Grafikleri Anlama

### 1. three_model_comparison.png
- **4 çizgi**: Gerçek veri + 3 model tahmini
- **Kesikli dikey çizgiler**: Şok noktaları
- **Kerr en iyiyse**: Gerçeğe en yakın olmalı

### 2. spin_evolution.png
- **Üst panel**: Dönme a(t) - otokorelasyon
  - Pozitif (turuncu): Pozitif momentum
  - Negatif (kırmızı): Negatif momentum
  - Sıfıra yakın: Bağımsız gözlemler
- **Alt panel**: Kütle M(t) - volatilite (referans)

### 3. mass_evolution_kerr.png
- **Turuncu çizgi**: Kütle M(t)
- **Kırmızı kesikli**: Olay ufku (kritik eşik)
- **X işaretleri**: Algılanan şoklar

## 🧪 Hızlı Deneyler

### Deney 1: Saf Schwarzschild
```python
# config_phase2.py
KERR_CONFIG = {
    'regime': 'schwarzschild',  # Kerr'i kapat
}
```
**Beklenti**: Kerr ve Schwarzschild aynı sonucu verir

### Deney 2: Saf Kerr
```python
KERR_CONFIG = {
    'regime': 'kerr',  # Schwarzschild'i kapat
}
```
**Beklenti**: Kerr, otokorelasyonu zorla modeller

### Deney 3: Linear vs Non-linear
```python
# Linear
KERR_CONFIG = {'use_tanh': False}
# vs
# Non-linear
KERR_CONFIG = {'use_tanh': True}
```
**Beklenti**: tanh, aşırı tahminleri önler

### Deney 4: Güçlü Momentum
```python
SHOCK_CONFIG = {
    'decay_rate': 0.01,  # Çok yavaş sönümleme
    'n_shocks': 3,
    'shock_std': 40.0,   # Çok güçlü şoklar
}
```
**Beklenti**: Kerr büyük avantaj gösterir

## 🔍 Sorun Giderme

### Sorun: "detected_regime: schwarzschild" (hep)
**Neden**: Artıklarda otokorelasyon yok
**Çözüm**: Şok parametrelerini artır:
```python
SHOCK_CONFIG = {
    'decay_rate': 0.02,  # Daha yavaş
    'shock_std': 30.0,   # Daha güçlü
}
```

### Sorun: Kerr ve Schwarzschild aynı sonucu veriyor
**Neden**: γ parametresi 0 olarak seçilmiş olabilir
**Kontrol**: `results/phase2_results.txt` içinde γ değerine bakın
**Çözüm**: `gamma_range`'i genişletin

### Sorun: Çok uzun sürüyor
**Neden**: Geniş parametre arama
**Çözüm**: Aralıkları daraltın:
```python
KERR_CONFIG = {
    'alpha_range': [0.5, 1.0, 2.0],  # 5 yerine 3 değer
    'beta_range': [0.05, 0.1],       # 4 yerine 2 değer
    'gamma_range': [0, 0.5, 1.0],    # 4 yerine 3 değer
}
```

## 💡 İpuçları

1. **Önce FAZE 1'i çalıştırın**: Baseline'ı anlayın
2. **Schwarzschild'i referans alın**: Kerr'in katkısını ölçün
3. **Rejim tespitine dikkat**: 'adaptive' en iyi sonucu verir
4. **γ = 0 ise**: Model Schwarzschild'e eşdeğer
5. **Dönme grafığine bakın**: Yüksek |a(t)| → Kerr avantajlı

## 📚 Karşılaştırma Tablosu

| Metrik | Baseline | Schwarzschild | Kerr | Kazanan |
|--------|----------|---------------|------|---------|
| RMSE   | 12.34    | 10.78         | 9.23 | Kerr ✅ |
| MAE    | 9.87     | 8.54          | 7.89 | Kerr ✅ |
| R²     | 0.823    | 0.857         | 0.889| Kerr ✅ |

## 🎯 Başarı Kriterleri (Kerr için)

Kerr'in başarılı olması için:
1. ✅ Kerr RMSE < Schwarzschild RMSE
2. ✅ Kerr vs Schwarzschild DM p < 0.05
3. ✅ γ > 0 (dönme etkisi var)
4. ✅ Ortalama |a(t)| > 0.1 (otokorelasyon var)

## 🔬 İleri Seviye

### Ablasyon Çalışması
4 varyant test edin:
1. Sadece Kütle (Schwarzschild Linear)
2. Kütle + Dönme (Kerr Linear)
3. Sadece Kütle + tanh (Schwarzschild Non-linear)
4. Kütle + Dönme + tanh (Kerr Non-linear) ← En iyi

### Manuel Rejim Seçimi
```python
# Zorla Kerr kullan
kerr_model = KerrGRM(regime='kerr')

# Zorla Schwarzschild kullan
kerr_model = KerrGRM(regime='schwarzschild')

# Otomatik (önerilen)
kerr_model = KerrGRM(regime='adaptive')
```

---

**🚀 İyi şanslar! Kerr metriği ile zaman serilerini bükmek üzeresiniz!**

