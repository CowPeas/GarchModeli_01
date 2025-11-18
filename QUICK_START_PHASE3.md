# 🚀 GRM FAZE 3 - Hızlı Başlangıç

## ⚡ 3 Adımda Gerçek Veri Testi

### 1️⃣ Yeni Kütüphaneleri Yükleyin

```bash
pip install yfinance arch
```

### 2️⃣ Simülasyonu Çalıştırın

```bash
python main_phase3.py
```

### 3️⃣ Sonuçları İnceleyin

```bash
cat results/phase3_results.txt
```

## 🎯 İlk Çalıştırma

İlk kez çalıştırıyorsanız:
- ✅ İnternet bağlantınızı kontrol edin
- ✅ Varsayılan varlık: Bitcoin (BTC-USD)
- ✅ Varsayılan periyot: 2 yıl
- ⏱️ İndirme + analiz: ~3-5 dakika

## 📊 Farklı Varlıklar Test Edin

### Bitcoin
```python
# config_phase3.py
REAL_DATA_CONFIG = {
    'asset': 'BTC-USD',
    'period': '2y',
}
```

### S&P 500
```python
REAL_DATA_CONFIG = {
    'asset': '^GSPC',
    'period': '5y',
}
```

### Ethereum
```python
REAL_DATA_CONFIG = {
    'asset': 'ETH-USD',
    'period': '1y',
}
```

## 🔍 Ne Beklemeli?

### ✅ Başarılı Çalıştırma

```
📥 BTC-USD verisi indiriliyor...
✓ 730 gözlem indirildi

📊 Getiri İstatistikleri (log):
  - Ortalama: 0.001234
  - Std Sapma: 0.045678

🔧 GARCH(1,1) Modeli Eğitiliyor...
✓ Model eğitimi tamamlandı

🔍 Kerr GRM Parametre Optimizasyonu (Rejim: kerr):
✓ En iyi parametreler: α=2.00, β=0.050, γ=1.00

Model                    RMSE        MAE
Baseline             0.045678   0.034567
GARCH                0.043210   0.032101
Schwarzschild        0.041234   0.030456
Kerr                 0.038901   0.028901

✅ FAZE 3 SİMÜLASYONU TAMAMLANDI!
```

### ⚠️ Olası Sorunlar

#### Sorun 1: "No module named 'yfinance'"
```bash
pip install yfinance
```

#### Sorun 2: "No module named 'arch'"
```bash
pip install arch
```

#### Sorun 3: "No data found for ticker"
**Çözüm**: Ticker sembolünü kontrol edin
- Bitcoin: `BTC-USD` (BTC değil!)
- S&P 500: `^GSPC` (GSPC değil!)

#### Sorun 4: GARCH convergence hatası
**Endişelenmeyin!** Basit volatilite modeli kullanılır:
```
⚠️ GARCH modeli başarısız: ...
   Basit volatilite modeli kullanılıyor...
```

## 📈 Sonuçları Anlama

### RMSE Değerleri
- **< 0.03**: Mükemmel (düşük volatilite)
- **0.03-0.05**: İyi (orta volatilite)
- **> 0.05**: Zorlayıcı (yüksek volatilite)

### İyileşme Yüzdeleri
```
GARCH            +5.40%   → Küçük iyileşme
Schwarzschild    +9.72%   → Orta iyileşme
Kerr            +14.84%   → İyi iyileşme ✨
```

### p-değerleri
- **p < 0.05**: İstatistiksel olarak anlamlı ✅
- **p > 0.05**: Anlamlı değil ❌

## 🧪 Hızlı Deneyler

### Deney 1: Farklı Periyotlar
```python
# Kısa dönem (volatil)
'period': '6mo'

# Orta dönem (dengeli)
'period': '1y'

# Uzun dönem (trend)
'period': '5y'
```

### Deney 2: GARCH Tipleri
```python
GARCH_CONFIG = {
    'model_types': ['GARCH'],     # Standart
    # veya
    'model_types': ['EGARCH'],    # Asimetrik
    # veya
    'model_types': ['GJR-GARCH'], # Kaldıraç etkisi
}
```

### Deney 3: GRM Rejim Seçimi
```python
KERR_CONFIG = {
    'regime': 'adaptive',      # Otomatik (önerilen)
    # veya
    'regime': 'schwarzschild', # Sadece kütle
    # veya
    'regime': 'kerr',          # Zorla dönme
}
```

## 💡 İpuçları

### Veri İndirme
- **İlk indirme**: Yavaş olabilir (~30 saniye)
- **Tekrar çalıştırma**: Yahoo Finance cache kullanır
- **Farklı varlık**: Yeniden indirir

### Model Eğitimi
- **ARIMA Grid Search**: En uzun adım (~1-2 dk)
- **GARCH**: Bazen convergence sorunu
- **GRM**: Hızlı (~10 saniye)

### Performans Beklentileri
- **Kripto (BTC, ETH)**: GRM genelde avantajlı
- **Endeks (S&P 500)**: GARCH vs GRM rekabetçi
- **Kriz dönemleri**: Kerr GRM öne çıkar

## 🎯 Başarı Kriterleri

Kerr GRM'nin başarılı sayılması için:
1. ✅ Kerr RMSE < GARCH RMSE
2. ✅ Kerr vs GARCH DM p < 0.10
3. ✅ Kerr RMSE < Baseline RMSE (en az %5)

## 📊 Karşılaştırma Tablosu Örneği

| Metrik | Baseline | GARCH | Schwarzschild | Kerr | Kazanan |
|--------|----------|-------|---------------|------|---------|
| RMSE   | 0.0457   | 0.0432| 0.0412        | 0.0389| Kerr ✅ |
| MAE    | 0.0346   | 0.0321| 0.0305        | 0.0289| Kerr ✅ |
| R²     | 0.1234   | 0.1856| 0.2245        | 0.2789| Kerr ✅ |

## 🆚 FAZE 2 vs FAZE 3

### FAZE 2 (Sentetik)
- 4 şok, bilinen pozisyonlar
- %20 iyileşme (kontrollü)
- Tüm testler başarılı

### FAZE 3 (Gerçek)
- Bilinmeyen şoklar, gerçek volatilite
- %10-15 iyileşme (gerçekçi)
- Bazı dönemlerde başarısız

## 🔬 Ek Analiz (Manuel)

### Farklı Tarih Aralığı
```python
from models import RealDataLoader

loader = RealDataLoader()
df = loader.load_yahoo_finance(
    ticker='BTC-USD',
    start_date='2020-01-01',
    end_date='2020-12-31'  # COVID dönemi
)
```

### Volatilite Kümeleri
```python
df, metadata = load_popular_assets('BTC-USD', '2y')

# Yüksek volatilite dönemleri
high_vol = df[df['high_volatility'] == True]
print(f"Yüksek volatilite günleri: {len(high_vol)}")
```

## ⚡ Hızlı Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| ModuleNotFoundError | `pip install yfinance arch` |
| No data found | Ticker sembolünü kontrol et |
| GARCH error | Normal, basit model kullanılır |
| Çok yavaş | Period'u kısalt ('6mo') |
| Kötü sonuçlar | Farklı varlık/period dene |

---

**🚀 Hadi başla! Gerçek piyasalarda GRM'yi test et!**

```bash
python main_phase3.py
```

