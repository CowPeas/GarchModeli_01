# GRM (Gravitational Residual Model) - FAZE 3

## 📖 Proje Hakkında

**FAZE 3**, GRM projesinin **gerçek dünya testi** fazıdır. Sentetik veriden (FAZE 1-2) gerçek finansal verilere geçiş yaparak modellerin pratik değerini test eder.

### 🆕 FAZE 3 Yenilikleri

1. **Gerçek Finansal Veri**: Bitcoin, S&P 500, vb. (Yahoo Finance)
2. **GARCH Benchmark**: Endüstri standardı volatilite modeli
3. **Dört Model Karşılaştırması**: Baseline + GARCH + Schwarzschild + Kerr
4. **Kapsamlı İstatistiksel Testler**: Diebold-Mariano, Ljung-Box, ARCH-LM
5. **Risk Metrikleri**: VaR, CVaR, Sharpe Ratio (geliştirilecek)

## 🌍 Gerçek Veri vs Sentetik Veri

### FAZE 1-2 (Sentetik)
- ✅ Kontrollü test ortamı
- ✅ Bilinen şoklar ve parametreler
- ✅ Hipotez doğrulaması
- ❌ Gerçek dünya karmaşıklığı yok

### FAZE 3 (Gerçek)
- ✅ Gerçek piyasa dinamikleri
- ✅ Volatilite kümelenmesi
- ✅ Asimetrik şoklar
- ✅ Pratik uygulanabilirlik testi
- ⚠️ Gürültülü, beklenmedik olaylar

## 🏗️ Proje Yapısı (FAZE 3)

```
.
├── models/
│   ├── real_data_loader.py    # 🆕 Gerçek veri yükleme
│   ├── garch_model.py          # 🆕 GARCH implementasyonu
│   ├── kerr_grm_model.py       # Kerr GRM (FAZE 2)
│   ├── grm_model.py            # Schwarzschild (FAZE 1)
│   └── ...
├── config_phase3.py            # 🆕 FAZE 3 konfigürasyonu
├── main_phase3.py              # 🆕 FAZE 3 ana script
├── README_PHASE3.md            # Bu dosya
└── requirements.txt            # Güncellenmiş (yfinance + arch)
```

## 🚀 Kurulum ve Çalıştırma

### 1. Yeni Gereksinimleri Yükleyin

```bash
pip install yfinance arch
```

Veya tüm gereksinimleri:

```bash
pip install -r requirements.txt
```

### 2. FAZE 3 Simülasyonunu Çalıştırın

```bash
python main_phase3.py
```

## 📊 Desteklenen Varlıklar

### Kripto Para
- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum

### Hisse Senedi Endeksleri
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones Industrial Average

### Forex
- `EURUSD=X` - EUR/USD

### Emtia
- `GC=F` - Altın Vadeli İşlemleri

## ⚙️ Özelleştirme

### config_phase3.py

#### Farklı Varlık Seç
```python
REAL_DATA_CONFIG = {
    'asset': '^GSPC',       # Bitcoin yerine S&P 500
    'period': '5y',         # 5 yıllık veri
    'use_returns': True,
}
```

#### GARCH Model Tipi
```python
GARCH_CONFIG = {
    'model_types': ['EGARCH'],  # Asimetrik GARCH
    # veya
    'model_types': ['GJR-GARCH'],  # GJR-GARCH
}
```

## 🎯 Beklenen Sonuçlar

### Performans Sıralaması (Genel Beklenti)

**Yüksek Volatilite Dönemlerinde**:
1. 🥇 Kerr GRM (momentum + volatilite)
2. 🥈 Schwarzschild GRM (volatilite)
3. 🥉 GARCH (volatilite)
4. Baseline ARIMA

**Düşük Volatilite Dönemlerinde**:
1. 🥇 Baseline ARIMA
2. 🥈 GARCH
3. 🥉 Schwarzschild/Kerr (benzer)

## 📈 Örnek Çıktı

```
================================================================================
GRM (GRAVITATIONAL RESIDUAL MODEL) - FAZE 3 SİMÜLASYONU
================================================================================

📥 ADIM 1: Gerçek Finansal Veri Yükleme
--------------------------------------------------------------------------------
📥 BTC-USD verisi indiriliyor...
   Tarih aralığı: 2023-11-09 - 2025-11-09
✓ 730 gözlem indirildi
  - İlk tarih: 2023-11-09
  - Son tarih: 2025-11-09

📊 Getiri İstatistikleri (log):
  - Ortalama: 0.001234
  - Std Sapma: 0.045678
  - Çarpıklık: -0.234
  - Basıklık: 5.678

🔥 Volatilite Analizi:
  - Ortalama volatilite: 0.042
  - Max volatilite: 0.123
  - Yüksek volatilite dönemleri: 182 gözlem (25.0%)

📈 ADIM 7: Dört Model Kapsamlı Karşılaştırma
================================================================================

Model                    RMSE        MAE       MAPE         R²
--------------------------------------------------------------------------------
Baseline             0.045678   0.034567     78.90     0.1234
GARCH                0.043210   0.032101     75.23     0.1856
Schwarzschild        0.041234   0.030456     72.45     0.2245
Kerr                 0.038901   0.028901     68.12     0.2789

================================================================================
İYİLEŞME YÜZDE LERİ (Baseline'a göre)
================================================================================
GARCH                   +5.40%
Schwarzschild          +9.72%
Kerr                  +14.84%

================================================================================
DIEBOLD-MARIANO TEST SONUÇLARI
================================================================================
GARCH vs Baseline              p = 0.1234
Schwarzschild vs Baseline      p = 0.0456
Kerr vs Baseline               p = 0.0089
Kerr vs GARCH                  p = 0.0234
Kerr vs Schwarzschild          p = 0.0678
================================================================================

✅ FAZE 3 SİMÜLASYONU TAMAMLANDI!
```

## 🔬 Hipotez (FAZE 3)

**H₁ (Gerçek Veri)**: GRM modelleri (Schwarzschild ve Kerr), gerçek finansal verilerde standart volatilite modelleri (GARCH) ile karşılaştırılabilir veya daha iyi tahmin performansı gösterir, özellikle yüksek volatilite ve rejim değişikliği dönemlerinde.

**Başarı Kriterleri**:
- ✅ En az bir GRM modeli GARCH'tan istatistiksel olarak anlamlı şekilde iyi (p < 0.05)
- ✅ Kerr > Schwarzschild (otokorelasyon varsa)
- ✅ Yüksek volatilite dönemlerinde GRM avantajı belirgin

## 📊 Model Karşılaştırması

| Özellik | Baseline | GARCH | Schwarzschild | Kerr |
|---------|----------|-------|---------------|------|
| Volatilite Modeli | ❌ | ✅ | ✅ | ✅ |
| Otokorelasyon | ❌ | ❌ | ❌ | ✅ |
| Non-linear | ❌ | ✅ | ❌ | ✅ |
| Asimetri | ❌ | ✅* | ❌ | ✅** |
| Parametre Sayısı | 2-3 | 3-5 | 2 | 3 |
| Hesaplama Hızı | Hızlı | Orta | Hızlı | Orta |

\* EGARCH/GJR-GARCH kullanılırsa  
\** Dönme parametresi ile

## 🧪 Test Senaryoları

### Senaryo 1: Bitcoin (Yüksek Volatilite)
```python
REAL_DATA_CONFIG = {
    'asset': 'BTC-USD',
    'period': '2y',
}
```
**Beklenti**: Kerr > Schwarzschild > GARCH > Baseline

### Senaryo 2: S&P 500 (Orta Volatilite)
```python
REAL_DATA_CONFIG = {
    'asset': '^GSPC',
    'period': '5y',
}
```
**Beklenti**: GARCH ≈ Kerr > Schwarzschild > Baseline

### Senaryo 3: Kriz Dönemi (2020 COVID)
```python
# Manuel tarih aralığı için RealDataLoader kullan
loader = RealDataLoader()
df = loader.load_yahoo_finance(
    '^GSPC',
    '2019-01-01',
    '2021-12-31'
)
```
**Beklenti**: Kerr > diğer modeller (rejim değişikliği)

## 📝 Çıktılar

### Veri
- `data/real_data_phase3.csv` - İndirilen ve işlenmiş gerçek veri

### Sonuçlar
- `results/phase3_results.txt` - Detaylı karşılaştırma raporu

### Grafikler (FAZE 3 için genişletilebilir)
- Zaman serisi karşılaştırması
- Volatilite evrimi
- Kümülatif getiriler

## ❗ Önemli Notlar

### İnternet Bağlantısı
FAZE 3, Yahoo Finance'den veri indirdiği için **internet bağlantısı** gerektirir.

### Veri Kalitesi
- Bazı varlıklarda veri eksiklikleri olabilir
- Hafta sonları ve tatillerde veri olmaz
- Çok kısa periyotlarda (<1 ay) sonuçlar güvenilir olmayabilir

### GARCH Convergence
GARCH modeli bazen yakınsamama sorunu yaşar. Bu durumda:
- Basit volatilite modeli kullanılır
- Uyarı mesajı görüntülenir

## 🔜 Gelecek Geliştirmeler

- [ ] Risk metrikleri (VaR, CVaR, Sharpe)
- [ ] Yuvarlanan pencere analizi
- [ ] Çoklu varlık portföy testi
- [ ] LSTM/Transformer karşılaştırması
- [ ] Gerçek zamanlı tahmin API'si
- [ ] Dashboard ve görselleştirme

## 🎓 Bilimsel Değer

FAZE 3'ün katkıları:

1. **Pratik Uygulanabilirlik**: Gerçek piyasalarda çalışıyor mu?
2. **GARCH Karşılaştırması**: Endüstri standardı ile kıyaslama
3. **Robustluk Testi**: Farklı varlık sınıflarında performans
4. **Rejim Değişikliği**: Kriz dönemlerinde davranış

---

**🎉 FAZE 3 ile GRM'yi gerçek dünyada test edin!**

Sorularınız için `README.md` (FAZE 1), `README_PHASE2.md` (FAZE 2) ve bu dosyaya başvurabilirsiniz.

