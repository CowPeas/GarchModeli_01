# 📥 MANUEL VERİ İNDİRME REHBERİ

Bu rehber, SSL sertifika sorunları nedeniyle otomatik veri indirme yapılamadığında
manuel olarak veri indirme işlemini açıklar.

## 🎯 Adım 1: Yahoo Finance'a Erişim

1. Tarayıcınızı açın (Chrome, Firefox, Edge, vb.)
2. Şu URL'yi ziyaret edin:
   ```
   https://finance.yahoo.com/
   ```

## 🔍 Adım 2: Varlık Arama

1. Arama kutusuna varlık sembolünü yazın:
   - Bitcoin: `BTC-USD`
   - Ethereum: `ETH-USD`
   - Apple: `AAPL`
   - S&P 500: `^GSPC`

2. Varlığı seçin ve sayfasını açın

## 📊 Adım 3: Tarihsel Veri Sekmesi

1. Sayfada "Historical Data" sekmesine tıklayın
2. "Time Period" seçeneğini ayarlayın:
   - 2 yıllık veri için: Son 2 yıl (Last 2 Years)
   - Özel aralık: Custom date range

## 💾 Adım 4: Veri İndirme

1. "Download" butonuna tıklayın
2. CSV dosyası bilgisayarınıza indirilecek
3. Dosya adı genellikle: `BTC-USD.csv` formatında olur

## 📁 Adım 5: Dosyayı Proje Klasörüne Taşıma

1. İndirilen CSV dosyasını bulun (genellikle Downloads klasöründe)
2. Proje klasörünüzdeki `data/` dizinine kopyalayın:
   ```
   C:\Users\asus\Desktop\Ders\4.sınıf\zamanSerisi\Proje\data\
   ```

## 🔧 Adım 6: Kodu Güncelleme

`main_phase3.py` dosyasını açın ve veri yükleme bölümünü güncelleyin:

```python
# Otomatik yükleme yerine (hatalı):
# df, metadata = loader.load_yahoo_finance(...)

# Manuel CSV yükleme kullanın:
from models.alternative_data_loader import AlternativeDataLoader

alt_loader = AlternativeDataLoader()
df = alt_loader.load_from_csv(
    filepath='data/BTC-USD.csv',
    date_column='Date',
    price_column='Close'
)

metadata = {
    'asset': 'BTC-USD',
    'period': '2023-2025',
    'n_samples': len(df),
    'data_type': 'manual_csv'
}
```

## ✅ Adım 7: Programı Çalıştırma

```bash
python main_phase3.py
```

## 📋 Beklenen CSV Formatı

İndirilen CSV dosyası şu formatta olmalıdır:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2023-11-10,35000.00,36000.00,34500.00,35800.00,35800.00,25000000000
2023-11-11,35800.00,37000.00,35500.00,36500.00,36500.00,28000000000
...
```

**NOT:** Program sadece `Date` ve `Close` sütunlarını kullanır!

## 🔄 Alternatif Veri Kaynakları

Eğer Yahoo Finance'dan da veri alamazsanız:

### 1. CoinGecko (Kripto paralar için)
- Web: https://www.coingecko.com/
- API: Ücretsiz, kayıt gerekli
- CSV export: Mevcut

### 2. Alpha Vantage (Hisse senetleri için)
- Web: https://www.alphavantage.co/
- API Key: Ücretsiz (kayıt gerekli)
- Günlük limit: 500 istek

### 3. FRED (Ekonomik veriler için)
- Web: https://fred.stlouisfed.org/
- API: Ücretsiz
- Python paketi: `pandas-datareader`

## ❓ Sorun Giderme

### Problem: CSV dosyası açılmıyor
**Çözüm:** Excel yerine Not Defteri ile açın, encoding sorunları olabilir

### Problem: Date sütunu tanınmıyor
**Çözüm:** CSV'deki tarih formatını kontrol edin (YYYY-MM-DD olmalı)

### Problem: Hala hata alıyorum
**Çözüm:** Sentetik veri ile test yapın (FAZE 2):
```bash
python main_phase2.py
```

## 📞 Destek

Bu rehber yeterli değilse:
1. `SSL_COZUM.md` dosyasına bakın
2. `README_PHASE3.md` dosyasını inceleyin
3. GitHub Issues'da sorun bildirin

---
**Son Güncelleme:** 2025-11-09
**Versiyon:** 1.0
