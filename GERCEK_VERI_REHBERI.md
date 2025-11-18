# 📊 GERÇEK VERİ İLE ÇALIŞMA REHBERİ - GRM FAZE 3

Bu rehber, FAZE 3'ü **gerçek finansal verilerle** çalıştırmak için tüm seçenekleri açıklar.

---

## 🎯 ÖNEMLİ: 3 Farklı Yöntem Var!

Program **otomatik olarak** şu sırayla dener:
1. ✅ **Manuel CSV** (varsa)
2. 📡 **Otomatik indirme** (SSL bypass ile)
3. 🔄 **Gerçekçi sentetik veri** (fallback)

---

## 🥇 YÖNTEM 1: MANUEL VERİ İNDİRME (ÖNERİLEN - %100 ÇALIŞIR)

### Neden Manuel İndirme?
- ✅ **%100 başarı oranı** (SSL sorunu yok)
- ✅ **Hızlı** (5 dakika)
- ✅ **Güvenilir** (doğrudan Yahoo Finance'dan)

### Adımlar:

#### 1. Yahoo Finance'a Git
```
https://finance.yahoo.com/quote/BTC-USD/history
```
*(BTC-USD yerine istediğiniz ticker'ı kullanabilirsiniz)*

#### 2. Tarih Aralığını Seç
- **Time Period** → **Custom**
- **Start Date**: `2023-11-10`
- **End Date**: `2025-11-09`
- (Veya istediğiniz tarih aralığı - en az 2 yıl önerilir)

#### 3. Veriyi İndir
- **Download** butonuna tıklayın
- CSV dosyası bilgisayarınıza indirilecek (genellikle `BTC-USD.csv`)

#### 4. Dosyayı Proje Klasörüne Taşı
İndirilen dosyayı buraya kopyalayın:
```
C:\Users\asus\Desktop\Ders\4.sınıf\zamanSerisi\Proje\data\BTC-USD.csv
```

**DİKKAT:** Dosya adı `config_phase3.py` dosyasındaki `ticker` değeri ile **tam olarak** aynı olmalı!
```python
# config_phase3.py
REAL_DATA_CONFIG = {
    'ticker': 'BTC-USD',  # ← Bu isimle kaydedin!
    ...
}
```

#### 5. Programı Çalıştır
```bash
python main_phase3.py
```

Program otomatik olarak CSV'yi algılayacak ve kullanacaktır! 🎉

---

## 🥈 YÖNTEM 2: OTOMATİK İNDİRME (SSL BYPASS)

### Açıklama
Program **güçlü SSL bypass** mekanizmalarıyla donatılmıştır:
- SSL sertifika doğrulaması devre dışı
- Retry mekanizması (10 deneme)
- Alternatif indirme yöntemleri
- User-Agent maskeleme

### Nasıl Çalışır?
Eğer manuel CSV yoksa, program **otomatik olarak** indirir.

### SSL Hatası Alırsanız?
Program otomatik olarak YÖNTEM 3'e geçer (gerçekçi sentetik veri).

### SSL'i Tamamen Düzeltmek İsterseniz:
```bash
# 1. Paketleri güncelleyin
pip install --upgrade certifi urllib3 requests yfinance

# 2. Certifi konumunu kontrol edin
python -m certifi

# 3. Sisteminizin sertifikasını güncelleyin
# Windows: Windows Update çalıştırın
# Python: pip install --upgrade certifi
```

---

## 🥉 YÖNTEM 3: GERÇEKÇİ SENTETİK VERİ (OTOM ATİK FALLBACK)

### Ne Zaman Kullanılır?
- Manuel CSV yok
- Otomatik indirme başarısız (SSL hatası)
- Program **otomatik olarak** bu yönteme geçer

### Özellikler:
- ✅ Gerçekçi fiyat hareketleri
- ✅ Volatilite kümelenmesi (GARCH-like)
- ✅ Bull/Bear fazları
- ✅ Şok olayları (kripto crash/pump benzeri)
- ✅ Trend bileşenleri

### Avantajları:
- **Hızlı** (anında oluşur)
- **Test için yeterli** (modelleri karşılaştırabilirsiniz)
- **Tekrarlanabilir** (seed kontrolü)

### Dezavantajları:
- ❌ Gerçek piyasa verileri değil
- ❌ Akademik çalışmalar için uygun değil

---

## 🔧 FARKLI VARLIKLAR KULLANMA

### config_phase3.py'yi Düzenleyin:

```python
REAL_DATA_CONFIG = {
    'ticker': 'AAPL',  # ← Değiştirin!
    'start_date': '2023-11-10',
    'end_date': '2025-11-09',
}
```

### Popüler Ticker'lar:

#### Kripto Paralar:
- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum
- `DOGE-USD` - Dogecoin

#### Hisse Senetleri:
- `AAPL` - Apple
- `GOOGL` - Google/Alphabet
- `MSFT` - Microsoft
- `TSLA` - Tesla
- `AMZN` - Amazon

#### Endeksler:
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones
- `^IXIC` - NASDAQ

#### Forex:
- `EURUSD=X` - EUR/USD
- `GBPUSD=X` - GBP/USD

---

## 📋 BEKLENEN CSV FORMATI

Yahoo Finance'dan indirilen CSV şu formatta olmalı:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2023-11-10,35000.00,36000.00,34500.00,35800.00,35800.00,25000000000
2023-11-11,35800.00,37000.00,35500.00,36500.00,36500.00,28000000000
...
```

**NOT:** Program sadece şu sütunları kullanır:
- `Date` - Tarih
- `Close` - Kapanış fiyatı

Diğer sütunlar (Open, High, Low, Volume) opsiyoneldir.

---

## 🐛 SORUN GİDERME

### Problem: "SSL certificate verify locations" hatası
**Çözüm 1:** Manuel CSV kullanın (YÖNTEM 1)
**Çözüm 2:** Gerçekçi sentetik veri ile devam edin (otomatik)
**Çözüm 3:** SSL paketlerini güncelleyin (yukarıdaki komutlar)

### Problem: CSV dosyası bulunamıyor
**Kontrol edin:**
1. Dosya adı doğru mu? (`BTC-USD.csv` gibi)
2. Dosya `data/` klasöründe mi?
3. Dosya yolu doğru mu?

```python
# Dosya yolunu görmek için:
import os
print(os.path.abspath('data/BTC-USD.csv'))
```

### Problem: "Date" sütunu bulunamıyor
**Çözüm:** CSV'deki tarih sütununun adı "Date" olmalı.
Excel'de açıp sütun adını değiştirin.

### Problem: Veri çok kısa (< 100 gözlem)
**Çözüm:** Daha uzun tarih aralığı seçin (en az 2 yıl önerilir)

### Problem: "Empty DataFrame" hatası
**Çözüm:** 
- Ticker doğru mu? (örn: `BTC-USD`, `AAPL`)
- Tarih aralığı uygun mu?
- Weekend/tatil günleri hariç yeterli gün var mı?

---

## ✅ DOĞRU ÇALIŞTIĞINI NASIL ANLARIM?

### Manuel CSV Başarılı:
```
✅ MANUEL CSV BULUNDU: data/BTC-USD.csv
✅ CSV'DEN YÜKLEME BAŞARILI!
   - Varlık: BTC-USD
   - Gözlem: 730
   - Tarih: 2023-11-10 - 2025-11-09
```

### Otomatik İndirme Başarılı:
```
📡 OTOMATİK İNDİRME BAŞLATILIYOR...
   ✓ 730 gözlem indirildi
✅ VERİ YÜKLEME BAŞARILI!
   📊 İstatistikler:
      - Gözlem sayısı: 730
      - Fiyat aralığı: $28,500.00 - $42,300.00
```

### Sentetik Veri Kullanıldı:
```
❌ OTOMATİK İNDİRME BAŞARISIZ!
🔄 GERÇEKÇİ SENTETİK VERİ OLUŞTURULUYOR...
   ✓ 730 gözlem oluşturuldu
   Fiyat aralığı: $28,234.56 - $41,234.87
```

---

## 📊 SONUÇ DOSYALARI

Program çalıştıktan sonra şu dosyalar oluşur:

### Veri Dosyaları:
- `data/BTC-USD.csv` - Manuel indirilen (sizin eklediğiniz)
- `data/real_data_phase3.csv` - Otomatik indirilen
- `data/realistic_synthetic_data.csv` - Sentetik veri (fallback)

### Sonuç Dosyaları:
- `results/phase3_results.txt` - Performans metrikleri
- `results/phase3_comparison.png` - Model karşılaştırması
- `results/phase3_residuals.png` - Rezidüel analizi
- `results/phase3_performance.png` - Performans grafikleri

### Metadata:
Her sonuç dosyasında hangi veri tipinin kullanıldığı belirtilir:
- `data_type: 'manual_csv'` - Manuel CSV
- `data_type: 'real_yahoo_finance'` - Otomatik indirme
- `data_type: 'realistic_synthetic'` - Sentetik veri

---

## 🚀 HIZLI BAŞLANGIÇ

### En Kolay Yol (5 Dakika):

1. **Tarayıcınızda açın:**
   ```
   https://finance.yahoo.com/quote/BTC-USD/history
   ```

2. **Download → CSV indir**

3. **Dosyayı taşı:**
   ```
   İndirilen dosya → data/BTC-USD.csv
   ```

4. **Çalıştır:**
   ```bash
   python main_phase3.py
   ```

5. **Bitir! 🎉**
   ```
   results/ klasöründe tüm sonuçlar hazır!
   ```

---

## 📞 EK KAYNAKLAR

- **SSL Sorunları:** `SSL_COZUM.md`
- **Manuel İndirme Detay:** `data/MANUAL_DOWNLOAD_GUIDE.md` (otomatik oluşur)
- **Genel FAZE 3 Bilgisi:** `README_PHASE3.md`
- **Hızlı Başlangıç:** `QUICK_START_PHASE3.md`

---

## 🎓 AKADEMİK KULLANIM

Eğer sonuçları **akademik çalışmada** kullanacaksanız:

✅ **Mutlaka gerçek veri kullanın** (YÖNTEM 1 veya 2)

❌ **Sentetik veri kullanmayın** (sadece test amaçlı)

Veri kaynağını belirtin:
```
Veri Kaynağı: Yahoo Finance (https://finance.yahoo.com/)
Ticker: BTC-USD
Tarih Aralığı: 2023-11-10 - 2025-11-09
Gözlem Sayısı: 730
İndirme Tarihi: 2025-11-09
```

---

**Son Güncelleme:** 2025-11-09  
**Versiyon:** 1.0  
**Durum:** Gerçek veri garantili! 🎯

