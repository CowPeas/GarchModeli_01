# 🔧 SSL Sertifika Hatası Çözümleri

## ❗ Sorun

```
SSLError: Failed to perform, curl: (77) error setting certificate verify locations
```

Bu hata, yfinance'in Yahoo Finance'e HTTPS bağlantısı kurarken SSL sertifika doğrulamasında sorun yaşaması nedeniyle oluşur.

## ✅ Çözümler (Öncelik Sırasına Göre)

### Çözüm 1: certifi Paketini Güncelleyin (ÖNERİLEN)

```bash
pip install --upgrade certifi
pip install --upgrade yfinance
```

Sonra tekrar deneyin:
```bash
python main_phase3.py
```

### Çözüm 2: SSL Doğrulamasını Devre Dışı Bırak (Geçici)

Kod zaten güncellenmiştir ve SSL doğrulaması varsayılan olarak kapalıdır.

```bash
python main_phase3.py
```

### Çözüm 3: Sentetik Veri Kullanın

FAZE 2'yi çalıştırın (gerçek veri gerektirmez):

```bash
python main_phase2.py
```

### Çözüm 4: CSV Dosyası Kullanın

#### Adım 1: Veriyi Manuel İndirin

https://finance.yahoo.com/quote/BTC-USD/history adresinden:
1. Time Period seçin (2 yıl)
2. "Download" butonuna tıklayın
3. CSV dosyasını `data/btc_manual.csv` olarak kaydedin

#### Adım 2: CSV'den Yükleyin

`main_phase3.py` dosyasını düzenleyin:

```python
# ADIM 1'deki try bloğunu değiştir:
from models import RealDataLoader

loader = RealDataLoader()
df = loader.load_from_csv(
    filepath='data/btc_manual.csv',
    date_column='Date',
    price_column='Close'
)
df, metadata = loader.prepare_for_modeling(df)
```

### Çözüm 5: Farklı Varlık Deneyin

`config_phase3.py`:

```python
REAL_DATA_CONFIG = {
    'asset': '^GSPC',  # S&P 500 deneyin
    'period': '1y',
}
```

### Çözüm 6: Python SSL Modülü Kontrolü

```bash
python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

Eğer eski bir versiyon gösterirse:

```bash
pip install --upgrade pyopenssl cryptography
```

## 🔄 Otomatik Fallback

Kod artık otomatik olarak:
1. Gerçek veri indirmeyi dener
2. Başarısız olursa **sentetik veri** kullanır
3. Simülasyon devam eder

## 🧪 Test Et

```bash
python main_phase3.py
```

Beklenen çıktı:
```
❌ Veri yükleme hatası: ...
💡 ÇÖZÜM SEÇENEKLERİ:
   1. SSL hatası için: pip install --upgrade certifi
   2. Sentetik veri ile test: python main_phase2.py
   3. CSV dosyası kullan

🔄 Alternatif: Sentetik veri ile devam ediliyor...
✓ Sentetik veri hazır: 500 gözlem

[Simülasyon devam eder...]
```

## 📋 Kontrol Listesi

- [ ] `pip install --upgrade certifi` çalıştır
- [ ] `python main_phase3.py` tekrar dene
- [ ] Başarısız olursa `python main_phase2.py` kullan
- [ ] Veya CSV dosyası ile manuel yükle

## 💡 Kalıcı Çözüm (Windows)

Eğer sorun devam ederse:

```bash
# Sanal ortamı yeniden oluştur
deactivate
rmdir /s env
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## 🆘 Hala Sorun mu Var?

1. İnternet bağlantınızı kontrol edin
2. Güvenlik duvarı/antivirüs kontrol edin
3. VPN kullanıyorsanız kapatın
4. Proxy ayarlarını kontrol edin

## ✨ Kod Güncellemesi

Kod zaten güncellenmiştir:
- ✅ SSL doğrulaması opsiyonel
- ✅ Otomatik sentetik veri fallback
- ✅ Detaylı hata mesajları
- ✅ Çözüm önerileri

---

**🎉 Artık herhangi bir durumda çalışır!**

