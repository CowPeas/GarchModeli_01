"""
Alternative Data Loader Module - SSL sorunları için yedek veri kaynakları.

Bu modül, Yahoo Finance'dan veri çekilemediğinde alternatif yöntemler sunar.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import warnings


class AlternativeDataLoader:
    """Alternatif veri yükleme yöntemleri."""
    
    def __init__(self):
        """Initialize alternative data loader."""
        self.data_sources = ['csv', 'manual', 'synthetic']
    
    def load_from_csv(
        self, 
        filepath: str,
        date_column: str = 'Date',
        price_column: str = 'Close'
    ) -> pd.DataFrame:
        """
        CSV dosyasından veri yükle.
        
        Parameters
        ----------
        filepath : str
            CSV dosya yolu
        date_column : str
            Tarih sütunu adı
        price_column : str
            Fiyat sütunu adı
            
        Returns
        -------
        pd.DataFrame
            Zaman serisi verisi
        """
        print(f"📂 CSV'den veri yükleniyor: {filepath}")
        
        try:
            # CSV oku
            df = pd.read_csv(filepath)
            
            # Tarih sütununu datetime'a çevir
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Sadece gerekli sütunları al
            df = df[[date_column, price_column]].copy()
            df.columns = ['date', 'price']
            
            # Sırala
            df = df.sort_values('date').reset_index(drop=True)
            
            # Getiri hesapla
            df['returns'] = df['price'].pct_change()
            df = df.dropna()
            
            print(f"   ✓ {len(df)} gözlem yüklendi")
            print(f"   Tarih aralığı: {df['date'].min()} - {df['date'].max()}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"CSV yükleme hatası: {str(e)}")
    
    def create_sample_csv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        output_path: str
    ) -> str:
        """
        Manuel indirme için örnek CSV şablonu oluştur.
        
        Parameters
        ----------
        ticker : str
            Varlık sembolü
        start_date : str
            Başlangıç tarihi
        end_date : str
            Bitiş tarihi
        output_path : str
            Çıktı dosya yolu
            
        Returns
        -------
        str
            Manuel indirme talimatları
        """
        instructions = f"""
================================================================================
MANUEL VERİ İNDİRME TALİMATLARI - {ticker}
================================================================================

1. ADIM: Yahoo Finance'a Git
   URL: https://finance.yahoo.com/quote/{ticker}/history

2. ADIM: Tarih Aralığını Ayarla
   Başlangıç: {start_date}
   Bitiş: {end_date}

3. ADIM: Veriyi İndir
   - "Time Period" seçeneğini ayarla
   - "Download" butonuna tıkla
   - CSV dosyası indirilecek

4. ADIM: Dosyayı Kaydet
   İndirilen dosyayı buraya kaydet:
   {output_path}

5. ADIM: Kodu Çalıştır
   main_phase3.py içinde load_csv() fonksiyonunu kullan

================================================================================
ÖRNEGİN BEKLENİLEN CSV FORMATI:
================================================================================
Date,Open,High,Low,Close,Volume
2023-11-10,100.00,105.00,99.00,104.50,1000000
2023-11-11,104.50,107.00,103.00,106.20,1200000
...

NOT: Sadece 'Date' ve 'Close' sütunları yeterli!
================================================================================
"""
        
        # Talimatları dosyaya kaydet
        with open(output_path.replace('.csv', '_instructions.txt'), 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        print(instructions)
        return instructions
    
    def generate_realistic_crypto_data(
        self,
        days: int = 730,
        initial_price: float = 30000.0,
        volatility: float = 0.03
    ) -> pd.DataFrame:
        """
        Gerçekçi kripto para verisi oluştur (son çare).
        
        Parameters
        ----------
        days : int
            Gün sayısı
        initial_price : float
            Başlangıç fiyatı
        volatility : float
            Volatilite seviyesi
            
        Returns
        -------
        pd.DataFrame
            Sentetik ama gerçekçi veri
        """
        print(f"🔄 Gerçekçi sentetik veri oluşturuluyor ({days} gün)...")
        
        # Tarih aralığı oluştur
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Fiyat serisi oluştur (geometrik Brownian motion + trendler)
        np.random.seed(42)
        
        # Trend bileşeni (bull/bear fazları)
        trend = np.zeros(len(dates))
        phase_length = len(dates) // 4
        
        for i in range(4):
            start_idx = i * phase_length
            end_idx = (i + 1) * phase_length if i < 3 else len(dates)
            
            if i % 2 == 0:  # Bull phase
                trend[start_idx:end_idx] = np.linspace(0, 0.5, end_idx - start_idx)
            else:  # Bear phase
                trend[start_idx:end_idx] = np.linspace(0.5, -0.3, end_idx - start_idx)
        
        # Rastgele yürüyüş + volatilite kümelenmesi
        returns = np.random.normal(0.001, volatility, len(dates))
        
        # Volatilite kümelenmesi (GARCH-like)
        volatility_process = np.ones(len(dates)) * volatility
        for i in range(1, len(dates)):
            if abs(returns[i-1]) > 2 * volatility:
                volatility_process[i] = min(volatility_process[i-1] * 1.5, 0.1)
            else:
                volatility_process[i] = max(volatility_process[i-1] * 0.95, volatility)
            
            returns[i] = np.random.normal(trend[i]/100, volatility_process[i])
        
        # Fiyat serisi oluştur
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Birkaç büyük şok ekle
        shock_indices = np.random.choice(len(dates), size=5, replace=False)
        for idx in shock_indices:
            shock = np.random.choice([-0.15, -0.10, 0.10, 0.15])
            prices[idx:] *= (1 + shock)
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        # Getiri hesapla
        df['returns'] = df['price'].pct_change()
        df = df.dropna()
        
        print(f"   ✓ {len(df)} gözlem oluşturuldu")
        print(f"   Fiyat aralığı: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"   Ortalama volatilite: {df['returns'].std():.4f}")
        
        return df


def create_manual_download_guide(output_path: str = 'data/MANUAL_DOWNLOAD_GUIDE.md'):
    """
    Manuel veri indirme için detaylı rehber oluştur.
    
    Parameters
    ----------
    output_path : str
        Rehber dosya yolu
    """
    guide = """# 📥 MANUEL VERİ İNDİRME REHBERİ

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
   C:\\Users\\asus\\Desktop\\Ders\\4.sınıf\\zamanSerisi\\Proje\\data\\
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
"""
    
    # Dosyayı kaydet
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"\n✅ Manuel indirme rehberi oluşturuldu: {output_path}\n")
    return guide

