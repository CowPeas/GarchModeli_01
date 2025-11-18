"""
Gerçek finansal veri yükleme modülü.

Bu modül, Yahoo Finance ve diğer kaynaklardan gerçek finansal zaman
serilerini indirmek ve hazırlamak için fonksiyonlar içerir.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
import warnings
import ssl


class RealDataLoader:
    """
    Gerçek finansal veri yükleme sınıfı.
    
    Bu sınıf, çeşitli finansal veri kaynaklarından veri indirip
    GRM modelleri için uygun formata dönüştürür.
    
    Attributes
    ----------
    data_source : str
        Veri kaynağı ('yahoo', 'csv', vb.)
    """
    
    def __init__(self, data_source: str = 'yahoo'):
        """
        RealDataLoader sınıfını başlatır.
        
        Parameters
        ----------
        data_source : str
            Veri kaynağı seçimi
        """
        self.data_source = data_source
    
    def load_yahoo_finance(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        column: str = 'Close',
        verify_ssl: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Yahoo Finance'dan veri yükle (GÜÇLÜ SSL BYPASS).
        
        Parameters
        ----------
        ticker : str
            Hisse senedi sembolü (örn: 'BTC-USD', 'AAPL')
        start_date : str
            Başlangıç tarihi (YYYY-MM-DD)
        end_date : str
            Bitiş tarihi (YYYY-MM-DD)
        column : str
            Kullanılacak sütun (default: 'Close')
        verify_ssl : bool
            SSL sertifikası doğrulaması (default: False)
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (veri, metadata)
        """
        print(f"📥 {ticker} verisi indiriliyor...")
        print(f"   Tarih aralığı: {start_date} - {end_date}")
        
        # ============================================================
        # GÜÇLÜ SSL BYPASS - TÜM YÖNTEMLER BİRDEN
        # ============================================================
        import os
        
        # 1. Ortam değişkenleri
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['SSL_CERT_FILE'] = ''
        os.environ['HTTPS_PROXY'] = ''
        
        # 2. SSL context'i devre dışı bırak
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
        
        # 3. urllib3 uyarılarını kapat
        try:
            import urllib3
            urllib3.disable_warnings()
        except:
            pass
        
        warnings.filterwarnings('ignore')
        
        # 4. Güçlü requests session
        import requests
        from requests.adapters import HTTPAdapter
        
        try:
            from requests.packages.urllib3.util.retry import Retry
        except:
            from urllib3.util.retry import Retry
        
        session = requests.Session()
        session.verify = False
        session.trust_env = False
        
        # Retry stratejisi
        retry = Retry(
            total=10,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # User-Agent ve Headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # ============================================================
        # YFINANCE İLE VERİ İNDİRME
        # ============================================================
        try:
            import yfinance as yf
            
            # YFinance 0.2.66+ kendi curl_cffi session'ını kullanır
            # Biz session atamayacağız, yfinance kendi handle etsin
            print("   [DOWNLOAD] Indirme baslatiiliyor...")
            
            # Veriyi indir (session OLMADAN)
            ticker_obj = yf.Ticker(ticker)
            
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                back_adjust=False,
                repair=True
            )
            
            # Boşsa alternatif yöntem
            if data.empty:
                print("   [RETRY] history() bos dondu, download() deneniyor...")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
            
            if data.empty:
                raise ValueError(f"{ticker} icin veri bulunamadi (bos DataFrame)!")
            
            print(f"   [OK] {len(data)} gozlem indirildi")
            
            # Secilen sutunu al
            if column not in data.columns:
                print(f"   [WARN] '{column}' sutunu bulunamadi, kullanilabilir: {data.columns.tolist()}")
                column = 'Close' if 'Close' in data.columns else data.columns[0]
                print(f"   [INFO] '{column}' sutunu kullanilacak")
            
            df = pd.DataFrame({
                'date': data.index,
                'price': data[column].values
            })
            
            # Getiri hesapla
            df['returns'] = df['price'].pct_change()
            df = df.dropna()
            
            # Metadata
            metadata = {
                'asset': ticker,
                'period': f"{start_date} - {end_date}",
                'n_samples': len(df),
                'start_date': df['date'].min(),
                'end_date': df['date'].max(),
                'mean_return': df['returns'].mean(),
                'std_return': df['returns'].std(),
                'min_price': df['price'].min(),
                'max_price': df['price'].max(),
                'data_type': 'real_yahoo_finance',
                'column_used': column
            }
            
            print(f"\n   [SUCCESS] VERI YUKLEME BASARILI!")
            print(f"   [STATS] Istatistikler:")
            print(f"      - Gozlem sayisi: {len(df)}")
            print(f"      - Tarih araligi: {metadata['start_date']} - {metadata['end_date']}")
            print(f"      - Ortalama getiri: {metadata['mean_return']:.6f}")
            print(f"      - Std sapma: {metadata['std_return']:.6f}")
            print(f"      - Fiyat araligi: ${metadata['min_price']:.2f} - ${metadata['max_price']:.2f}\n")
            
            return df, metadata
            
        except ImportError as ie:
            print(f"\n[ERROR] HATA: yfinance paketi yuklu degil!")
            print(f"   Cozum: pip install yfinance")
            raise ImportError("yfinance paketi bulunamadi") from ie
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n{'='*80}")
            print(f"[ERROR] VERI INDIRME HATASI!")
            print(f"{'='*80}")
            print(f"Ticker: {ticker}")
            print(f"Hata: {error_msg[:200]}")
            print(f"{'='*80}\n")
            
            # Hata turune gore oneriler
            if "SSL" in error_msg or "certificate" in error_msg or "curl: (77)" in error_msg or "curl_cffi" in error_msg:
                print("[SSL] SSL/CURL SORUNU TESPIT EDILDI!")
                print("-" * 80)
                print("\n[COZUM-1] MANUEL VERI INDIRME (EN KOLAY - %100 CALISIR)")
                print("-" * 80)
                print(f"1. Tarayicinizda acin:")
                print(f"   https://finance.yahoo.com/quote/{ticker}/history")
                print(f"\n2. Tarih araligi secin: {start_date} - {end_date}")
                print(f"\n3. 'Download' butonuna tiklayin")
                print(f"\n4. Indirilen CSV'yi buraya kaydedin:")
                print(f"   data/{ticker}.csv")
                print(f"\n5. Programi tekrar calistirin (otomatik CSV'den yukleyecek)")
                
                print("\n\n[COZUM-2] PYTHON PAKETLERINI GUNCELLE")
                print("-" * 80)
                print("Su komutlari sirayla calistirin:")
                print("  pip install --upgrade yfinance")
                print("  pip install --upgrade curl-cffi")
                print("  pip install --upgrade certifi")
                
                print("\n\n[COZUM-3] ALTERNATIF VARLIK DENE")
                print("-" * 80)
                print("config_phase3.py'de farkli bir varlik deneyin:")
                print("  REAL_DATA_CONFIG = {")
                print("      'ticker': 'AAPL',  # veya 'GOOGL', 'MSFT'")
                print("      ...")
                print("  }")
                
                print("\n\n[COZUM-4] GERCEKCI SENTETIK VERI (OTOMATIK)")
                print("-" * 80)
                print("Program otomatik olarak gercekci sentetik veri olusturur.")
                print("(Kripto benzeri volatilite ve trendler)")
                
                print(f"\n{'='*80}\n")
            
            raise ValueError(f"{ticker} icin veri indirilemedi: {error_msg}")


def load_popular_assets(
    assets: list = ['BTC-USD', '^GSPC', 'AAPL'],
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 730
) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Popüler varlıklar için veri yükle.
    
    Parameters
    ----------
    assets : list
        Varlık sembolleri listesi
    start_date : str
        Başlangıç tarihi (None ise otomatik)
    end_date : str
        Bitiş tarihi (None ise bugün)
    lookback_days : int
        Geriye bakış günü (start_date None ise)
        
    Returns
    -------
    Dict[str, Tuple[pd.DataFrame, Dict]]
        {ticker: (df, metadata)} dictionary
    """
    loader = RealDataLoader()
    
    # Tarih hesapla
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"\n{'='*80}")
    print(f"📦 {len(assets)} VARLIK İÇİN VERİ YÜKLEME")
    print(f"{'='*80}")
    print(f"Tarih aralığı: {start_date} - {end_date}\n")
    
    results = {}
    success_count = 0
    fail_count = 0
    
    for ticker in assets:
        try:
            df, metadata = loader.load_yahoo_finance(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            results[ticker] = (df, metadata)
            success_count += 1
            print(f"✅ {ticker}: {len(df)} gözlem\n")
            
        except Exception as e:
            fail_count += 1
            print(f"❌ {ticker}: HATA - {str(e)[:100]}\n")
    
    print(f"{'='*80}")
    print(f"✅ Başarılı: {success_count}/{len(assets)}")
    print(f"❌ Başarısız: {fail_count}/{len(assets)}")
    print(f"{'='*80}\n")
    
    return results
