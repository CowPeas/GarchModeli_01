"""
GARCH model modülü.

Bu modül, GARCH/EGARCH gibi volatilite modellerini içerir ve
GRM modelleriyle karşılaştırma için benchmark sağlar.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings


class GARCHModel:
    """
    GARCH(p,q) model sınıfı.
    
    Bu sınıf, GARCH ailesi volatilite modellerini uygular ve
    GRM modelleriyle karşılaştırma için kullanılır.
    
    Attributes
    ----------
    p : int
        ARCH sırası
    q : int
        GARCH sırası
    model_type : str
        Model tipi: 'GARCH', 'EGARCH', 'GJR-GARCH'
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model_type: str = 'GARCH'
    ):
        """
        GARCHModel sınıfını başlatır.
        
        Parameters
        ----------
        p : int, optional
            ARCH sırası (varsayılan: 1)
        q : int, optional
            GARCH sırası (varsayılan: 1)
        model_type : str, optional
            Model tipi (varsayılan: 'GARCH')
        """
        self.p = p
        self.q = q
        self.model_type = model_type
        self.fitted_model = None
        self.mean_model = None
    
    def fit(
        self,
        data: pd.Series,
        mean_model: str = 'AR',
        ar_lags: int = 1,
        verbose: bool = True
    ):
        """
        GARCH modelini eğitir.
        
        Parameters
        ----------
        data : pd.Series
            Getiri serisi
        mean_model : str, optional
            Ortalama model: 'Constant', 'Zero', 'AR', 'ARX'
            (varsayılan: 'AR')
        ar_lags : int, optional
            AR modeli için gecikme sayısı (varsayılan: 1)
        verbose : bool, optional
            Çıktı göster (varsayılan: True)
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "arch kütüphanesi yüklü değil. "
                "Lütfen 'pip install arch' komutunu çalıştırın."
            )
        
        warnings.filterwarnings('ignore')
        
        if verbose:
            print(f"\n🔧 GARCH({self.p},{self.q}) Modeli Eğitiliyor...")
            print(f"   Ortalama model: {mean_model}")
            if mean_model == 'AR':
                print(f"   AR gecikmeleri: {ar_lags}")
        
        # GARCH modeli oluştur
        try:
            if mean_model == 'AR':
                self.mean_model = arch_model(
                    data * 100,  # Ölçeklendirme (daha iyi convergence)
                    mean=mean_model,
                    lags=ar_lags,
                    vol=self.model_type,
                    p=self.p,
                    q=self.q
                )
            else:
                self.mean_model = arch_model(
                    data * 100,
                    mean=mean_model,
                    vol=self.model_type,
                    p=self.p,
                    q=self.q
                )
            
            # Model eğitimi
            self.fitted_model = self.mean_model.fit(disp='off', show_warning=False)
            
            if verbose:
                print(f"✓ Model eğitimi tamamlandı")
                print(f"  - Log Likelihood: {self.fitted_model.loglikelihood:.2f}")
                print(f"  - AIC: {self.fitted_model.aic:.2f}")
                print(f"  - BIC: {self.fitted_model.bic:.2f}")
        
        except Exception as e:
            print(f"⚠️ GARCH eğitimi başarısız: {str(e)}")
            print(f"   Basit volatilite modeli kullanılıyor...")
            self.fitted_model = None
        
        warnings.filterwarnings('default')
    
    def predict(
        self,
        steps: int = 1,
        method: str = 'analytic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gelecek değerleri tahmin eder.
        
        Parameters
        ----------
        steps : int, optional
            Kaç adım ileriye tahmin (varsayılan: 1)
        method : str, optional
            Tahmin yöntemi: 'analytic' veya 'simulation'
            (varsayılan: 'analytic')
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Ortalama tahminler ve volatilite tahminleri
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        forecast = self.fitted_model.forecast(horizon=steps, method=method)
        
        # Ortalama ve volatilite tahminleri
        mean_forecast = forecast.mean.values[-1, :] / 100  # Ölçeği geri döndür
        variance_forecast = forecast.variance.values[-1, :] / 10000
        
        return mean_forecast, np.sqrt(variance_forecast)
    
    def get_conditional_volatility(self) -> np.ndarray:
        """
        Koşullu volatilite serisini döndürür.
        
        Returns
        -------
        np.ndarray
            Koşullu volatilite σ(t)
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        return self.fitted_model.conditional_volatility.values / 100
    
    def get_standardized_residuals(self) -> np.ndarray:
        """
        Standardize edilmiş artıkları döndürür.
        
        Returns
        -------
        np.ndarray
            Standardize artıklar
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        return self.fitted_model.std_resid.values
    
    def get_diagnostics(self) -> Dict[str, any]:
        """
        Model tanısal bilgilerini döndürür.
        
        Returns
        -------
        Dict[str, any]
            Tanısal bilgiler
        """
        if self.fitted_model is None:
            return {
                'model_type': self.model_type,
                'p': self.p,
                'q': self.q,
                'fitted': False
            }
        
        diagnostics = {
            'model_type': self.model_type,
            'p': self.p,
            'q': self.q,
            'fitted': True,
            'loglikelihood': self.fitted_model.loglikelihood,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'num_params': self.fitted_model.num_params,
            'mean_conditional_vol': self.get_conditional_volatility().mean(),
            'max_conditional_vol': self.get_conditional_volatility().max()
        }
        
        return diagnostics


class SimpleVolatilityModel:
    """
    Basit volatilite modeli (GARCH yoksa fallback).
    
    Bu sınıf, GARCH kurulamamışsa kullanılmak üzere
    basit hareketli volatilite modeli sağlar.
    """
    
    def __init__(self, window: int = 20):
        """
        SimpleVolatilityModel sınıfını başlatır.
        
        Parameters
        ----------
        window : int, optional
            Hareketli pencere boyutu (varsayılan: 20)
        """
        self.window = window
        self.volatility = None
    
    def fit(self, data: pd.Series):
        """
        Basit volatilite modelini eğitir.
        
        Parameters
        ----------
        data : pd.Series
            Getiri serisi
        """
        self.volatility = data.rolling(window=self.window).std()
        self.volatility = self.volatility.fillna(self.volatility.mean())
    
    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basit tahmin (son volatiliteyi kullan).
        
        Parameters
        ----------
        steps : int, optional
            Kaç adım ileriye (varsayılan: 1)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Sıfır ortalama ve son volatilite
        """
        last_vol = self.volatility.iloc[-1]
        return np.zeros(steps), np.full(steps, last_vol)
    
    def get_conditional_volatility(self) -> np.ndarray:
        """
        Hareketli volatilite serisini döndürür.
        
        Returns
        -------
        np.ndarray
            Volatilite serisi
        """
        return self.volatility.values
    
    def get_diagnostics(self) -> Dict[str, any]:
        """
        Basit model tanısal bilgileri.
        
        Returns
        -------
        Dict[str, any]
            Tanısal bilgiler
        """
        return {
            'model_type': 'Simple Moving Volatility',
            'window': self.window,
            'mean_volatility': self.volatility.mean(),
            'max_volatility': self.volatility.max()
        }

