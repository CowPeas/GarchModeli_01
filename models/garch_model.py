"""
GARCH model implementasyonu.

Bu modül, ARCH kütüphanesini kullanarak GARCH, EGARCH ve GJR-GARCH
modellerini eğitir ve tahmin yapar.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional
from arch import arch_model


class GARCHModel:
    """
    GARCH ailesi modelleri için wrapper sınıfı.
    
    Bu sınıf, volatilite modellemesi için GARCH, EGARCH ve GJR-GARCH
    modellerini destekler.
    
    Attributes
    ----------
    model_type : str
        Model tipi ('GARCH', 'EGARCH', 'GJR-GARCH')
    p : int
        ARCH sırası
    q : int
        GARCH sırası
    mean_model : str
        Ortalama model tipi
    fitted_model : arch.univariate.base.ARCHModelResult
        Eğitilmiş model
    """
    
    def __init__(
        self,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        mean_model: str = 'Constant'
    ):
        """
        GARCH model parametreleri.
        
        Parameters
        ----------
        model_type : str, optional
            Model tipi: 'GARCH', 'EGARCH', 'GJR-GARCH' (varsayılan: 'GARCH')
        p : int, optional
            ARCH sırası (varsayılan: 1)
        q : int, optional
            GARCH sırası (varsayılan: 1)
        mean_model : str, optional
            Ortalama model: 'Constant', 'Zero', 'AR' (varsayılan: 'Constant')
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.fitted_model = None
        self.data = None
    
    def fit(self, data: pd.Series, verbose: bool = False) -> 'GARCHModel':
        """
        GARCH modelini eğit.
        
        Parameters
        ----------
        data : pd.Series
            Eğitim verisi (getiri serisi)
        verbose : bool, optional
            Eğitim çıktısını göster (varsayılan: False)
            
        Returns
        -------
        GARCHModel
            Eğitilmiş model (self)
        """
        self.data = data
        
        # Model tipi mapping
        vol_map = {
            'GARCH': 'Garch',
            'EGARCH': 'EGARCH',
            'GJR-GARCH': 'GJRGARCH'
        }
        
        vol_model = vol_map.get(self.model_type, 'Garch')
        
        # Model oluştur
        try:
            # Veriyi ölçeklendir (numerical stability için)
            scaled_data = data * 100
            
            model = arch_model(
                scaled_data,
                mean=self.mean_model,
                vol=vol_model,
                p=self.p,
                q=self.q,
                rescale=False
            )
            
            # Eğit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.fitted_model = model.fit(
                    disp='off' if not verbose else 'final',
                    show_warning=False
                )
            
            if verbose:
                print(f"[OK] {self.model_type}({self.p},{self.q}) eğitildi")
            
        except Exception as e:
            print(f"[HATA] GARCH eğitimi başarısız: {str(e)}")
            # Fallback: Basit GARCH(1,1)
            model = arch_model(
                data * 100,
                mean='Constant',
                vol='Garch',
                p=1,
                q=1
            )
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.fitted_model = model.fit(disp='off', show_warning=False)
        
        return self
    
    def predict(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> np.ndarray:
        """
        Volatilite tahmini yap.
        
        Parameters
        ----------
        horizon : int, optional
            Tahmin ufku (varsayılan: 1)
        method : str, optional
            Tahmin metodu: 'analytic', 'simulation' (varsayılan: 'analytic')
            
        Returns
        -------
        np.ndarray
            Tahmin edilen volatilite
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        try:
            forecast = self.fitted_model.forecast(
                horizon=horizon,
                method=method,
                reindex=False
            )
            
            # Variance'ı std'ye çevir ve ölçeği düzelt
            variance = forecast.variance.values[-1, :]
            volatility = np.sqrt(variance) / 100  # Ölçeği geri al
            
            return volatility
            
        except Exception as e:
            print(f"[UYARI] GARCH tahmin hatası: {str(e)}")
            # Fallback: Son volatilite
            return np.array([self.fitted_model.conditional_volatility[-1] / 100])
    
    def forecast_mean(self, horizon: int = 1) -> np.ndarray:
        """
        Ortalama (getiri) tahmini.
        
        Parameters
        ----------
        horizon : int, optional
            Tahmin ufku (varsayılan: 1)
            
        Returns
        -------
        np.ndarray
            Tahmin edilen getiri
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        try:
            forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
            mean_forecast = forecast.mean.values[-1, :] / 100  # Ölçeği geri al
            return mean_forecast
        except Exception as e:
            print(f"[UYARI] Mean forecast hatası: {str(e)}")
            return np.zeros(horizon)
    
    def get_residuals(self) -> np.ndarray:
        """
        Standardize artıkları döndür.
        
        Returns
        -------
        np.ndarray
            Standardize artıklar
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        return self.fitted_model.std_resid
    
    def get_volatility(self) -> np.ndarray:
        """
        Conditional volatility serisini döndür.
        
        Returns
        -------
        np.ndarray
            Conditional volatility (ölçeklendirilmiş)
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        return self.fitted_model.conditional_volatility / 100
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Model parametrelerini döndür.
        
        Returns
        -------
        Dict[str, float]
            Model parametreleri
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        params = self.fitted_model.params.to_dict()
        return params
    
    def grid_search(
        self,
        train_data: pd.Series,
        val_data: pd.Series,
        p_range: list = [1, 2],
        q_range: list = [1, 2],
        verbose: bool = False
    ) -> Tuple[int, int]:
        """
        Grid search ile optimal (p, q) parametrelerini bul.
        
        Parameters
        ----------
        train_data : pd.Series
            Eğitim verisi
        val_data : pd.Series
            Doğrulama verisi
        p_range : list, optional
            Test edilecek p değerleri (varsayılan: [1, 2])
        q_range : list, optional
            Test edilecek q değerleri (varsayılan: [1, 2])
        verbose : bool, optional
            İlerlemeyi göster (varsayılan: False)
            
        Returns
        -------
        Tuple[int, int]
            Optimal (p, q) parametreleri
        """
        best_aic = np.inf
        best_p, best_q = 1, 1
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            for p in p_range:
                for q in q_range:
                    try:
                        # Model oluştur ve eğit
                        model = arch_model(
                            train_data * 100,
                            mean=self.mean_model,
                            vol='Garch',
                            p=p,
                            q=q
                        )
                        result = model.fit(disp='off', show_warning=False)
                        
                        # AIC ile karşılaştır
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_p, best_q = p, q
                        
                        if verbose:
                            print(f"GARCH({p},{q}): AIC = {result.aic:.2f}")
                    
                    except Exception:
                        continue
        
        if verbose:
            print(f"\n[OK] En iyi parametreler: GARCH({best_p},{best_q})")
        
        return best_p, best_q
    
    def get_info_criteria(self) -> Dict[str, float]:
        """
        Model bilgi kriterlerini döndür.
        
        Returns
        -------
        Dict[str, float]
            AIC, BIC değerleri
        """
        if self.fitted_model is None:
            raise ValueError("Model eğitilmemiş. Önce fit() çağırın.")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'loglikelihood': self.fitted_model.loglikelihood
        }


def compare_garch_models(
    train_data: pd.Series,
    val_data: pd.Series,
    model_types: list = None
) -> pd.DataFrame:
    """
    Farklı GARCH modellerini karşılaştır.
    
    Parameters
    ----------
    train_data : pd.Series
        Eğitim verisi
    val_data : pd.Series
        Doğrulama verisi
    model_types : list, optional
        Test edilecek model tipleri (varsayılan: None)
        
    Returns
    -------
    pd.DataFrame
        Model karşılaştırma tablosu
    """
    if model_types is None:
        model_types = ['GARCH', 'EGARCH', 'GJR-GARCH']
    
    results = []
    
    for model_type in model_types:
        try:
            # Model eğit
            model = GARCHModel(model_type=model_type, p=1, q=1)
            model.fit(train_data, verbose=False)
            
            # Val seti üzerinde tahmin
            # Not: GARCH için one-step-ahead forecasting gerekir
            val_predictions = []
            for i in range(len(val_data)):
                pred = model.predict(horizon=1)[0]
                val_predictions.append(pred)
            
            # RMSE (volatility tahminleri için)
            # Realized volatility hesapla (rolling window)
            realized_vol = val_data.rolling(5).std()
            rmse = np.sqrt(np.mean((realized_vol[5:] - val_predictions[:len(realized_vol)-5])**2))
            
            # Info criteria
            info = model.get_info_criteria()
            
            results.append({
                'Model': model_type,
                'AIC': info['aic'],
                'BIC': info['bic'],
                'LogLik': info['loglikelihood'],
                'Val_RMSE': rmse
            })
        
        except Exception as e:
            print(f"[HATA] {model_type} test edilemedi: {str(e)}")
            continue
    
    return pd.DataFrame(results)
