"""
Baseline ARIMA model modülü.

Bu modül, zaman serisi için baseline ARIMA modeli oluşturur ve
artıkları hesaplar.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from typing import Tuple, Optional, Dict
import warnings


class BaselineARIMA:
    """
    Baseline ARIMA model sınıfı.
    
    Bu sınıf, ARIMA modeli oluşturur, optimal parametreleri bulur,
    tahmin yapar ve artıkları hesaplar.
    
    Attributes
    ----------
    best_order : Tuple[int, int, int]
        Optimal ARIMA(p,d,q) parametreleri
    model : ARIMA
        Eğitilmiş ARIMA modeli
    """
    
    def __init__(self):
        """BaselineARIMA sınıfını başlatır."""
        self.best_order = None
        self.model = None
        self.fitted_model = None
        self.train_residuals = None
    
    def grid_search(
        self,
        train_data: pd.Series,
        val_data: pd.Series,
        p_range: list = [0, 1, 2],
        d_range: list = [0, 1],
        q_range: list = [0, 1, 2],
        verbose: bool = True
    ) -> Tuple[int, int, int]:
        """
        Grid search ile optimal ARIMA parametrelerini bulur.
        
        Parameters
        ----------
        train_data : pd.Series
            Eğitim verisi
        val_data : pd.Series
            Doğrulama verisi
        p_range : list, optional
            AR sırası arama aralığı (varsayılan: [0, 1, 2])
        d_range : list, optional
            Fark alma sırası (varsayılan: [0, 1])
        q_range : list, optional
            MA sırası arama aralığı (varsayılan: [0, 1, 2])
        verbose : bool, optional
            İlerleme bilgisi göster (varsayılan: True)
        
        Returns
        -------
        Tuple[int, int, int]
            Optimal (p, d, q) parametreleri
        """
        best_rmse = np.inf
        best_order = None
        
        warnings.filterwarnings('ignore')
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        # Model oluştur ve eğit
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted = model.fit()
                        
                        # Doğrulama seti üzerinde tahmin yap
                        forecast = fitted.forecast(steps=len(val_data))
                        
                        # RMSE hesapla
                        rmse = np.sqrt(np.mean((val_data - forecast) ** 2))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_order = (p, d, q)
                        
                        if verbose:
                            print(f"ARIMA{(p, d, q)}: RMSE = {rmse:.4f}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"ARIMA{(p, d, q)}: Hata - {str(e)[:50]}")
                        continue
        
        warnings.filterwarnings('default')
        
        self.best_order = best_order
        
        if verbose:
            print(f"\n✓ En iyi model: ARIMA{best_order}, RMSE = {best_rmse:.4f}")
        
        return best_order
    
    def fit(
        self,
        data: pd.Series,
        order: Optional[Tuple[int, int, int]] = None
    ):
        """
        ARIMA modelini eğitir.
        
        Parameters
        ----------
        data : pd.Series
            Eğitim verisi
        order : Tuple[int, int, int], optional
            ARIMA(p,d,q) parametreleri, None ise best_order kullanılır
        """
        if order is None:
            if self.best_order is None:
                raise ValueError(
                    "Order parametresi belirtilmeli veya önce "
                    "grid_search() çalıştırılmalı."
                )
            order = self.best_order
        
        warnings.filterwarnings('ignore')
        self.model = ARIMA(data, order=order)
        self.fitted_model = self.model.fit()
        warnings.filterwarnings('default')
        
        # Eğitim artıklarını sakla
        self.train_residuals = self.fitted_model.resid
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Gelecek değerleri tahmin eder.
        
        Parameters
        ----------
        steps : int, optional
            Kaç adım ileriye tahmin yapılacağı (varsayılan: 1)
        
        Returns
        -------
        np.ndarray
            Tahmin değerleri
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def get_residuals(self) -> np.ndarray:
        """
        Model artıklarını döndürür.
        
        Returns
        -------
        np.ndarray
            Artık dizisi
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        return self.fitted_model.resid.values
    
    def diagnose_residuals(self) -> Dict[str, any]:
        """
        Artıkların istatistiksel özelliklerini analiz eder.
        
        Returns
        -------
        Dict[str, any]
            Tanısal test sonuçları:
            - ljung_box_pvalue: Ljung-Box test p-değeri (otokorelasyon)
            - arch_lm_pvalue: ARCH-LM test p-değeri (koşullu varyans)
            - mean: Artıkların ortalaması
            - std: Artıkların standart sapması
        """
        if self.fitted_model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce fit() çağırın.")
        
        residuals = self.get_residuals()
        
        # Ljung-Box testi (otokorelasyon)
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=False)
        ljung_box_pvalue = np.min(lb_test['lb_pvalue'])
        
        # ARCH-LM testi (koşullu değişen varyans)
        try:
            arch_test = het_arch(residuals, nlags=10)
            arch_lm_pvalue = arch_test[1]
        except:
            arch_lm_pvalue = np.nan
        
        diagnostics = {
            'ljung_box_pvalue': ljung_box_pvalue,
            'arch_lm_pvalue': arch_lm_pvalue,
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'autocorr_detected': ljung_box_pvalue < 0.05,
            'heteroscedasticity_detected': arch_lm_pvalue < 0.05
        }
        
        return diagnostics

