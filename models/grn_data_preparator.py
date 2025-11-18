"""
GRN Data Preparator - Veri Hazırlama Modülü.

Bu modül, GRN eğitimi için gerekli feature'ları hazırlar.

FAZE 5: PIML TEMEL ENTEGRASYONU
"""

import numpy as np
from typing import Tuple, Optional


class GRNDataPreparator:
    """
    GRN eğitimi için veri hazırlama sınıfı.
    
    Artıklardan GRM feature'larını (M, a, τ, ε) çıkarır ve
    hedef değerleri (gelecekteki artık) hazırlar.
    """
    
    def __init__(self, shock_threshold_quantile: float = 0.95):
        """
        GRNDataPreparator sınıfını başlatır.
        
        Parameters
        ----------
        shock_threshold_quantile : float, optional
            Şok tespiti için quantile eşiği (varsayılan: 0.95)
        """
        self.shock_threshold_quantile = shock_threshold_quantile
    
    def prepare_features(
        self,
        residuals: np.ndarray,
        window_size: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GRM feature'larını hazırla.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        window_size : int, optional
            Pencere boyutu (varsayılan: 20)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (mass, spin, tau, residuals_history, targets)
        """
        n = len(residuals)
        mass_list = []
        spin_list = []
        tau_list = []
        residuals_history_list = []
        targets_list = []
        
        # Şok tespiti için eşik hesapla
        abs_residuals = np.abs(residuals)
        shock_threshold = np.quantile(abs_residuals, self.shock_threshold_quantile)
        
        for t in range(window_size, n - 1):
            window = residuals[t - window_size:t]
            
            # Features
            mass = np.var(window)
            
            # Spin (otokorelasyon)
            if len(window) > 1 and np.std(window) > 1e-8:
                spin = np.corrcoef(window[1:], window[:-1])[0, 1]
                spin = np.clip(spin, -1, 1)
            else:
                spin = 0.0
            
            # Tau (time since last shock)
            tau = self.compute_tau(residuals[:t], shock_threshold)
            
            # Residuals history
            residuals_history = window
            
            mass_list.append(mass)
            spin_list.append(spin)
            tau_list.append(tau)
            residuals_history_list.append(residuals_history)
            targets_list.append(residuals[t + 1])
        
        return (
            np.array(mass_list),
            np.array(spin_list),
            np.array(tau_list),
            np.array(residuals_history_list),
            np.array(targets_list)
        )
    
    def compute_tau(
        self,
        residuals: np.ndarray,
        threshold: float
    ) -> float:
        """
        Time since last shock hesapla.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi (t zamanına kadar)
        threshold : float
            Şok eşiği
            
        Returns
        -------
        float
            Son şoktan geçen zaman, hiç şok yoksa len(residuals)
        """
        abs_res = np.abs(residuals)
        shock_indices = np.where(abs_res > threshold)[0]
        
        if len(shock_indices) == 0:
            return float(len(residuals))
        
        last_shock = shock_indices[-1]
        tau = len(residuals) - last_shock
        return float(tau)

