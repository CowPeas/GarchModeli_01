"""
Kütleçekimsel Artık Modeli (GRM) - Schwarzschild Rejimi.

Bu modül, Schwarzschild metriğinden ilham alan basit bükülme fonksiyonu
ile artıkları modelleyen GRM'yi uygular (FAZE 1).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class SchwarzschildGRM:
    """
    Schwarzschild Kütleçekimsel Artık Modeli.
    
    Bu sınıf, sadece "kütle" parametresi (volatilite) kullanarak
    basit Schwarzschild bükülme fonksiyonu uygular.
    
    Γ(t+1) = α * M(t) * sign(ε(t)) * decay(τ)
    
    Attributes
    ----------
    window_size : int
        Volatilite hesaplama pencere boyutu
    alpha : float
        Kütleçekimsel etkileşim katsayısı
    beta : float
        Sönümleme hızı parametresi
    shock_threshold : float
        Olay ufku eşiği (kritik varyans)
    """
    
    def __init__(
        self,
        window_size: int = 20,
        alpha: float = 1.0,
        beta: float = 0.05
    ):
        """
        SchwarzschildGRM sınıfını başlatır.
        
        Parameters
        ----------
        window_size : int, optional
            Volatilite hesaplama pencere boyutu (varsayılan: 20)
        alpha : float, optional
            Kütleçekimsel etkileşim katsayısı (varsayılan: 1.0)
        beta : float, optional
            Sönümleme hızı (varsayılan: 0.05)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.shock_threshold = None
        self.mass_series = None
        self.shock_times = []
    
    def compute_mass(self, residuals: np.ndarray) -> np.ndarray:
        """
        Kütle parametresi M(t)'yi hesaplar (yerel volatilite).
        
        M(t) = variance(ε[t-w:t])
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık serisi
        
        Returns
        -------
        np.ndarray
            Kütle serisi M(t)
        """
        n = len(residuals)
        mass = np.zeros(n)
        
        for t in range(self.window_size, n):
            window = residuals[t - self.window_size:t]
            mass[t] = np.var(window)
        
        # İlk pencere için ortalama değer kullan
        if self.window_size > 0:
            avg_mass = np.mean(mass[self.window_size:])
            mass[:self.window_size] = avg_mass
        
        self.mass_series = mass
        return mass
    
    def compute_event_horizon(
        self,
        mass: np.ndarray,
        quantile: float = 0.99
    ) -> float:
        """
        Olay ufku eşiğini hesaplar (kritik kütle).
        
        Parameters
        ----------
        mass : np.ndarray
            Kütle serisi
        quantile : float, optional
            Eşik yüzdeliği (varsayılan: 0.99)
        
        Returns
        -------
        float
            Kritik kütle eşiği
        """
        self.shock_threshold = np.quantile(mass[mass > 0], quantile)
        return self.shock_threshold
    
    def compute_decay(self, time_since_shock: int) -> float:
        """
        Sönümleme faktörünü hesaplar.
        
        decay(τ) = 1 / (1 + β*τ)
        
        Parameters
        ----------
        time_since_shock : int
            Son şoktan bu yana geçen zaman
        
        Returns
        -------
        float
            Sönümleme faktörü [0, 1]
        """
        decay = 1.0 / (1.0 + self.beta * time_since_shock)
        return decay
    
    def compute_curvature(
        self,
        residuals: np.ndarray,
        mass: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Bükülme fonksiyonunu hesaplar.
        
        Γ(t+1) = α * M(t) * sign(ε(t)) * decay(τ)
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık serisi
        mass : np.ndarray, optional
            Kütle serisi, None ise hesaplanır
        
        Returns
        -------
        np.ndarray
            Bükülme serisi Γ(t)
        """
        if mass is None:
            mass = self.compute_mass(residuals)
        
        n = len(residuals)
        curvature = np.zeros(n)
        time_since_last_shock = 0
        
        for t in range(1, n):
            # Şok yönü
            sign = np.sign(residuals[t - 1])
            if sign == 0:
                sign = 1  # Nötr durumda pozitif kabul et
            
            # Büyük şok algılandı mı?
            if mass[t] > self.shock_threshold:
                time_since_last_shock = 0
                self.shock_times.append(t)
            else:
                time_since_last_shock += 1
            
            # Sönümleme
            decay = self.compute_decay(time_since_last_shock)
            
            # Bükülme
            curvature[t] = self.alpha * np.sqrt(mass[t]) * sign * decay
        
        return curvature
    
    def fit(
        self,
        residuals: np.ndarray,
        alpha_range: list = [0.1, 0.5, 1.0, 2.0],
        beta_range: list = [0.01, 0.05, 0.1],
        val_residuals: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Grid search ile optimal α ve β parametrelerini bulur.
        
        Parameters
        ----------
        residuals : np.ndarray
            Eğitim artıkları
        alpha_range : list, optional
            α arama aralığı
        beta_range : list, optional
            β arama aralığı
        val_residuals : np.ndarray, optional
            Doğrulama artıkları, None ise train kullanılır
        
        Returns
        -------
        Tuple[float, float]
            Optimal (α, β) parametreleri
        """
        if val_residuals is None:
            val_residuals = residuals
        
        # Kütle hesapla
        mass = self.compute_mass(residuals)
        self.compute_event_horizon(mass)
        
        best_rmse = np.inf
        best_alpha = None
        best_beta = None
        
        print("\n🔍 GRM Parametre Optimizasyonu:")
        print("-" * 50)
        
        for alpha in alpha_range:
            for beta in beta_range:
                # Geçici parametrelerle bükülme hesapla
                self.alpha = alpha
                self.beta = beta
                self.shock_times = []
                
                curvature = self.compute_curvature(residuals, mass)
                
                # RMSE hesapla (1 adım ileriye)
                predictions = curvature[:-1]
                targets = val_residuals[1:]
                
                # Uygun uzunluğa getir
                min_len = min(len(predictions), len(targets))
                rmse = np.sqrt(np.mean(
                    (predictions[:min_len] - targets[:min_len]) ** 2
                ))
                
                print(f"α={alpha:.2f}, β={beta:.3f} → RMSE={rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha
                    best_beta = beta
        
        # En iyi parametreleri ayarla
        self.alpha = best_alpha
        self.beta = best_beta
        self.shock_times = []
        
        print("-" * 50)
        print(f"✓ En iyi parametreler: α={best_alpha:.2f}, "
              f"β={best_beta:.3f}, RMSE={best_rmse:.4f}\n")
        
        return best_alpha, best_beta
    
    def get_diagnostics(self) -> Dict[str, any]:
        """
        Model tanısal bilgilerini döndürür.
        
        Returns
        -------
        Dict[str, any]
            Tanısal bilgiler (eşik, şok sayısı, vb.)
        """
        diagnostics = {
            'alpha': self.alpha,
            'beta': self.beta,
            'window_size': self.window_size,
            'shock_threshold': self.shock_threshold,
            'n_shocks_detected': len(self.shock_times),
            'shock_times': self.shock_times,
            'avg_mass': np.mean(self.mass_series) if self.mass_series is not None else None,
            'max_mass': np.max(self.mass_series) if self.mass_series is not None else None
        }
        
        return diagnostics

