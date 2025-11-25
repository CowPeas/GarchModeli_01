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
        beta: float = 0.05,
        use_decay: bool = True,
        shock_threshold_quantile: float = 0.95
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
        use_decay : bool, optional
            Decay factor kullanılsın mı (varsayılan: True)
        shock_threshold_quantile : float, optional
            Şok tespiti için quantile eşiği (varsayılan: 0.95)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.use_decay = use_decay
        self.shock_threshold_quantile = shock_threshold_quantile
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
            # NaN temizle
            window_clean = window[~np.isnan(window)]
            if len(window_clean) > 1:
                mass[t] = np.var(window_clean)
            else:
                mass[t] = 0.0
        
        # İlk pencere için ortalama değer kullan
        if self.window_size > 0:
            valid_mass = mass[self.window_size:]
            valid_mass = valid_mass[~np.isnan(valid_mass)]
            if len(valid_mass) > 0:
                avg_mass = np.mean(valid_mass)
            else:
                avg_mass = 0.0
            mass[:self.window_size] = avg_mass
        
        self.mass_series = mass
        return mass
    
    def compute_event_horizon(
        self,
        mass: np.ndarray,
        quantile: Optional[float] = None,
        method: str = 'quantile'
    ) -> float:
        """
        Olay ufku eşiğini hesaplar (kritik kütle).
        
        İstatistiksel olarak tanımlanmış eşik hesaplama:
        - Quantile yöntemi: quantile(mass, quantile)
        - Statistical yöntemi: mean(mass) + 3 * std(mass)
        
        Parameters
        ----------
        mass : np.ndarray
            Kütle serisi
        quantile : float, optional
            Eşik yüzdeliği, None ise self.shock_threshold_quantile kullanılır
        method : str, optional
            Hesaplama yöntemi: 'quantile' veya 'statistical' (varsayılan: 'quantile')
        
        Returns
        -------
        float
            Kritik kütle eşiği
        """
        if quantile is None:
            quantile = self.shock_threshold_quantile
        
        valid_mass = mass[mass > 0]
        if len(valid_mass) == 0:
            self.shock_threshold = 0.0
            return self.shock_threshold
        
        if method == 'statistical':
            # İstatistiksel yöntem: mean + 3*std
            mean_mass = np.mean(valid_mass)
            std_mass = np.std(valid_mass)
            self.shock_threshold = mean_mass + 3 * std_mass
        else:
            # Quantile yöntemi
            self.shock_threshold = np.quantile(valid_mass, quantile)
        
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
        
        print("\n[OPTIMIZE] GRM Parametre Optimizasyonu:")
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
                
                print(f"alpha={alpha:.2f}, beta={beta:.3f} -> RMSE={rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha
                    best_beta = beta
        
        # En iyi parametreleri ayarla
        self.alpha = best_alpha
        self.beta = best_beta
        self.shock_times = []
        
        print("-" * 50)
        print(f"[OK] En iyi parametreler: alpha={best_alpha:.2f}, "
              f"beta={best_beta:.3f}, RMSE={best_rmse:.4f}\n")
        
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
    
    def detect_shocks(
        self,
        residuals: np.ndarray,
        threshold_quantile: Optional[float] = None
    ) -> np.ndarray:
        """
        Büyük şokları tespit et (olay ufku analojisi).
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        threshold_quantile : float, optional
            Şok eşiği quantile değeri, None ise self.shock_threshold_quantile kullanılır
            
        Returns
        -------
        np.ndarray
            Şok zamanlarının indeksleri
        """
        if threshold_quantile is None:
            threshold_quantile = self.shock_threshold_quantile
        
        abs_residuals = np.abs(residuals)
        threshold = np.quantile(abs_residuals, threshold_quantile)
        shock_times = np.where(abs_residuals > threshold)[0]
        
        self.shock_times = shock_times.tolist()
        return shock_times
    
    def compute_time_since_shock(
        self,
        current_time: int,
        shock_times: Optional[np.ndarray] = None
    ) -> float:
        """
        Her zaman noktası için son şoktan geçen zamanı hesapla.
        
        Parameters
        ----------
        current_time : int
            Güncel zaman indeksi
        shock_times : np.ndarray, optional
            Şok zamanlarının indeksleri, None ise self.shock_times kullanılır
            
        Returns
        -------
        float
            Son şoktan geçen zaman (adım sayısı), hiç şok yoksa inf
        """
        if shock_times is None:
            if len(self.shock_times) == 0:
                return float('inf')
            shock_times = np.array(self.shock_times)
        
        if len(shock_times) == 0 or current_time < shock_times[0]:
            return float('inf')
        
        past_shocks = shock_times[shock_times < current_time]
        if len(past_shocks) == 0:
            return float('inf')
        
        last_shock = past_shocks[-1]
        tau = current_time - last_shock
        return float(tau)
    
    def compute_curvature_with_decay(
        self,
        residuals: np.ndarray,
        mass: np.ndarray,
        time_since_shock: np.ndarray
    ) -> np.ndarray:
        """
        Decay factor eklenmiş bükülme fonksiyonu.
        
        Γ(t) = α * M(t) * tanh(ε(t)) * decay(τ)
        decay(τ) = 1 / (1 + β * τ)
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        mass : np.ndarray
            Kütle (volatilite) dizisi
        time_since_shock : np.ndarray
            Her zaman noktası için son büyük şoktan geçen zaman
            
        Returns
        -------
        np.ndarray
            Bükülme düzeltmeleri
        """
        # Decay factor: 1 / (1 + β * τ)
        if self.use_decay:
            # Inf değerlerini büyük bir sayıyla değiştir
            tau_safe = np.where(
                np.isinf(time_since_shock),
                1e6,  # Çok büyük sayı (decay ≈ 0)
                time_since_shock
            )
            decay = 1.0 / (1.0 + self.beta * tau_safe)
        else:
            decay = np.ones_like(time_since_shock)
        
        # Base curvature
        base_curvature = self.alpha * mass * np.tanh(residuals)
        
        # With decay
        curvature = base_curvature * decay
        
        return curvature
    
    def compute_curvature_single(
        self,
        residual: float,
        mass: float,
        time_since_shock: float = 0.0
    ) -> float:
        """
        Tek bir zaman adımı için GRM bükülmesi hesapla.
        
        Parameters
        ----------
        residual : float
            Güncel rezidüel
        mass : float
            Güncel kütle
        time_since_shock : float, optional
            Son şoktan geçen zaman (varsayılan: 0.0)
            
        Returns
        -------
        float
            Bükülme düzeltmesi
        """
        if mass <= 0:
            return 0.0
        
        # Decay factor
        if self.use_decay:
            if np.isinf(time_since_shock):
                decay = 0.0
            else:
                decay = 1.0 / (1.0 + self.beta * time_since_shock)
        else:
            decay = 1.0
        
        # Base curvature
        base_curvature = self.alpha * mass * np.sign(residual)
        
        # With decay
        curvature = base_curvature * decay
        
        return curvature

