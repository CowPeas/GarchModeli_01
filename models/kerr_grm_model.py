"""
Kütleçekimsel Artık Modeli (GRM) - Kerr Rejimi.

Bu modül, Kerr metriğinden ilham alan gelişmiş bükülme fonksiyonu
ile artıkları modelleyen GRM'yi uygular (FAZE 2).

Kerr rejimi, Schwarzschild'e ek olarak "dönme" parametresi (otokorelasyon)
içerir ve non-linear aktivasyon fonksiyonu kullanır.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.stattools import acf


class KerrGRM:
    """
    Kerr Kütleçekimsel Artık Modeli.
    
    Bu sınıf, hem "kütle" (volatilite) hem de "dönme" (otokorelasyon)
    parametrelerini kullanarak Kerr bükülme fonksiyonu uygular.
    
    Γ(t+1) = tanh(α * M(t) * [1 + γ*a(t)]) * decay(τ)
    
    Attributes
    ----------
    window_size : int
        Volatilite ve otokorelasyon hesaplama pencere boyutu
    alpha : float
        Kütleçekimsel etkileşim katsayısı
    beta : float
        Sönümleme hızı parametresi
    gamma : float
        Dönme etkisinin ağırlığı
    use_tanh : bool
        Non-linear aktivasyon kullanılsın mı
    shock_threshold : float
        Olay ufku eşiği (kritik varyans)
    regime : str
        Model rejimi: 'schwarzschild', 'kerr', veya 'adaptive'
    """
    
    def __init__(
        self,
        window_size: int = 20,
        alpha: float = 1.0,
        beta: float = 0.05,
        gamma: float = 0.5,
        use_tanh: bool = True,
        regime: str = 'adaptive',
        use_decay: bool = True,
        shock_threshold_quantile: float = 0.95
    ):
        """
        KerrGRM sınıfını başlatır.
        
        Parameters
        ----------
        window_size : int, optional
            Pencere boyutu (varsayılan: 20)
        alpha : float, optional
            Kütleçekimsel etkileşim katsayısı (varsayılan: 1.0)
        beta : float, optional
            Sönümleme hızı (varsayılan: 0.05)
        gamma : float, optional
            Dönme etkisinin ağırlığı (varsayılan: 0.5)
        use_tanh : bool, optional
            Non-linear aktivasyon kullan (varsayılan: True)
        regime : str, optional
            Model rejimi: 'schwarzschild', 'kerr', 'adaptive'
            (varsayılan: 'adaptive')
        use_decay : bool, optional
            Decay factor kullanılsın mı (varsayılan: True)
        shock_threshold_quantile : float, optional
            Şok tespiti için quantile eşiği (varsayılan: 0.95)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_tanh = use_tanh
        self.regime = regime
        self.use_decay = use_decay
        self.shock_threshold_quantile = shock_threshold_quantile
        
        self.shock_threshold = None
        self.mass_series = None
        self.spin_series = None
        self.shock_times = []
        self.detected_regime = None
    
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
        if self.window_size > 0 and np.any(mass[self.window_size:] > 0):
            avg_mass = np.mean(mass[self.window_size:])
            mass[:self.window_size] = avg_mass
        
        self.mass_series = mass
        return mass
    
    def compute_spin(self, residuals: np.ndarray) -> np.ndarray:
        """
        Dönme parametresi a(t)'yi hesaplar (otokorelasyon).
        
        a(t) = ACF(ε[t-w:t], lag=1)
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık serisi
        
        Returns
        -------
        np.ndarray
            Dönme serisi a(t) ∈ [-1, 1]
        """
        n = len(residuals)
        spin = np.zeros(n)
        
        for t in range(self.window_size, n):
            window = residuals[t - self.window_size:t]
            
            # Otokorelasyon hesapla (lag=1)
            try:
                if len(window) > 2 and np.std(window) > 1e-8:
                    acf_values = acf(window, nlags=1, fft=False)
                    spin[t] = acf_values[1]  # lag=1
                else:
                    spin[t] = 0.0
            except:
                spin[t] = 0.0
            
            # Sınırla: [-1, 1]
            spin[t] = np.clip(spin[t], -1, 1)
        
        # İlk pencere için ortalama değer
        if self.window_size > 0 and n > self.window_size:
            avg_spin = np.mean(spin[self.window_size:])
            spin[:self.window_size] = avg_spin
        
        self.spin_series = spin
        return spin
    
    def detect_regime(
        self,
        residuals: np.ndarray,
        significance_level: float = 0.05
    ) -> str:
        """
        Artıkların özelliklerine göre optimal rejimi tespit eder.
        
        Ljung-Box testi ile otokorelasyon varlığını kontrol eder.
        Eğer anlamlı otokorelasyon varsa Kerr, yoksa Schwarzschild.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık serisi
        significance_level : float, optional
            Anlamlılık seviyesi (varsayılan: 0.05)
        
        Returns
        -------
        str
            'schwarzschild' veya 'kerr'
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            # Ljung-Box testi
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=False)
            min_pvalue = np.min(lb_test['lb_pvalue'])
            
            if min_pvalue < significance_level:
                regime = 'kerr'  # Otokorelasyon var
            else:
                regime = 'schwarzschild'  # Otokorelasyon yok
        except:
            # Hata durumunda basit ACF kontrolü
            try:
                acf_vals = acf(residuals, nlags=5, fft=False)
                if np.any(np.abs(acf_vals[1:]) > 0.2):
                    regime = 'kerr'
                else:
                    regime = 'schwarzschild'
            except:
                regime = 'schwarzschild'  # Varsayılan
        
        self.detected_regime = regime
        return regime
    
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
        Gelişmiş sönümleme faktörünü hesaplar.
        
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
        mass: Optional[np.ndarray] = None,
        spin: Optional[np.ndarray] = None,
        use_detected_regime: bool = True
    ) -> np.ndarray:
        """
        Kerr bükülme fonksiyonunu hesaplar.
        
        Schwarzschild: Γ(t) = α * M(t) * sign(ε)
        Kerr: Γ(t) = α * M(t) * [1 + γ*a(t)] * sign(ε)
        Non-linear: Γ(t) = tanh(...) * decay(τ)
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık serisi
        mass : np.ndarray, optional
            Kütle serisi, None ise hesaplanır
        spin : np.ndarray, optional
            Dönme serisi, None ise hesaplanır
        use_detected_regime : bool, optional
            Otomatik rejim tespiti kullan (varsayılan: True)
        
        Returns
        -------
        np.ndarray
            Bükülme serisi Γ(t)
        """
        if mass is None:
            mass = self.compute_mass(residuals)
        
        # Rejim belirleme
        if self.regime == 'adaptive' and use_detected_regime:
            current_regime = self.detect_regime(residuals)
        else:
            current_regime = self.regime
        
        # Kerr rejimi için dönme hesapla
        if current_regime == 'kerr':
            if spin is None:
                spin = self.compute_spin(residuals)
        else:
            spin = np.zeros_like(mass)
        
        n = len(residuals)
        curvature = np.zeros(n)
        time_since_last_shock = 0
        
        for t in range(1, n):
            # Şok yönü
            sign = np.sign(residuals[t - 1])
            if sign == 0:
                sign = 1
            
            # Büyük şok algılandı mı?
            if self.shock_threshold is not None and mass[t] > self.shock_threshold:
                time_since_last_shock = 0
                self.shock_times.append(t)
            else:
                time_since_last_shock += 1
            
            # Sönümleme
            decay = self.compute_decay(time_since_last_shock)
            
            # Bükülme hesapla
            if current_regime == 'kerr':
                # Kerr: kütle + dönme
                base_curvature = self.alpha * np.sqrt(mass[t]) * (1 + self.gamma * spin[t])
            else:
                # Schwarzschild: sadece kütle
                base_curvature = self.alpha * np.sqrt(mass[t])
            
            # Non-linear aktivasyon
            if self.use_tanh:
                curvature[t] = np.tanh(base_curvature) * sign * decay
            else:
                curvature[t] = base_curvature * sign * decay
        
        return curvature
    
    def fit(
        self,
        residuals: np.ndarray,
        alpha_range: list = [0.1, 0.5, 1.0, 2.0],
        beta_range: list = [0.01, 0.05, 0.1],
        gamma_range: list = [0, 0.5, 1.0],
        val_residuals: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[float, float, float]:
        """
        Grid search ile optimal α, β ve γ parametrelerini bulur.
        
        Parameters
        ----------
        residuals : np.ndarray
            Eğitim artıkları
        alpha_range : list, optional
            α arama aralığı
        beta_range : list, optional
            β arama aralığı
        gamma_range : list, optional
            γ arama aralığı (Kerr için)
        val_residuals : np.ndarray, optional
            Doğrulama artıkları, None ise train kullanılır
        verbose : bool, optional
            İlerleme bilgisi göster (varsayılan: True)
        
        Returns
        -------
        Tuple[float, float, float]
            Optimal (α, β, γ) parametreleri
        """
        if val_residuals is None:
            val_residuals = residuals
        
        # Kütle ve rejim tespiti
        mass = self.compute_mass(residuals)
        self.compute_event_horizon(mass)
        
        # Rejim belirle
        if self.regime == 'adaptive':
            detected_regime = self.detect_regime(residuals)
        else:
            detected_regime = self.regime
        
        # Dönme hesapla (Kerr için)
        if detected_regime == 'kerr':
            spin = self.compute_spin(residuals)
        else:
            spin = None
            gamma_range = [0]  # Schwarzschild'de gamma kullanılmaz
        
        best_rmse = np.inf
        best_alpha = None
        best_beta = None
        best_gamma = None
        
        if verbose:
            print(f"\n🔍 Kerr GRM Parametre Optimizasyonu (Rejim: {detected_regime}):")
            print("-" * 60)
        
        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    # Geçici parametrelerle bükülme hesapla
                    self.alpha = alpha
                    self.beta = beta
                    self.gamma = gamma
                    self.shock_times = []
                    
                    curvature = self.compute_curvature(
                        residuals, mass, spin, use_detected_regime=False
                    )
                    
                    # RMSE hesapla
                    predictions = curvature[:-1]
                    targets = val_residuals[1:]
                    
                    min_len = min(len(predictions), len(targets))
                    rmse = np.sqrt(np.mean(
                        (predictions[:min_len] - targets[:min_len]) ** 2
                    ))
                    
                    if verbose:
                        print(f"alpha={alpha:.2f}, beta={beta:.3f}, gamma={gamma:.2f} -> RMSE={rmse:.4f}")
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
        
        # En iyi parametreleri ayarla
        self.alpha = best_alpha
        self.beta = best_beta
        self.gamma = best_gamma
        self.shock_times = []
        
        if verbose:
            print("-" * 60)
            print(f"[OK] En iyi parametreler: alpha={best_alpha:.2f}, "
                  f"beta={best_beta:.3f}, gamma={best_gamma:.2f}, RMSE={best_rmse:.4f}")
            print(f"[OK] Tespit edilen rejim: {detected_regime}\n")
        
        return best_alpha, best_beta, best_gamma
    
    def get_diagnostics(self) -> Dict[str, any]:
        """
        Model tanısal bilgilerini döndürür.
        
        Returns
        -------
        Dict[str, any]
            Tanısal bilgiler
        """
        diagnostics = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'window_size': self.window_size,
            'use_tanh': self.use_tanh,
            'regime': self.regime,
            'detected_regime': self.detected_regime,
            'shock_threshold': self.shock_threshold,
            'n_shocks_detected': len(self.shock_times),
            'shock_times': self.shock_times,
            'avg_mass': np.mean(self.mass_series) if self.mass_series is not None else None,
            'max_mass': np.max(self.mass_series) if self.mass_series is not None else None,
            'avg_spin': np.mean(np.abs(self.spin_series)) if self.spin_series is not None else None,
            'max_spin': np.max(np.abs(self.spin_series)) if self.spin_series is not None else None
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
    
    def compute_curvature_single(
        self,
        residual: float,
        mass: float,
        spin: float = 0.0,
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
        spin : float, optional
            Güncel spin (Kerr için)
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
        if self.use_tanh:
            base_curvature = self.alpha * mass * np.tanh(residual) + self.gamma * spin * np.sign(residual)
        else:
            base_curvature = self.alpha * mass * np.sign(residual) + self.gamma * spin * residual
        
        # With decay
        curvature = base_curvature * decay
        
        return curvature

