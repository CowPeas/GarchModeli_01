"""
Sentetik zaman serisi veri üretici modülü.

Bu modül, GRM hipotezini test etmek için kontrollü şoklar içeren
sentetik zaman serileri oluşturur.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List


class SyntheticDataGenerator:
    """
    Sentetik zaman serisi veri üretici sınıfı.
    
    Bu sınıf, trend, mevsimsellik, ARIMA bileşeni ve kontrollü şoklar
    içeren sentetik zaman serileri oluşturur.
    
    Attributes
    ----------
    n_samples : int
        Toplam gözlem sayısı
    trend_coef : float
        Trend eğim katsayısı
    trend_intercept : float
        Trend kesişim noktası
    seasonal_amplitude : float
        Mevsimsel bileşenin genliği
    seasonal_period : int
        Mevsimsel periyot
    noise_std : float
        Beyaz gürültü standart sapması
    random_seed : int
        Rastgelelik için seed değeri
    """
    
    def __init__(
        self,
        n_samples: int = 500,
        trend_coef: float = 0.05,
        trend_intercept: float = 100.0,
        seasonal_amplitude: float = 5.0,
        seasonal_period: int = 50,
        noise_std: float = 2.0,
        random_seed: int = 42
    ):
        """
        SyntheticDataGenerator sınıfını başlatır.
        
        Parameters
        ----------
        n_samples : int, optional
            Toplam gözlem sayısı (varsayılan: 500)
        trend_coef : float, optional
            Trend eğim katsayısı (varsayılan: 0.05)
        trend_intercept : float, optional
            Trend kesişim noktası (varsayılan: 100.0)
        seasonal_amplitude : float, optional
            Mevsimsel genlik (varsayılan: 5.0)
        seasonal_period : int, optional
            Mevsimsel periyot (varsayılan: 50)
        noise_std : float, optional
            Beyaz gürültü std sapması (varsayılan: 2.0)
        random_seed : int, optional
            Rastgelelik seed'i (varsayılan: 42)
        """
        self.n_samples = n_samples
        self.trend_coef = trend_coef
        self.trend_intercept = trend_intercept
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.noise_std = noise_std
        self.random_seed = random_seed
        
        np.random.seed(self.random_seed)
    
    def _generate_trend(self) -> np.ndarray:
        """
        Lineer trend bileşeni oluşturur.
        
        Returns
        -------
        np.ndarray
            Trend bileşeni: β₀ + β₁*t
        """
        t = np.arange(self.n_samples)
        trend = self.trend_intercept + self.trend_coef * t
        return trend
    
    def _generate_seasonal(self) -> np.ndarray:
        """
        Sinüzoidal mevsimsel bileşen oluşturur.
        
        Returns
        -------
        np.ndarray
            Mevsimsel bileşen: A*sin(2π*t/T)
        """
        t = np.arange(self.n_samples)
        frequency = 2 * np.pi / self.seasonal_period
        seasonal = self.seasonal_amplitude * np.sin(frequency * t)
        return seasonal
    
    def _generate_noise(self) -> np.ndarray:
        """
        Beyaz gürültü bileşeni oluşturur.
        
        Returns
        -------
        np.ndarray
            Beyaz gürültü: ε(t) ~ N(0, σ²)
        """
        noise = np.random.normal(0, self.noise_std, self.n_samples)
        return noise
    
    def _generate_shocks(
        self,
        n_shocks: int = 3,
        shock_std: float = 20.0,
        decay_rate: float = 0.1,
        shock_positions: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Kontrollü şoklar ve sönümleme etkisi oluşturur.
        
        Parameters
        ----------
        n_shocks : int, optional
            Şok sayısı (varsayılan: 3)
        shock_std : float, optional
            Şok büyüklüğü std sapması (varsayılan: 20.0)
        decay_rate : float, optional
            Sönümleme oranı τ (varsayılan: 0.1)
        shock_positions : List[int], optional
            Şok pozisyonları, None ise rastgele (varsayılan: None)
        
        Returns
        -------
        Tuple[np.ndarray, List[int]]
            Şok serisi ve şok pozisyonları listesi
        """
        shock_series = np.zeros(self.n_samples)
        
        # Şok pozisyonlarını belirle
        if shock_positions is None:
            # İlk ve son %10'luk alanı hariç tut
            min_pos = int(self.n_samples * 0.1)
            max_pos = int(self.n_samples * 0.9)
            shock_positions = np.random.choice(
                range(min_pos, max_pos),
                size=n_shocks,
                replace=False
            )
            shock_positions = sorted(shock_positions)
        
        # Her şok için etki oluştur
        for shock_pos in shock_positions:
            # Şok büyüklüğü
            shock_magnitude = np.random.normal(0, shock_std)
            
            # Şok etkisi: Γ(t) = M * exp(-(t-t_shock)/τ) for t > t_shock
            for t in range(shock_pos, self.n_samples):
                time_since_shock = t - shock_pos
                decay = np.exp(-time_since_shock * decay_rate)
                shock_series[t] += shock_magnitude * decay
        
        return shock_series, list(shock_positions)
    
    def generate(
        self,
        n_shocks: int = 3,
        shock_std: float = 20.0,
        decay_rate: float = 0.1,
        shock_positions: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Tam sentetik zaman serisi oluşturur.
        
        Parameters
        ----------
        n_shocks : int, optional
            Şok sayısı (varsayılan: 3)
        shock_std : float, optional
            Şok büyüklüğü std sapması (varsayılan: 20.0)
        decay_rate : float, optional
            Sönümleme oranı (varsayılan: 0.1)
        shock_positions : List[int], optional
            Şok pozisyonları (varsayılan: None)
        
        Returns
        -------
        Tuple[pd.DataFrame, dict]
            DataFrame: zaman serisi ve bileşenleri
            dict: metadata (şok pozisyonları, parametreler)
        """
        # Bileşenleri oluştur
        trend = self._generate_trend()
        seasonal = self._generate_seasonal()
        noise = self._generate_noise()
        shock_series, actual_shock_positions = self._generate_shocks(
            n_shocks, shock_std, decay_rate, shock_positions
        )
        
        # Baseline (şoksuz) seri
        baseline = trend + seasonal + noise
        
        # Nihai seri (şoklu)
        y = baseline + shock_series
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'time': np.arange(self.n_samples),
            'y': y,
            'baseline': baseline,
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'shocks': shock_series
        })
        
        # Metadata
        metadata = {
            'n_samples': self.n_samples,
            'n_shocks': n_shocks,
            'shock_positions': actual_shock_positions,
            'shock_std': shock_std,
            'decay_rate': decay_rate,
            'parameters': {
                'trend_coef': self.trend_coef,
                'trend_intercept': self.trend_intercept,
                'seasonal_amplitude': self.seasonal_amplitude,
                'seasonal_period': self.seasonal_period,
                'noise_std': self.noise_std
            }
        }
        
        return df, metadata

