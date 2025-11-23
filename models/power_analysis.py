"""
İstatistiksel güç analizi modülü.

Bu modül, hypothesis testleri için gerekli sample size hesaplama
ve mevcut sample size için statistical power tahminleme araçları sağlar.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


class StatisticalPowerAnalyzer:
    """
    İstatistiksel güç analizi ve optimal sample size hesaplayıcı.
    
    Bu sınıf, Diebold-Mariano ve diğer hypothesis testleri için
    power analysis yapar ve optimal sample size önerir.
    
    Attributes
    ----------
    alpha : float
        Type I error rate (significance level)
    beta : float
        Type II error rate
    power : float
        Statistical power (1 - beta)
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """
        StatisticalPowerAnalyzer başlatıcı.
        
        Parameters
        ----------
        alpha : float, optional
            Type I error rate (varsayılan: 0.05)
        power : float, optional
            Hedef statistical power (varsayılan: 0.80)
        """
        self.alpha = alpha
        self.power = power
        self.beta = 1 - power
    
    def compute_required_sample_size(
        self,
        delta: float,
        sigma: float,
        test_type: str = 'two-sided'
    ) -> int:
        """
        Diebold-Mariano test için gerekli sample size.
        
        Formula:
        n = ((z_α/2 + z_β) · σ / δ)²
        
        Parameters
        ----------
        delta : float
            Hedef effect size (performans farkı)
        sigma : float
            Loss differential'ın standard deviation
        test_type : str, optional
            Test tipi: 'two-sided', 'one-sided' (varsayılan: 'two-sided')
            
        Returns
        -------
        int
            Gerekli minimum sample size
            
        Examples
        --------
        >>> analyzer = StatisticalPowerAnalyzer(alpha=0.05, power=0.80)
        >>> n = analyzer.compute_required_sample_size(delta=0.001, sigma=0.025)
        >>> print(f"Gerekli sample size: {n}")
        """
        if test_type == 'two-sided':
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        
        z_beta = stats.norm.ppf(self.power)
        
        if delta == 0:
            return np.inf
        
        n = ((z_alpha + z_beta) * sigma / delta) ** 2
        
        return int(np.ceil(n))
    
    def estimate_power(
        self,
        n: int,
        delta: float,
        sigma: float,
        test_type: str = 'two-sided'
    ) -> float:
        """
        Verilen sample size için test gücünü hesapla.
        
        Power = Φ(δ/(σ/√n) - z_α/2)
        
        Parameters
        ----------
        n : int
            Sample size
        delta : float
            Effect size
        sigma : float
            Standard deviation
        test_type : str, optional
            Test tipi (varsayılan: 'two-sided')
            
        Returns
        -------
        float
            Statistical power (0 ile 1 arası)
            
        Examples
        --------
        >>> analyzer = StatisticalPowerAnalyzer()
        >>> power = analyzer.estimate_power(n=110, delta=0.000041, sigma=0.025)
        >>> print(f"Current power: {power:.4f}")
        """
        if test_type == 'two-sided':
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        
        if n <= 0 or sigma == 0:
            return 0.0
        
        test_statistic = delta / (sigma / np.sqrt(n))
        power = stats.norm.cdf(test_statistic - z_alpha)
        
        return max(0.0, min(1.0, power))
    
    def power_analysis_report(
        self,
        n_current: int,
        delta_observed: float,
        sigma_observed: float,
        delta_target: float = None
    ) -> Dict[str, float]:
        """
        Kapsamlı power analysis raporu.
        
        Parameters
        ----------
        n_current : int
            Mevcut sample size
        delta_observed : float
            Gözlemlenen effect size
        sigma_observed : float
            Gözlemlenen standard deviation
        delta_target : float, optional
            Hedef effect size (None ise observed kullanılır)
            
        Returns
        -------
        Dict[str, float]
            Power analysis sonuçları
        """
        if delta_target is None:
            delta_target = delta_observed
        
        # Mevcut power
        current_power = self.estimate_power(
            n_current, delta_observed, sigma_observed
        )
        
        # Hedef delta için gereken n
        required_n = self.compute_required_sample_size(
            delta_target, sigma_observed
        )
        
        # Effect size iyileşmesi gerekli mi?
        if required_n > 10000:  # Pratik olmayan
            # Mevcut n ile ulaşılabilir minimum delta
            min_detectable_delta = self._compute_minimum_detectable_effect(
                n_current, sigma_observed
            )
        else:
            min_detectable_delta = delta_target
        
        return {
            'current_n': n_current,
            'current_power': current_power,
            'observed_delta': delta_observed,
            'observed_sigma': sigma_observed,
            'target_delta': delta_target,
            'required_n_for_target': int(required_n) if required_n != np.inf else None,
            'min_detectable_delta': min_detectable_delta,
            'is_adequately_powered': current_power >= self.power
        }
    
    def _compute_minimum_detectable_effect(
        self,
        n: int,
        sigma: float
    ) -> float:
        """
        Verilen n ve power ile detect edilebilir minimum effect size.
        
        Parameters
        ----------
        n : int
            Sample size
        sigma : float
            Standard deviation
            
        Returns
        -------
        float
            Minimum detectable effect size
        """
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)
        
        mde = (z_alpha + z_beta) * sigma / np.sqrt(n)
        
        return mde
    
    def sample_size_curve(
        self,
        delta_range: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        Farklı effect size değerleri için sample size curve.
        
        Parameters
        ----------
        delta_range : np.ndarray
            Effect size değerleri
        sigma : float
            Standard deviation
            
        Returns
        -------
        np.ndarray
            Her delta için gerekli sample size
        """
        n_values = []
        
        for delta in delta_range:
            if delta > 0:
                n = self.compute_required_sample_size(delta, sigma)
                n_values.append(min(n, 100000))  # Cap at reasonable value
            else:
                n_values.append(np.inf)
        
        return np.array(n_values)
    
    @staticmethod
    def interpret_power(power: float) -> str:
        """
        Power değerini yorumla.
        
        Parameters
        ----------
        power : float
            Statistical power
            
        Returns
        -------
        str
            Yorumlama metni
        """
        if power >= 0.80:
            return f"✅ Yeterli güç (power={power:.2%})"
        elif power >= 0.50:
            return f"⚠️  Orta düzeyde güç (power={power:.2%})"
        else:
            return f"❌ Yetersiz güç (power={power:.2%})"
    
    @staticmethod
    def compare_scenarios(
        scenarios: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Farklı senaryoları karşılaştır ve rapor oluştur.
        
        Parameters
        ----------
        scenarios : Dict[str, Dict[str, float]]
            Her senaryo için {'n': ..., 'delta': ..., 'sigma': ...}
            
        Returns
        -------
        str
            Karşılaştırma raporu
        """
        analyzer = StatisticalPowerAnalyzer()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("POWER ANALYSIS - SENARYO KARŞILAŞTIRMASI")
        report_lines.append("=" * 80)
        
        for name, params in scenarios.items():
            power = analyzer.estimate_power(
                params['n'], params['delta'], params['sigma']
            )
            
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Sample size: {params['n']}")
            report_lines.append(f"  Effect size: {params['delta']:.6f}")
            report_lines.append(f"  Std deviation: {params['sigma']:.6f}")
            report_lines.append(f"  Power: {power:.4f} ({power*100:.2f}%)")
            report_lines.append(f"  {analyzer.interpret_power(power)}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)


def quick_power_check(
    n: int,
    rmse_baseline: float,
    rmse_model: float,
    sigma: float = None,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Hızlı power check için utility fonksiyon.
    
    Parameters
    ----------
    n : int
        Sample size
    rmse_baseline : float
        Baseline RMSE
    rmse_model : float
        Model RMSE
    sigma : float, optional
        Std deviation (None ise RMSE'den tahmin edilir)
    alpha : float, optional
        Significance level
        
    Returns
    -------
    Dict[str, any]
        Power analysis sonuçları
    """
    delta = abs(rmse_baseline - rmse_model)
    
    if sigma is None:
        # Conservative estimate
        sigma = (rmse_baseline + rmse_model) / 2
    
    analyzer = StatisticalPowerAnalyzer(alpha=alpha)
    
    current_power = analyzer.estimate_power(n, delta, sigma)
    required_n = analyzer.compute_required_sample_size(delta, sigma)
    
    return {
        'n': n,
        'delta': delta,
        'sigma': sigma,
        'current_power': current_power,
        'required_n': int(required_n) if required_n != np.inf else None,
        'interpretation': analyzer.interpret_power(current_power),
        'is_significant_possible': current_power >= 0.80
    }

