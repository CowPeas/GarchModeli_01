"""
Kapsamlı istatistiksel test modülü.

Bu modül, model performanslarını istatistiksel olarak karşılaştırmak
ve artıkların özelliklerini test etmek için kullanılır.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


class StatisticalTests:
    """
    Zaman serisi modelleri için kapsamlı istatistiksel testler.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Parameters
        ----------
        significance_level : float
            Anlamlılık seviyesi (varsayılan: 0.05)
        """
        self.alpha = significance_level
    
    @staticmethod
    def diebold_mariano_test(
        errors1: np.ndarray,
        errors2: np.ndarray,
        alternative: str = 'two-sided',
        h: int = 1
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano testi - İki modelin tahmin performansını karşılaştırır.
        
        Parameters
        ----------
        errors1 : np.ndarray
            Model 1 tahmin hataları
        errors2 : np.ndarray
            Model 2 tahmin hataları
        alternative : str
            'two-sided', 'less', 'greater'
        h : int
            Tahmin ufku (varsayılan: 1)
            
        Returns
        -------
        Dict[str, float]
            Test istatistiği, p-value, sonuç
        """
        # Loss differential
        d = errors1**2 - errors2**2
        
        # Mean loss differential
        d_mean = np.mean(d)
        
        # Variance (Newey-West düzeltmesi ile)
        n = len(d)
        
        # HAC variance estimation (simplified)
        gamma_0 = np.var(d, ddof=1)
        
        # Harvey, Leybourne, Newbold (1997) düzeltmesi
        dm_stat = d_mean / np.sqrt(gamma_0 / n)
        
        # P-value hesaplama
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        elif alternative == 'less':
            p_value = stats.norm.cdf(dm_stat)
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(dm_stat)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        
        return float(dm_stat), float(p_value)
    
    def _interpret_dm_test(
        self,
        dm_stat: float,
        p_value: float,
        alternative: str
    ) -> str:
        """DM test sonucunu yorumla."""
        if alternative == 'two-sided':
            if p_value < self.alpha:
                if dm_stat > 0:
                    return f"Model 2 istatistiksel olarak ANLAMLI şekilde DAHA İYİ (p={p_value:.4f} < {self.alpha})"
                else:
                    return f"Model 1 istatistiksel olarak ANLAMLI şekilde DAHA İYİ (p={p_value:.4f} < {self.alpha})"
            else:
                return f"İki model arasında istatistiksel olarak ANLAMLI fark YOK (p={p_value:.4f} >= {self.alpha})"
        return "Test sonucu belirsiz"
    
    @staticmethod
    def arch_lm_test(
        residuals: np.ndarray,
        lags: int = 5
    ) -> Tuple[float, float]:
        """
        ARCH-LM testi - Artıklarda ARCH etkisi (heteroskedastisite) var mı?
        
        H0: Artıklarda heteroskedastisite yok
        H1: ARCH etkisi var (heteroskedastisite var)
        
        Parameters
        ----------
        residuals : np.ndarray
            Model artıkları
        lags : int
            Test için kullanılacak gecikme sayısı
            
        Returns
        -------
        Dict[str, float]
            Test istatistiği, p-value, sonuç
        """
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=lags)
        return float(lm_stat), float(lm_pvalue)
    
    def _interpret_arch_test(self, p_value: float, is_significant: bool) -> str:
        """ARCH test sonucunu yorumla."""
        if is_significant:
            return f"⚠️ Artıklarda ARCH etkisi VAR (heteroskedastisite) - p={p_value:.4f} < {self.alpha}. Model volatiliteyi tam yakalayamadı."
        else:
            return f"✓ Artıklarda ARCH etkisi YOK - p={p_value:.4f} >= {self.alpha}. Model volatiliteyi iyi modelledi."
    
    @staticmethod
    def ljung_box_test(
        residuals: np.ndarray,
        lags: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ljung-Box testi - Artıklarda otokorelasyon var mı?
        
        H0: Artıklarda otokorelasyon yok (white noise)
        H1: Otokorelasyon var
        
        Parameters
        ----------
        residuals : np.ndarray
            Model artıkları
        lags : int
            Test edilecek maksimum gecikme
            
        Returns
        -------
        Dict[str, float]
            Test istatistikleri, p-values
        """
        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        return lb_result['lb_stat'].values, lb_result['lb_pvalue'].values
    
    def _interpret_ljung_box(
        self,
        any_significant: bool,
        first_significant: int
    ) -> str:
        """Ljung-Box test sonucunu yorumla."""
        if any_significant:
            return f"⚠️ Artıklarda OTOKORELASYON VAR (lag {first_significant}). Model tüm yapıyı yakalayamadı."
        else:
            return f"✓ Artıklarda otokorelasyon YOK. Model yapıyı iyi yakaladı."
    
    def stationarity_test(
        self,
        series: np.ndarray,
        regression: str = 'c'
    ) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller (ADF) durağanlık testi.
        
        H0: Seri durağan değil (birim kök var)
        H1: Seri durağan
        
        Parameters
        ----------
        series : np.ndarray
            Test edilecek seri
        regression : str
            'c' (sabit), 'ct' (sabit+trend), 'ctt' (sabit+trend^2), 'n' (yok)
            
        Returns
        -------
        Dict[str, float]
            Test istatistikleri
        """
        try:
            adf_stat, p_value, usedlag, nobs, critical_values, icbest = adfuller(
                series, regression=regression
            )
            
            is_stationary = p_value < self.alpha
            
            return {
                'adf_statistic': float(adf_stat),
                'p_value': float(p_value),
                'used_lag': int(usedlag),
                'n_obs': int(nobs),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'is_stationary': bool(is_stationary),
                'interpretation': self._interpret_adf(is_stationary, p_value)
            }
        except Exception as e:
            return {
                'error': str(e),
                'interpretation': 'Test başarısız'
            }
    
    def _interpret_adf(self, is_stationary: bool, p_value: float) -> str:
        """ADF test sonucunu yorumla."""
        if is_stationary:
            return f"✓ Seri DURAĞAN - p={p_value:.4f} < {self.alpha}"
        else:
            return f"⚠️ Seri DURAĞAN DEĞİL - p={p_value:.4f} >= {self.alpha}. Fark alma gerekebilir."
    
    def normality_test(
        self,
        residuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Jarque-Bera normallik testi.
        
        H0: Artıklar normal dağılımlı
        H1: Artıklar normal dağılımlı değil
        
        Parameters
        ----------
        residuals : np.ndarray
            Model artıkları
            
        Returns
        -------
        Dict[str, float]
            Test istatistikleri
        """
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            
            is_normal = jb_pvalue >= self.alpha
            
            # Çarpıklık ve basıklık
            skewness = float(stats.skew(residuals))
            kurtosis = float(stats.kurtosis(residuals))
            
            return {
                'jb_statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': bool(is_normal),
                'interpretation': self._interpret_normality(is_normal, jb_pvalue, skewness, kurtosis)
            }
        except Exception as e:
            return {
                'error': str(e),
                'interpretation': 'Test başarısız'
            }
    
    def _interpret_normality(
        self,
        is_normal: bool,
        p_value: float,
        skewness: float,
        kurtosis: float
    ) -> str:
        """Normallik testi sonucunu yorumla."""
        if is_normal:
            return f"✓ Artıklar NORMAL dağılımlı - p={p_value:.4f} >= {self.alpha}"
        else:
            issues = []
            if abs(skewness) > 0.5:
                issues.append(f"çarpık (skew={skewness:.2f})")
            if abs(kurtosis) > 3:
                issues.append(f"kalın kuyruklu (kurt={kurtosis:.2f})")
            
            issue_str = ", ".join(issues) if issues else "anormal"
            return f"⚠️ Artıklar NORMAL DEĞİL ({issue_str}) - p={p_value:.4f} < {self.alpha}"
    
    def comprehensive_diagnostic(
        self,
        residuals: np.ndarray,
        lags: int = 10
    ) -> Dict[str, Dict]:
        """
        Artıklar için kapsamlı tanısal testler.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model artıkları
        lags : int
            Test gecikmeleri
            
        Returns
        -------
        Dict[str, Dict]
            Tüm test sonuçları
        """
        results = {
            'arch_lm': self.arch_lm_test(residuals, lags=lags),
            'ljung_box': self.ljung_box_test(residuals, lags=lags),
            'stationarity': self.stationarity_test(residuals),
            'normality': self.normality_test(residuals)
        }
        
        # Genel skor (passed testler)
        passed = sum([
            not results['arch_lm'].get('is_significant', False),
            not results['ljung_box'].get('any_significant', False),
            results['stationarity'].get('is_stationary', False),
            results['normality'].get('is_normal', False)
        ])
        
        results['overall'] = {
            'tests_passed': passed,
            'total_tests': 4,
            'pass_rate': passed / 4,
            'summary': self._summarize_diagnostics(passed)
        }
        
        return results
    
    def _summarize_diagnostics(self, passed: int) -> str:
        """Tanısal testlerin özetini hazırla."""
        if passed == 4:
            return "✅ MÜKEMMEL - Tüm tanısal testler başarılı"
        elif passed >= 3:
            return "✓ İYİ - Çoğu tanısal test başarılı"
        elif passed >= 2:
            return "⚠️ ORTA - Bazı sorunlar var"
        else:
            return "❌ ZAYIF - Ciddi tanısal sorunlar var"
    
    def compare_models_comprehensive(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        baseline_name: str = 'baseline'
    ) -> pd.DataFrame:
        """
        Birden fazla modeli kapsamlı şekilde karşılaştır.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        predictions_dict : Dict[str, np.ndarray]
            Model isimleri ve tahminleri
        baseline_name : str
            Baseline model ismi
            
        Returns
        -------
        pd.DataFrame
            Karşılaştırma tablosu
        """
        results = []
        
        baseline_errors = y_true - predictions_dict[baseline_name]
        
        for model_name, predictions in predictions_dict.items():
            if model_name == baseline_name:
                continue
            
            model_errors = y_true - predictions
            
            # DM test
            dm_result = self.diebold_mariano_test(baseline_errors, model_errors)
            
            # RMSE
            rmse_baseline = np.sqrt(np.mean(baseline_errors**2))
            rmse_model = np.sqrt(np.mean(model_errors**2))
            improvement = (rmse_baseline - rmse_model) / rmse_baseline * 100
            
            results.append({
                'Model': model_name,
                'RMSE': rmse_model,
                'vs_Baseline_Improvement_%': improvement,
                'DM_Statistic': dm_result['statistic'],
                'DM_p_value': dm_result['p_value'],
                'Is_Significant': 'Evet' if dm_result['is_significant'] else 'Hayır',
                'Interpretation': dm_result['interpretation']
            })
        
        return pd.DataFrame(results)


def print_test_results(results: Dict, title: str = "Test Sonuçları"):
    """
    Test sonuçlarını formatlanmış şekilde yazdır.
    
    Parameters
    ----------
    results : Dict
        Test sonuçları
    title : str
        Başlık
    """
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    for test_name, test_result in results.items():
        if test_name == 'overall':
            continue
        
        print(f"\n🔬 {test_name.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        if 'interpretation' in test_result:
            print(test_result['interpretation'])
        
        # Detayları göster
        for key, value in test_result.items():
            if key not in ['interpretation', 'statistics', 'p_values']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                elif isinstance(value, bool):
                    print(f"  {key}: {'Evet' if value else 'Hayır'}")
                elif value is not None:
                    print(f"  {key}: {value}")
    
    # Overall summary
    if 'overall' in results:
        print("\n" + "=" * 80)
        print(f"📈 GENEL DEĞERLENDIRME")
        print("=" * 80)
        overall = results['overall']
        print(f"  Başarılı Testler: {overall['tests_passed']}/{overall['total_tests']}")
        print(f"  Başarı Oranı: {overall['pass_rate']*100:.1f}%")
        print(f"  {overall['summary']}")
        print("=" * 80)

