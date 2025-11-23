"""
Model performans değerlendirme ve metrik hesaplama modülü.

Bu modül, baseline ve GRM modellerini karşılaştırmak için
çeşitli metrikler ve istatistiksel testler sağlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


class ModelEvaluator:
    """
    Model değerlendirme sınıfı.
    
    Bu sınıf, tahmin performansını ölçmek için çeşitli metrikler
    ve istatistiksel testler içerir.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error hesaplar.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        
        Returns
        -------
        float
            RMSE değeri
        """
        # NaN ve inf değerlerini temizle
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                 np.isinf(y_true) | np.isinf(y_pred))
        
        if not np.any(mask):
            return np.nan
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        return np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error hesaplar.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        
        Returns
        -------
        float
            MAE değeri
        """
        # NaN ve inf değerlerini temizle
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                 np.isinf(y_true) | np.isinf(y_pred))
        
        if not np.any(mask):
            return np.nan
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        return np.mean(np.abs(y_true_clean - y_pred_clean))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error hesaplar.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        
        Returns
        -------
        float
            MAPE değeri (yüzde olarak)
        """
        # Sıfır değerleri kontrol et
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R² (coefficient of determination) hesaplar.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        
        Returns
        -------
        float
            R² değeri
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def diebold_mariano_test(
        errors1: np.ndarray,
        errors2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano testi ile iki modelin tahmin performansını karşılaştırır.
        
        H0: İki modelin tahmin performansı eşittir
        H1: Performanslar farklıdır
        
        Parameters
        ----------
        errors1 : np.ndarray
            Model 1 hataları
        errors2 : np.ndarray
            Model 2 hataları
        alternative : str, optional
            Test tipi: 'two-sided', 'less', 'greater' (varsayılan: 'two-sided')
        
        Returns
        -------
        Tuple[float, float]
            DM test istatistiği ve p-değeri
        """
        # Hata farklarının kareleri
        d = errors1 ** 2 - errors2 ** 2
        
        # Ortalama fark
        d_mean = np.mean(d)
        
        # Varyans hesabı (Newey-West düzeltmesi olmadan basit versiyon)
        d_var = np.var(d, ddof=1)
        
        # Test istatistiği
        n = len(d)
        dm_stat = d_mean / np.sqrt(d_var / n)
        
        # P-değeri
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        elif alternative == 'less':
            p_value = stats.norm.cdf(dm_stat)
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(dm_stat)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        
        return dm_stat, p_value
    
    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Bir model için tüm metrikleri hesaplar.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        model_name : str, optional
            Model ismi (varsayılan: "Model")
        
        Returns
        -------
        Dict[str, float]
            Tüm performans metrikleri
        """
        metrics = {
            'model_name': model_name,
            'rmse': ModelEvaluator.rmse(y_true, y_pred),
            'mae': ModelEvaluator.mae(y_true, y_pred),
            'mape': ModelEvaluator.mape(y_true, y_pred),
            'r2': ModelEvaluator.r2_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def compare_models(
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_grm: np.ndarray
    ) -> Dict[str, any]:
        """
        Baseline ve GRM modellerini karşılaştırır.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred_baseline : np.ndarray
            Baseline model tahminleri
        y_pred_grm : np.ndarray
            GRM model tahminleri
        
        Returns
        -------
        Dict[str, any]
            Karşılaştırma sonuçları ve istatistiksel testler
        """
        # Her model için metrikler
        baseline_metrics = ModelEvaluator.evaluate_model(
            y_true, y_pred_baseline, "Baseline"
        )
        grm_metrics = ModelEvaluator.evaluate_model(
            y_true, y_pred_grm, "GRM"
        )
        
        # Hataları hesapla
        errors_baseline = y_true - y_pred_baseline
        errors_grm = y_true - y_pred_grm
        
        # Diebold-Mariano testi
        dm_stat, dm_pvalue = ModelEvaluator.diebold_mariano_test(
            errors_baseline, errors_grm
        )
        
        # İyileşme yüzdesi
        rmse_improvement = (
            (baseline_metrics['rmse'] - grm_metrics['rmse']) /
            baseline_metrics['rmse'] * 100
        )
        
        mae_improvement = (
            (baseline_metrics['mae'] - grm_metrics['mae']) /
            baseline_metrics['mae'] * 100
        )
        
        # Karşılaştırma sonuçları
        comparison = {
            'baseline_metrics': baseline_metrics,
            'grm_metrics': grm_metrics,
            'diebold_mariano_stat': dm_stat,
            'diebold_mariano_pvalue': dm_pvalue,
            'rmse_improvement_pct': rmse_improvement,
            'mae_improvement_pct': mae_improvement,
            'grm_is_better': dm_pvalue < 0.05 and rmse_improvement > 0
        }
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict[str, any]):
        """
        Karşılaştırma sonuçlarını formatlı olarak yazdırır.
        
        Parameters
        ----------
        comparison : Dict[str, any]
            compare_models() fonksiyonundan dönen sonuçlar
        """
        print("\n" + "=" * 70)
        print("MODEL KARŞILAŞTIRMA SONUÇLARI")
        print("=" * 70)
        
        # Baseline metrikleri
        baseline = comparison['baseline_metrics']
        print(f"\n📊 BASELINE MODEL:")
        print(f"   RMSE  : {baseline['rmse']:.4f}")
        print(f"   MAE   : {baseline['mae']:.4f}")
        print(f"   MAPE  : {baseline['mape']:.2f}%")
        print(f"   R²    : {baseline['r2']:.4f}")
        
        # GRM metrikleri
        grm = comparison['grm_metrics']
        print(f"\n🌀 GRM MODEL:")
        print(f"   RMSE  : {grm['rmse']:.4f}")
        print(f"   MAE   : {grm['mae']:.4f}")
        print(f"   MAPE  : {grm['mape']:.2f}%")
        print(f"   R²    : {grm['r2']:.4f}")
        
        # İyileşme
        print(f"\n📈 İYİLEŞME:")
        print(f"   RMSE  : {comparison['rmse_improvement_pct']:+.2f}%")
        print(f"   MAE   : {comparison['mae_improvement_pct']:+.2f}%")
        
        # İstatistiksel test
        print(f"\n📊 DIEBOLD-MARIANO TESTİ:")
        print(f"   İstatistik: {comparison['diebold_mariano_stat']:.4f}")
        print(f"   P-değeri  : {comparison['diebold_mariano_pvalue']:.4f}")
        
        # Sonuç
        print(f"\n{'🎯 SONUÇ:'}")
        if comparison['grm_is_better']:
            print("   ✓ GRM, baseline modele göre İSTATİSTİKSEL OLARAK ANLAMLI")
            print("     şekilde daha iyi performans gösterdi (p < 0.05)")
        else:
            print("   ✗ GRM ve baseline model arasında istatistiksel olarak")
            print("     anlamlı bir fark bulunamadı (p >= 0.05)")
        
        print("=" * 70 + "\n")


# Convenience functions for backward compatibility
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error hesaplar (convenience function).
    
    Parameters
    ----------
    y_true : np.ndarray
        Gerçek değerler
    y_pred : np.ndarray
        Tahmin değerleri
        
    Returns
    -------
    float
        RMSE değeri
    """
    return ModelEvaluator.rmse(y_true, y_pred)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error hesaplar (convenience function).
    
    Parameters
    ----------
    y_true : np.ndarray
        Gerçek değerler
    y_pred : np.ndarray
        Tahmin değerleri
        
    Returns
    -------
    float
        MAE değeri
    """
    return ModelEvaluator.mae(y_true, y_pred)

