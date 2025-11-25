"""
Gelişmiş performans metrikleri ve bootstrap analizi.

Bu modül, model performansını çok boyutlu değerlendirmek için
finansal metrikler, yön doğruluğu ve bootstrap güven aralıkları sağlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats


class AdvancedMetrics:
    """
    Gelişmiş performans metrikleri hesaplayıcı.
    
    Bu sınıf, zaman serisi tahmin modelleri için kapsamlı
    değerlendirme metrikleri sağlar.
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_series: bool = False
    ) -> Dict[str, float]:
        """
        Tüm metrikleri hesapla.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahmin değerleri
        return_series : bool, optional
            Getiri serisi mi? (varsayılan: False)
            
        Returns
        -------
        Dict[str, float]
            Tüm metrikler
        """
        # Temizlik
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                 np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        metrics = {}
        
        # Temel hata metrikleri
        metrics['rmse'] = AdvancedMetrics.rmse(y_true_clean, y_pred_clean)
        metrics['mae'] = AdvancedMetrics.mae(y_true_clean, y_pred_clean)
        metrics['mape'] = AdvancedMetrics.mape(y_true_clean, y_pred_clean)
        metrics['r2'] = AdvancedMetrics.r2_score(y_true_clean, y_pred_clean)
        
        # Yön doğruluğu
        metrics['mda'] = AdvancedMetrics.mean_directional_accuracy(
            y_true_clean, y_pred_clean
        )
        metrics['hit_ratio'] = AdvancedMetrics.hit_ratio(y_true_clean, y_pred_clean)
        
        # Getiri serisi ise ek metrikler
        if return_series:
            financial_metrics = AdvancedMetrics.financial_metrics(
                y_true_clean, y_pred_clean
            )
            metrics.update(financial_metrics)
        
        return metrics
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        
        Sıfır değerlerden kaçınmak için epsilon kullanır.
        """
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² (Coefficient of Determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mean_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Mean Directional Accuracy (MDA).
        
        Yön tahmini doğruluğu (yukarı/aşağı).
        """
        if len(y_true) < 2:
            return 0.0
        
        # Yön hesapla
        actual_direction = np.sign(y_true[1:] - y_true[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        
        # Doğru tahmin oranı
        correct = np.sum(actual_direction == pred_direction)
        total = len(actual_direction)
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Hit Ratio.
        
        Tahmin hatalarının gerçek değerlerle aynı yönde olma oranı.
        """
        # Hata
        errors = y_pred - y_true
        
        # Gerçek değerlerin işareti
        sign_true = np.sign(y_true)
        
        # Hatanın tersi yönünde olmalı (doğru tahmin)
        correct_sign = np.sign(errors) == -sign_true
        
        return np.mean(correct_sign)
    
    @staticmethod
    def financial_metrics(
        returns_true: np.ndarray,
        returns_pred: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Finansal performans metrikleri.
        
        Parameters
        ----------
        returns_true : np.ndarray
            Gerçek getiriler
        returns_pred : np.ndarray
            Tahmin edilen getiriler
        risk_free_rate : float, optional
            Risksiz faiz oranı (yıllık) (varsayılan: 0.02)
            
        Returns
        -------
        Dict[str, float]
            Finansal metrikler
        """
        metrics = {}
        
        # Trading strategy: Tahmin pozitifse long, negatifse short
        strategy_returns = returns_true * np.sign(returns_pred)
        
        # Kümülatif getiri
        cumulative_return = np.prod(1 + strategy_returns) - 1
        metrics['cumulative_return'] = cumulative_return
        
        # Sharpe Ratio
        excess_returns = strategy_returns - risk_free_rate / 252  # Günlük
        sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        metrics['sharpe_ratio'] = sharpe * np.sqrt(252)  # Yıllıklaştır
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Win Rate
        metrics['win_rate'] = np.mean(strategy_returns > 0)
        
        # Profit Factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['profit_factor'] = gains / losses if losses > 0 else np.inf
        
        return metrics
    
    @staticmethod
    def volatility_forecast_accuracy(
        returns_true: np.ndarray,
        volatility_pred: np.ndarray,
        window: int = 5
    ) -> Dict[str, float]:
        """
        Volatilite tahmin doğruluğu.
        
        Parameters
        ----------
        returns_true : np.ndarray
            Gerçek getiriler
        volatility_pred : np.ndarray
            Tahmin edilen volatilite
        window : int, optional
            Realized volatility için window (varsayılan: 5)
            
        Returns
        -------
        Dict[str, float]
            Volatilite metrikleri
        """
        # Realized volatility (rolling std)
        realized_vol = pd.Series(returns_true).rolling(window).std().values
        
        # Align arrays
        min_len = min(len(realized_vol), len(volatility_pred))
        realized_vol = realized_vol[-min_len:]
        volatility_pred = volatility_pred[-min_len:]
        
        # Temizlik
        mask = ~np.isnan(realized_vol)
        realized_vol = realized_vol[mask]
        volatility_pred = volatility_pred[mask]
        
        if len(realized_vol) == 0:
            return {'vol_rmse': np.nan, 'vol_mae': np.nan, 'vol_r2': np.nan}
        
        return {
            'vol_rmse': np.sqrt(np.mean((realized_vol - volatility_pred) ** 2)),
            'vol_mae': np.mean(np.abs(realized_vol - volatility_pred)),
            'vol_r2': AdvancedMetrics.r2_score(realized_vol, volatility_pred)
        }


class BootstrapCI:
    """
    Bootstrap güven aralığı hesaplayıcı.
    
    Model performans farklarının istatistiksel anlamlılığını
    bootstrap yöntemiyle test eder.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Bootstrap parametreleri.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Bootstrap iterasyon sayısı (varsayılan: 1000)
        confidence_level : float, optional
            Güven seviyesi (varsayılan: 0.95)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compute_rmse_ci(
        self,
        errors1: np.ndarray,
        errors2: np.ndarray
    ) -> tuple:
        """
        İki modelin RMSE farkı için bootstrap CI hesapla.
        
        Parameters
        ----------
        errors1 : np.ndarray
            Model 1 hataları (baseline)
        errors2 : np.ndarray
            Model 2 hataları (proposed)
            
        Returns
        -------
        tuple
            (ci_lower, ci_upper) - RMSE farkının CI'sı
        """
        n = len(errors1)
        rmse_diffs = []
        
        np.random.seed(42)
        
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            
            e1_boot = errors1[idx]
            e2_boot = errors2[idx]
            
            rmse1 = np.sqrt(np.mean(e1_boot ** 2))
            rmse2 = np.sqrt(np.mean(e2_boot ** 2))
            
            rmse_diffs.append(rmse1 - rmse2)
        
        rmse_diffs = np.array(rmse_diffs)
        
        ci_lower = np.percentile(rmse_diffs, self.alpha / 2 * 100)
        ci_upper = np.percentile(rmse_diffs, (1 - self.alpha / 2) * 100)
        
        return float(ci_lower), float(ci_upper)
    
    def performance_difference_ci(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        metric: str = 'rmse'
    ) -> Dict[str, float]:
        """
        İki model arasındaki performans farkının CI'sını hesapla.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred1 : np.ndarray
            Model 1 tahminleri
        y_pred2 : np.ndarray
            Model 2 tahminleri
        metric : str, optional
            Metrik: 'rmse', 'mae', 'r2' (varsayılan: 'rmse')
            
        Returns
        -------
        Dict[str, float]
            Ortalama fark, CI, anlamlılık
        """
        n = len(y_true)
        differences = []
        
        np.random.seed(42)  # Reproducibility
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            
            y_true_boot = y_true[idx]
            y_pred1_boot = y_pred1[idx]
            y_pred2_boot = y_pred2[idx]
            
            # Metrik hesapla
            if metric == 'rmse':
                perf1 = np.sqrt(np.mean((y_true_boot - y_pred1_boot) ** 2))
                perf2 = np.sqrt(np.mean((y_true_boot - y_pred2_boot) ** 2))
            elif metric == 'mae':
                perf1 = np.mean(np.abs(y_true_boot - y_pred1_boot))
                perf2 = np.mean(np.abs(y_true_boot - y_pred2_boot))
            elif metric == 'r2':
                perf1 = AdvancedMetrics.r2_score(y_true_boot, y_pred1_boot)
                perf2 = AdvancedMetrics.r2_score(y_true_boot, y_pred2_boot)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Fark (model1 - model2, negatif = model2 daha iyi)
            differences.append(perf1 - perf2)
        
        differences = np.array(differences)
        
        # CI hesapla
        ci_lower = np.percentile(differences, self.alpha / 2 * 100)
        ci_upper = np.percentile(differences, (1 - self.alpha / 2) * 100)
        
        # Anlamlılık: CI sıfırı içermiyor mu?
        is_significant = not (ci_lower < 0 < ci_upper)
        
        # Yorum
        mean_diff = np.mean(differences)
        if is_significant:
            if mean_diff > 0:
                interpretation = f"Model 2, Model 1'den İSTATİSTİKSEL OLARAK ANLAMLI şekilde daha iyi ({self.confidence_level*100:.0f}% CI: [{ci_lower:.6f}, {ci_upper:.6f}])"
            else:
                interpretation = f"Model 1, Model 2'den İSTATİSTİKSEL OLARAK ANLAMLI şekilde daha iyi ({self.confidence_level*100:.0f}% CI: [{ci_lower:.6f}, {ci_upper:.6f}])"
        else:
            interpretation = f"İki model arasında istatistiksel olarak ANLAMLI fark YOK ({self.confidence_level*100:.0f}% CI: [{ci_lower:.6f}, {ci_upper:.6f}] sıfırı içeriyor)"
        
        return {
            'mean_difference': float(mean_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'is_significant': bool(is_significant),
            'confidence_level': self.confidence_level,
            'n_bootstrap': self.n_bootstrap,
            'interpretation': interpretation
        }
    
    def metric_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str = 'rmse'
    ) -> Dict[str, float]:
        """
        Tek bir modelin metrik CI'sını hesapla.
        
        Parameters
        ----------
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahminler
        metric : str, optional
            Metrik: 'rmse', 'mae', 'r2' (varsayılan: 'rmse')
            
        Returns
        -------
        Dict[str, float]
            Metrik, CI
        """
        n = len(y_true)
        metrics = []
        
        np.random.seed(42)
        
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            
            y_true_boot = y_true[idx]
            y_pred_boot = y_pred[idx]
            
            if metric == 'rmse':
                m = np.sqrt(np.mean((y_true_boot - y_pred_boot) ** 2))
            elif metric == 'mae':
                m = np.mean(np.abs(y_true_boot - y_pred_boot))
            elif metric == 'r2':
                m = AdvancedMetrics.r2_score(y_true_boot, y_pred_boot)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            metrics.append(m)
        
        metrics = np.array(metrics)
        
        ci_lower = np.percentile(metrics, self.alpha / 2 * 100)
        ci_upper = np.percentile(metrics, (1 - self.alpha / 2) * 100)
        
        return {
            'metric': metric,
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'confidence_level': self.confidence_level
        }


def print_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Model Karşılaştırması"
):
    """
    Metrik karşılaştırmasını formatlanmış şekilde yazdır.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Model isimleri ve metrikleri
    title : str, optional
        Başlık
    """
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    # DataFrame'e çevir
    df = pd.DataFrame(metrics_dict).T
    
    # Sıralama (RMSE'ye göre, küçükten büyüğe)
    if 'rmse' in df.columns:
        df = df.sort_values('rmse')
    
    print("\n" + df.to_string())
    print("=" * 80)


def comprehensive_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    return_series: bool = False,
    bootstrap_ci: bool = True,
    baseline_name: str = 'baseline'
) -> pd.DataFrame:
    """
    Kapsamlı model karşılaştırması.
    
    Parameters
    ----------
    y_true : np.ndarray
        Gerçek değerler
    predictions_dict : Dict[str, np.ndarray]
        Model isimleri ve tahminleri
    return_series : bool, optional
        Getiri serisi mi? (varsayılan: False)
    bootstrap_ci : bool, optional
        Bootstrap CI hesapla (varsayılan: True)
    baseline_name : str, optional
        Baseline model ismi (varsayılan: 'baseline')
        
    Returns
    -------
    pd.DataFrame
        Kapsamlı karşılaştırma tablosu
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        # Temel metrikler
        metrics = AdvancedMetrics.calculate_all_metrics(
            y_true, y_pred, return_series=return_series
        )
        
        row = {'Model': model_name}
        row.update(metrics)
        
        # Bootstrap CI (baseline'a karşı)
        if bootstrap_ci and model_name != baseline_name and baseline_name in predictions_dict:
            boot = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
            ci_result = boot.performance_difference_ci(
                y_true,
                predictions_dict[baseline_name],
                y_pred,
                metric='rmse'
            )
            
            row['rmse_improvement_%'] = -(ci_result['mean_difference'] / metrics['rmse']) * 100
            row['improvement_significant'] = 'Evet' if ci_result['is_significant'] else 'Hayır'
            row['ci_95_lower'] = ci_result['ci_lower']
            row['ci_95_upper'] = ci_result['ci_upper']
        
        results.append(row)
    
    return pd.DataFrame(results)

