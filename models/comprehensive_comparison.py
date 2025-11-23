"""
Kapsamlı model karşılaştırma ve raporlama.

Bu modül, farklı modellerin performansını çok boyutlu karşılaştırır,
istatistiksel testler uygular ve sonuçları raporlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

from models.advanced_metrics import AdvancedMetrics, BootstrapCI
from models.statistical_tests import StatisticalTests


class ComprehensiveComparison:
    """
    Kapsamlı model karşılaştırma sınıfı.
    
    Bu sınıf, birden fazla modelin performansını değerlendirmek
    ve istatistiksel olarak karşılaştırmak için araçlar sağlar.
    
    Attributes
    ----------
    baseline_name : str
        Baseline model adı
    models : Dict[str, Dict]
        Model sonuçları
    """
    
    def __init__(self, baseline_name: str = 'ARIMA'):
        """
        ComprehensiveComparison başlatıcı.
        
        Parameters
        ----------
        baseline_name : str, optional
            Baseline model adı (varsayılan: 'ARIMA')
        """
        self.baseline_name = baseline_name
        self.models = {}
        self.y_true = None
    
    def add_model_results(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        additional_info: Optional[Dict] = None
    ):
        """
        Model sonuçlarını ekle.
        
        Parameters
        ----------
        model_name : str
            Model adı
        y_true : np.ndarray
            Gerçek değerler
        y_pred : np.ndarray
            Tahminler
        additional_info : Optional[Dict], optional
            Ek bilgiler (örn. training time)
        """
        if self.y_true is None:
            self.y_true = y_true
        
        # Temizlik
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                 np.isinf(y_true) | np.isinf(y_pred))
        
        self.models[model_name] = {
            'y_pred': y_pred[mask],
            'y_true_clean': y_true[mask],
            'mask': mask,
            'additional_info': additional_info or {}
        }
    
    def calculate_all_metrics(
        self,
        return_series: bool = False
    ) -> pd.DataFrame:
        """
        Tüm modeller için metrikleri hesapla.
        
        Parameters
        ----------
        return_series : bool, optional
            Getiri serisi mi? (varsayılan: False)
            
        Returns
        -------
        pd.DataFrame
            Metrik tablosu
        """
        results = []
        
        for model_name, data in self.models.items():
            metrics = AdvancedMetrics.calculate_all_metrics(
                data['y_true_clean'],
                data['y_pred'],
                return_series=return_series
            )
            
            row = {'Model': model_name}
            row.update(metrics)
            
            # Ek bilgiler varsa ekle
            if data['additional_info']:
                row.update(data['additional_info'])
            
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # RMSE'ye göre sırala
        if 'rmse' in df.columns:
            df = df.sort_values('rmse')
        
        return df
    
    def calculate_improvements(self) -> pd.DataFrame:
        """
        Baseline'a göre iyileştirmeleri hesapla.
        
        Returns
        -------
        pd.DataFrame
            İyileştirme tablosu
        """
        if self.baseline_name not in self.models:
            raise ValueError(f"Baseline model '{self.baseline_name}' bulunamadı.")
        
        baseline_data = self.models[self.baseline_name]
        baseline_y_pred = baseline_data['y_pred']
        baseline_y_true = baseline_data['y_true_clean']
        
        baseline_rmse = AdvancedMetrics.rmse(baseline_y_true, baseline_y_pred)
        baseline_mae = AdvancedMetrics.mae(baseline_y_true, baseline_y_pred)
        
        results = []
        
        for model_name, data in self.models.items():
            if model_name == self.baseline_name:
                continue
            
            model_rmse = AdvancedMetrics.rmse(data['y_true_clean'], data['y_pred'])
            model_mae = AdvancedMetrics.mae(data['y_true_clean'], data['y_pred'])
            
            # İyileştirme yüzdesi
            rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100
            mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
            
            results.append({
                'Model': model_name,
                'RMSE': model_rmse,
                'RMSE_Improvement_%': rmse_improvement,
                'MAE': model_mae,
                'MAE_Improvement_%': mae_improvement
            })
        
        return pd.DataFrame(results).sort_values('RMSE_Improvement_%', ascending=False)
    
    def perform_statistical_tests(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        İstatistiksel testleri uygula.
        
        Parameters
        ----------
        significance_level : float, optional
            Anlamlılık seviyesi (varsayılan: 0.05)
            
        Returns
        -------
        pd.DataFrame
            İstatistiksel test sonuçları
        """
        if self.baseline_name not in self.models:
            raise ValueError(f"Baseline model '{self.baseline_name}' bulunamadı.")
        
        baseline_data = self.models[self.baseline_name]
        baseline_y_pred = baseline_data['y_pred']
        baseline_y_true = baseline_data['y_true_clean']
        baseline_errors = baseline_y_true - baseline_y_pred
        
        results = []
        
        for model_name, data in self.models.items():
            if model_name == self.baseline_name:
                continue
            
            model_errors = data['y_true_clean'] - data['y_pred']
            
            # Diebold-Mariano test
            try:
                dm_stat, dm_pval = StatisticalTests.diebold_mariano_test(
                    baseline_errors, model_errors, alternative='two-sided'
                )
            except Exception as e:
                dm_stat, dm_pval = np.nan, np.nan
            
            # ARCH-LM test (model residuals)
            try:
                arch_lm, arch_pval = StatisticalTests.arch_lm_test(model_errors, lags=5)
            except Exception:
                arch_lm, arch_pval = np.nan, np.nan
            
            # Ljung-Box test (model residuals)
            try:
                lb_stats, lb_pvals = StatisticalTests.ljung_box_test(model_errors, lags=10)
                lb_pval = lb_pvals[-1]  # Son lag p-value
            except Exception:
                lb_pval = np.nan
            
            # Anlamlılık yorumları
            dm_significant = 'Evet' if dm_pval < significance_level else 'Hayır'
            arch_significant = 'Evet' if arch_pval < significance_level else 'Hayır'
            lb_significant = 'Evet' if lb_pval < significance_level else 'Hayır'
            
            results.append({
                'Model': model_name,
                'DM_Statistic': dm_stat,
                'DM_p_value': dm_pval,
                'DM_Significant': dm_significant,
                'ARCH_LM_p_value': arch_pval,
                'ARCH_Present': arch_significant,
                'LB_p_value': lb_pval,
                'Autocorrelation': lb_significant
            })
        
        return pd.DataFrame(results)
    
    def perform_bootstrap_comparison(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Bootstrap ile karşılaştırma.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Bootstrap iterasyon sayısı (varsayılan: 1000)
        confidence_level : float, optional
            Güven seviyesi (varsayılan: 0.95)
            
        Returns
        -------
        pd.DataFrame
            Bootstrap karşılaştırma sonuçları
        """
        if self.baseline_name not in self.models:
            raise ValueError(f"Baseline model '{self.baseline_name}' bulunamadı.")
        
        baseline_data = self.models[self.baseline_name]
        baseline_y_pred = baseline_data['y_pred']
        baseline_y_true = baseline_data['y_true_clean']
        
        boot = BootstrapCI(n_bootstrap=n_bootstrap, confidence_level=confidence_level)
        results = []
        
        for model_name, data in self.models.items():
            if model_name == self.baseline_name:
                continue
            
            # RMSE difference CI
            ci_result = boot.performance_difference_ci(
                baseline_y_true,
                baseline_y_pred,
                data['y_pred'],
                metric='rmse'
            )
            
            results.append({
                'Model': model_name,
                'Mean_Difference': ci_result['mean_difference'],
                'CI_Lower': ci_result['ci_lower'],
                'CI_Upper': ci_result['ci_upper'],
                'Is_Significant': 'Evet' if ci_result['is_significant'] else 'Hayır',
                'Interpretation': ci_result['interpretation']
            })
        
        return pd.DataFrame(results)
    
    def generate_comprehensive_report(
        self,
        return_series: bool = False,
        output_file: Optional[str] = None
    ) -> str:
        """
        Kapsamlı karşılaştırma raporu oluştur.
        
        Parameters
        ----------
        return_series : bool, optional
            Getiri serisi mi? (varsayılan: False)
        output_file : Optional[str], optional
            Rapor dosyası (varsayılan: None, ekrana yazdır)
            
        Returns
        -------
        str
            Rapor metni
        """
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("📊 KAPSAMLI MODEL KARŞILAŞTIRMA RAPORU")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # 1. Temel metrikler
        report_lines.append("1️⃣ TEMEL PERFORMANS METRİKLERİ")
        report_lines.append("-" * 100)
        metrics_df = self.calculate_all_metrics(return_series=return_series)
        report_lines.append(metrics_df.to_string(index=False))
        report_lines.append("")
        
        # 2. İyileştirme analizi
        if len(self.models) > 1:
            report_lines.append("2️⃣ BASELINE'A GÖRE İYİLEŞTİRMELER")
            report_lines.append("-" * 100)
            improvements_df = self.calculate_improvements()
            report_lines.append(improvements_df.to_string(index=False))
            report_lines.append("")
        
        # 3. İstatistiksel testler
        if len(self.models) > 1:
            report_lines.append("3️⃣ İSTATİSTİKSEL ANLAMLILIK TESTLERİ")
            report_lines.append("-" * 100)
            stats_df = self.perform_statistical_tests()
            report_lines.append(stats_df.to_string(index=False))
            report_lines.append("")
            report_lines.append("   Not:")
            report_lines.append("   - DM_Significant: Baseline'dan anlamlı farklılık var mı?")
            report_lines.append("   - ARCH_Present: ARCH etkileri var mı? (Evet = heteroskedasticity)")
            report_lines.append("   - Autocorrelation: Otokorelasyon var mı? (Evet = beyaz gürültü değil)")
            report_lines.append("")
        
        # 4. Bootstrap CI
        if len(self.models) > 1:
            report_lines.append("4️⃣ BOOTSTRAP GÜVEN ARALIKLARI")
            report_lines.append("-" * 100)
            try:
                boot_df = self.perform_bootstrap_comparison()
                report_lines.append(boot_df[['Model', 'Mean_Difference', 'CI_Lower', 'CI_Upper', 'Is_Significant']].to_string(index=False))
                report_lines.append("")
                report_lines.append("   Yorumlar:")
                for _, row in boot_df.iterrows():
                    report_lines.append(f"   - {row['Model']}: {row['Interpretation']}")
                report_lines.append("")
            except Exception as e:
                report_lines.append(f"   Bootstrap hesaplaması başarısız: {str(e)}")
                report_lines.append("")
        
        # 5. Genel Değerlendirme
        report_lines.append("5️⃣ GENEL DEĞERLENDİRME")
        report_lines.append("-" * 100)
        
        # En iyi model
        metrics_df_sorted = metrics_df.sort_values('rmse')
        best_model = metrics_df_sorted.iloc[0]
        report_lines.append(f"   ✅ En İyi Model (RMSE): {best_model['Model']} (RMSE = {best_model['rmse']:.6f})")
        
        # İyileştirme özeti
        if len(self.models) > 1:
            improvements_df_sorted = improvements_df.sort_values('RMSE_Improvement_%', ascending=False)
            if len(improvements_df_sorted) > 0:
                top_improvement = improvements_df_sorted.iloc[0]
                report_lines.append(f"   🚀 En Fazla İyileştirme: {top_improvement['Model']} ({top_improvement['RMSE_Improvement_%']:.2f}%)")
        
        # İstatistiksel anlamlılık özeti
        if len(self.models) > 1:
            stats_df = self.perform_statistical_tests()
            significant_models = stats_df[stats_df['DM_Significant'] == 'Evet']
            report_lines.append(f"   📈 İstatistiksel Olarak Anlamlı İyileştirme: {len(significant_models)}/{len(stats_df)} model")
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        report_text = "\n".join(report_lines)
        
        # Dosyaya yaz
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"[OK] Rapor kaydedildi: {output_file}")
        
        return report_text


def quick_compare(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    baseline_name: str = 'ARIMA',
    return_series: bool = False,
    output_file: Optional[str] = None
) -> str:
    """
    Hızlı karşılaştırma yapmak için yardımcı fonksiyon.
    
    Parameters
    ----------
    y_true : np.ndarray
        Gerçek değerler
    predictions_dict : Dict[str, np.ndarray]
        Model tahminleri
    baseline_name : str, optional
        Baseline model adı (varsayılan: 'ARIMA')
    return_series : bool, optional
        Getiri serisi mi? (varsayılan: False)
    output_file : Optional[str], optional
        Rapor dosyası (varsayılan: None)
        
    Returns
    -------
    str
        Rapor metni
    """
    comp = ComprehensiveComparison(baseline_name=baseline_name)
    
    for model_name, y_pred in predictions_dict.items():
        comp.add_model_results(model_name, y_true, y_pred)
    
    return comp.generate_comprehensive_report(
        return_series=return_series,
        output_file=output_file
    )

