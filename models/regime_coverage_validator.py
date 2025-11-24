"""
Rejim coverage validation ve analiz modülü.

Bu modül, train ve test setlerindeki rejim dağılımlarını analiz eder,
coverage adequacy kontrolü yapar ve iyileştirme önerileri sunar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from models.regime_markov_analysis import RegimeMarkovAnalyzer


class RegimeCoverageValidator:
    """
    Rejim coverage validator.
    
    Train ve test setlerindeki rejim coverage'ını validate eder,
    Multi-Body GRM için yeterli rejim çeşitliliği olup olmadığını kontrol eder.
    
    Attributes
    ----------
    train_labels : np.ndarray
        Train set rejim etiketleri
    test_labels : np.ndarray
        Test set rejim etiketleri
    markov_analyzer : RegimeMarkovAnalyzer
        Markov chain analyzer
    """
    
    def __init__(
        self,
        train_labels: np.ndarray,
        test_labels: np.ndarray
    ):
        """
        RegimeCoverageValidator başlatıcı.
        
        Parameters
        ----------
        train_labels : np.ndarray
            Train set rejim etiketleri
        test_labels : np.ndarray
            Test set rejim etiketleri
        """
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        # Markov analyzer
        self.markov_analyzer = RegimeMarkovAnalyzer()
        try:
            self.markov_analyzer.fit(train_labels)
        except Exception as e:
            print(f"[WARN] Markov analysis başarısız: {e}")
            self.markov_analyzer = None
    
    def compute_coverage_metrics(self) -> Dict[str, any]:
        """
        Coverage metriklerini hesapla.
        
        Returns
        -------
        Dict[str, any]
            Coverage metrikleri
        """
        # Train
        train_clean = self.train_labels[self.train_labels != -1]
        train_regimes = set(train_clean)
        train_counts = {
            int(r): int(np.sum(train_clean == r))
            for r in train_regimes
        }
        
        # Test
        test_clean = self.test_labels[self.test_labels != -1]
        test_regimes = set(test_clean)
        test_counts = {
            int(r): int(np.sum(test_clean == r))
            for r in test_regimes
        }
        
        # Coverage
        coverage_ratio = len(test_regimes) / len(train_regimes) if train_regimes else 0
        
        # Missing regimes
        missing_regimes = train_regimes - test_regimes
        
        # Common regimes
        common_regimes = train_regimes & test_regimes
        
        # Outlier ratios
        train_outlier_ratio = np.sum(self.train_labels == -1) / len(self.train_labels)
        test_outlier_ratio = np.sum(self.test_labels == -1) / len(self.test_labels)
        
        return {
            'n_train_regimes': len(train_regimes),
            'n_test_regimes': len(test_regimes),
            'coverage_ratio': coverage_ratio,
            'missing_regimes': sorted(missing_regimes),
            'common_regimes': sorted(common_regimes),
            'train_regime_counts': train_counts,
            'test_regime_counts': test_counts,
            'train_outlier_ratio': train_outlier_ratio,
            'test_outlier_ratio': test_outlier_ratio
        }
    
    def check_adequacy(
        self,
        min_coverage_ratio: float = 0.5,
        min_test_regimes: int = 3,
        min_samples_per_regime: int = 20
    ) -> Tuple[bool, List[str]]:
        """
        Coverage adequacy kontrolü.
        
        Parameters
        ----------
        min_coverage_ratio : float, optional
            Minimum coverage oranı (varsayılan: 0.5)
        min_test_regimes : int, optional
            Minimum test rejim sayısı (varsayılan: 3)
        min_samples_per_regime : int, optional
            Her rejimde minimum sample (varsayılan: 20)
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_adequate, issues)
        """
        metrics = self.compute_coverage_metrics()
        issues = []
        
        # Check 1: Coverage ratio
        if metrics['coverage_ratio'] < min_coverage_ratio:
            issues.append(
                f"❌ Coverage ratio yetersiz: {metrics['coverage_ratio']:.1%} "
                f"< {min_coverage_ratio:.1%}"
            )
        
        # Check 2: Test regime count
        if metrics['n_test_regimes'] < min_test_regimes:
            issues.append(
                f"❌ Test rejim sayısı yetersiz: {metrics['n_test_regimes']} "
                f"< {min_test_regimes}"
            )
        
        # Check 3: Missing regimes
        if metrics['missing_regimes']:
            issues.append(
                f"⚠️  Eksik rejimler: {metrics['missing_regimes']}"
            )
        
        # Check 4: Sample counts
        insufficient = [
            (r, count) for r, count in metrics['test_regime_counts'].items()
            if count < min_samples_per_regime
        ]
        
        if insufficient:
            issues.append(
                f"⚠️  Yetersiz sample: {insufficient}"
            )
        
        # Check 5: Outlier ratio
        if metrics['test_outlier_ratio'] > 0.3:
            issues.append(
                f"⚠️  Yüksek outlier oranı: {metrics['test_outlier_ratio']:.1%}"
            )
        
        is_adequate = len(issues) == 0
        
        return is_adequate, issues
    
    def recommend_improvements(self) -> List[Dict[str, str]]:
        """
        İyileştirme önerileri.
        
        Returns
        -------
        List[Dict[str, str]]
            Her öneri: {'problem': ..., 'solution': ..., 'priority': ...}
        """
        metrics = self.compute_coverage_metrics()
        is_adequate, issues = self.check_adequacy()
        
        recommendations = []
        
        # Problem 1: Low coverage
        if metrics['coverage_ratio'] < 0.5:
            recommendations.append({
                'problem': f"Coverage ratio: {metrics['coverage_ratio']:.1%}",
                'solution': 'Stratified split kullan (StratifiedTimeSeriesSplit)',
                'priority': '🔴 CRITICAL',
                'expected_improvement': 'Test coverage → 80-100%'
            })
        
        # Problem 2: Single or few regimes in test
        if metrics['n_test_regimes'] < 3:
            recommendations.append({
                'problem': f"Test'te sadece {metrics['n_test_regimes']} rejim",
                'solution': 'Test periyodunu uzat veya stratified split kullan',
                'priority': '🔴 CRITICAL',
                'expected_improvement': 'Test regimes → 3-5+'
            })
        
        # Problem 3: Missing regimes
        if metrics['missing_regimes']:
            recommendations.append({
                'problem': f"Eksik rejimler: {len(metrics['missing_regimes'])}",
                'solution': 'Her rejimden proportional sampling yap',
                'priority': '🟡 HIGH',
                'expected_improvement': 'All regimes represented'
            })
        
        # Problem 4: Insufficient samples
        insufficient_count = sum(
            1 for count in metrics['test_regime_counts'].values()
            if count < 20
        )
        
        if insufficient_count > 0:
            recommendations.append({
                'problem': f"{insufficient_count} rejim'de < 20 sample",
                'solution': 'Test ratio artır veya stratified split',
                'priority': '🟡 HIGH',
                'expected_improvement': 'All regimes ≥ 20 samples'
            })
        
        # Problem 5: Markov-based recommendation
        if self.markov_analyzer is not None:
            try:
                T_rec = self.markov_analyzer.recommend_test_size()
                T_current = len(self.test_labels)
                
                if T_current < T_rec:
                    recommendations.append({
                        'problem': f"Test size: {T_current} < {T_rec} (Markov önerisi)",
                        'solution': f"Test ratio artır: current → {T_rec / len(self.train_labels):.1%}",
                        'priority': '🟢 MEDIUM',
                        'expected_improvement': 'Ergodic coverage garantisi'
                    })
            except:
                pass
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Detaylı coverage raporu oluştur.
        
        Parameters
        ----------
        output_file : Optional[str], optional
            Rapor dosyası (varsayılan: None)
            
        Returns
        -------
        str
            Rapor metni
        """
        metrics = self.compute_coverage_metrics()
        is_adequate, issues = self.check_adequacy()
        recommendations = self.recommend_improvements()
        
        lines = []
        lines.append("=" * 80)
        lines.append("REGIME COVERAGE VALIDATION RAPORU")
        lines.append("=" * 80)
        lines.append("")
        
        # 1. Özet
        lines.append("1️⃣  ÖZET")
        lines.append("-" * 80)
        lines.append(f"Train Regimes: {metrics['n_train_regimes']}")
        lines.append(f"Test Regimes:  {metrics['n_test_regimes']}")
        lines.append(f"Coverage:      {metrics['coverage_ratio']:.1%}")
        lines.append(f"Status:        {'✅ ADEQUATE' if is_adequate else '❌ INADEQUATE'}")
        lines.append("")
        
        # 2. Detaylı metrikler
        lines.append("2️⃣  DETAYLI METRİKLER")
        lines.append("-" * 80)
        lines.append(f"Common Regimes:  {metrics['common_regimes']}")
        lines.append(f"Missing Regimes: {metrics['missing_regimes']}")
        lines.append(f"Train Outliers:  {metrics['train_outlier_ratio']:.1%}")
        lines.append(f"Test Outliers:   {metrics['test_outlier_ratio']:.1%}")
        lines.append("")
        
        # 3. Rejim dağılımları
        lines.append("3️⃣  REJİM DAĞILIMLARI")
        lines.append("-" * 80)
        lines.append(f"{'Regime':<10} {'Train Count':<15} {'Test Count':<15} {'Test %':<10}")
        lines.append("-" * 50)
        
        all_regimes = sorted(
            set(metrics['train_regime_counts'].keys()) | 
            set(metrics['test_regime_counts'].keys())
        )
        
        for regime in all_regimes:
            train_count = metrics['train_regime_counts'].get(regime, 0)
            test_count = metrics['test_regime_counts'].get(regime, 0)
            test_non_outlier = len(self.test_labels[self.test_labels != -1])
            test_pct = (test_count / test_non_outlier * 100) if test_non_outlier > 0 else 0.0
            
            lines.append(
                f"{regime:<10} {train_count:<15} {test_count:<15} {test_pct:<10.1f}%"
            )
        
        lines.append("")
        
        # 4. Sorunlar
        if issues:
            lines.append("4️⃣  TESPİT EDİLEN SORUNLAR")
            lines.append("-" * 80)
            for i, issue in enumerate(issues, 1):
                lines.append(f"{i}. {issue}")
            lines.append("")
        
        # 5. Öneriler
        if recommendations:
            lines.append("5️⃣  İYİLEŞTİRME ÖNERİLERİ")
            lines.append("-" * 80)
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"\n{rec['priority']} Öneri {i}:")
                lines.append(f"  Problem:     {rec['problem']}")
                lines.append(f"  Çözüm:       {rec['solution']}")
                lines.append(f"  Beklenen:    {rec['expected_improvement']}")
            lines.append("")
        
        # 6. Markov analizi
        if self.markov_analyzer is not None:
            lines.append("6️⃣  MARKOV CHAIN ANALİZİ")
            lines.append("-" * 80)
            
            try:
                stats = self.markov_analyzer.get_regime_statistics()
                lines.append(f"Mixing Time: {stats['mixing_time']:.2f}")
                lines.append(f"Stationary Distribution:")
                for i, pi in enumerate(stats['stationary_distribution']):
                    lines.append(f"  Regime {i}: π = {pi:.4f} ({pi*100:.2f}%)")
                
                T_rec = self.markov_analyzer.recommend_test_size()
                lines.append(f"\nÖnerilen Test Size: {T_rec}")
            except Exception as e:
                lines.append(f"[WARN] Markov analizi hatası: {e}")
            
            lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[OK] Rapor kaydedildi: {output_file}")
        
        return report


def quick_coverage_check(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Hızlı coverage kontrolü.
    
    Parameters
    ----------
    train_labels : np.ndarray
        Train rejim etiketleri
    test_labels : np.ndarray
        Test rejim etiketleri
    verbose : bool, optional
        Detaylı çıktı (varsayılan: True)
        
    Returns
    -------
    Dict[str, any]
        Coverage sonuçları
    """
    validator = RegimeCoverageValidator(train_labels, test_labels)
    
    metrics = validator.compute_coverage_metrics()
    is_adequate, issues = validator.check_adequacy()
    recommendations = validator.recommend_improvements()
    
    if verbose:
        print("\n" + "=" * 60)
        print("QUICK COVERAGE CHECK")
        print("=" * 60)
        print(f"Train Regimes: {metrics['n_train_regimes']}")
        print(f"Test Regimes:  {metrics['n_test_regimes']}")
        print(f"Coverage:      {metrics['coverage_ratio']:.1%}")
        print(f"Status:        {'✅ OK' if is_adequate else '❌ PROBLEM'}")
        
        if not is_adequate:
            print(f"\n⚠️  {len(issues)} sorun tespit edildi:")
            for issue in issues:
                print(f"  • {issue}")
            
            print(f"\n💡 {len(recommendations)} öneri:")
            for rec in recommendations[:3]:  # İlk 3 öneri
                print(f"  {rec['priority']} {rec['solution']}")
        
        print("=" * 60)
    
    return {
        'metrics': metrics,
        'is_adequate': is_adequate,
        'issues': issues,
        'recommendations': recommendations
    }

