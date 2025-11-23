"""
Rejim geçişleri için Markov chain analizi modülü.

Bu modül, tespit edilen rejimlerin geçiş dinamiklerini Markov zinciri
olarak modeller ve optimal test periyodu önerisi sağlar.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class RegimeMarkovAnalyzer:
    """
    Rejim geçişlerini Markov zinciri olarak modeller.
    
    Bu sınıf, rejim etiketlerinden transition matrix hesaplar,
    stationary distribution bulur ve mixing time analizi yapar.
    
    Attributes
    ----------
    transition_matrix : np.ndarray
        P[i,j] = P(R_t+1 = j | R_t = i)
    stationary_dist : np.ndarray
        Ergodic dağılım π
    mixing_time : float
        Stationary distribution'a yakınsama süresi
    """
    
    def __init__(self):
        """RegimeMarkovAnalyzer başlatıcı."""
        self.transition_matrix = None
        self.stationary_dist = None
        self.mixing_time = None
        self.n_regimes = None
    
    def fit(self, regime_labels: np.ndarray):
        """
        Rejim etiketlerinden Markov chain'i fit et.
        
        Parameters
        ----------
        regime_labels : np.ndarray
            Rejim etiketleri (outlier'lar -1 olarak)
        """
        # Outlier'ları filtrele
        valid_mask = regime_labels != -1
        labels_clean = regime_labels[valid_mask]
        
        if len(labels_clean) == 0:
            raise ValueError("Geçerli rejim etiketi yok (tümü outlier)")
        
        # Transition matrix hesapla
        self.transition_matrix = self.estimate_transition_matrix(labels_clean)
        self.n_regimes = self.transition_matrix.shape[0]
        
        # Stationary distribution
        self.stationary_dist = self.compute_stationary_distribution(
            self.transition_matrix
        )
        
        # Mixing time
        self.mixing_time = self.compute_mixing_time(self.transition_matrix)
    
    @staticmethod
    def estimate_transition_matrix(regime_labels: np.ndarray) -> np.ndarray:
        """
        Transition matrix P[i,j] = P(R_t+1 = j | R_t = i) hesapla.
        
        Parameters
        ----------
        regime_labels : np.ndarray
            Rejim etiketleri
            
        Returns
        -------
        np.ndarray
            Transition matrix (K x K)
            
        Examples
        --------
        >>> labels = np.array([0, 0, 1, 1, 0, 1])
        >>> P = RegimeMarkovAnalyzer.estimate_transition_matrix(labels)
        """
        unique_labels = np.unique(regime_labels)
        K = len(unique_labels)
        
        # Label mapping (eğer consecutive değilse)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        P = np.zeros((K, K))
        
        for t in range(len(regime_labels) - 1):
            i = label_to_idx[regime_labels[t]]
            j = label_to_idx[regime_labels[t + 1]]
            P[i, j] += 1
        
        # Normalize (her satır toplamı 1)
        row_sums = P.sum(axis=1, keepdims=True)
        
        # Zero rows için uniform distribution
        zero_rows = (row_sums == 0).flatten()
        P[zero_rows, :] = 1 / K
        row_sums[zero_rows] = K
        
        P = P / row_sums
        
        return P
    
    @staticmethod
    def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
        """
        Stationary distribution: π^T P = π^T.
        
        Eigenvalue decomposition kullanarak hesaplar:
        π = eigenvector corresponding to eigenvalue = 1
        
        Parameters
        ----------
        P : np.ndarray
            Transition matrix
            
        Returns
        -------
        np.ndarray
            Stationary distribution
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            
            # İlk eigenvector (eigenvalue ≈ 1)
            idx = np.argmin(np.abs(eigenvalues - 1))
            pi = np.real(eigenvectors[:, idx])
            
            # Normalize (sum = 1)
            pi = np.abs(pi)  # Ensure non-negative
            pi = pi / pi.sum()
            
            return pi
        
        except np.linalg.LinAlgError:
            # Fallback: uniform distribution
            warnings.warn("Eigenvalue decomposition başarısız, uniform dağılım kullanılıyor")
            return np.ones(P.shape[0]) / P.shape[0]
    
    @staticmethod
    def compute_mixing_time(
        P: np.ndarray,
        epsilon: float = 0.01,
        max_time: float = 10000
    ) -> float:
        """
        Mixing time: Stationary distribution'a yakınsamak için gereken zaman.
        
        Formula: τ_mix ≈ -1 / log|λ₂|
        
        λ₂ = second largest eigenvalue (magnitude)
        
        Parameters
        ----------
        P : np.ndarray
            Transition matrix
        epsilon : float, optional
            Convergence threshold (varsayılan: 0.01)
        max_time : float, optional
            Maximum mixing time cap (varsayılan: 10000)
            
        Returns
        -------
        float
            Mixing time
        """
        try:
            eigenvalues = np.linalg.eigvals(P)
            eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
            
            # İlk eigenvalue 1 olmalı
            lambda_1 = eigenvalues_sorted[0]
            
            if len(eigenvalues_sorted) < 2:
                return np.inf
            
            lambda_2 = eigenvalues_sorted[1]
            
            # λ₂ ≥ 1 ise mixing yok (irreducible/aperiodic değil)
            if lambda_2 >= 0.999:
                return max_time
            
            # τ_mix = -log(ε) / log(λ₂)
            if lambda_2 > 0:
                mixing_time = -np.log(epsilon) / np.log(lambda_2)
            else:
                mixing_time = max_time
            
            return min(mixing_time, max_time)
        
        except (np.linalg.LinAlgError, ValueError):
            warnings.warn("Mixing time hesaplanamadı, max value döndürülüyor")
            return max_time
    
    def recommend_test_size(
        self,
        coverage_confidence: float = 0.95,
        min_samples_per_regime: int = 20
    ) -> int:
        """
        Tüm rejimleri yeterince örneklemek için minimum test size.
        
        Formula: T_min = max(
            -log(1-γ) · τ_mix · K,
            K · min_samples_per_regime
        )
        
        Parameters
        ----------
        coverage_confidence : float, optional
            Coverage güven seviyesi (varsayılan: 0.95)
        min_samples_per_regime : int, optional
            Her rejimde minimum sample (varsayılan: 20)
            
        Returns
        -------
        int
            Önerilen minimum test size
        """
        if self.transition_matrix is None:
            raise ValueError("Önce fit() çağırın")
        
        K = self.n_regimes
        
        # Ergodic coverage için gerekli T
        if self.mixing_time < np.inf:
            T_ergodic = -np.log(1 - coverage_confidence) * self.mixing_time * K
        else:
            T_ergodic = K * min_samples_per_regime * 5  # Heuristic
        
        # Her rejim için minimum sample
        T_min_samples = K * min_samples_per_regime
        
        # Maximum al
        T_min = max(T_ergodic, T_min_samples)
        
        return int(np.ceil(T_min))
    
    def get_regime_statistics(self) -> Dict[str, any]:
        """
        Rejim istatistiklerini döndür.
        
        Returns
        -------
        Dict[str, any]
            İstatistikler
        """
        if self.transition_matrix is None:
            raise ValueError("Önce fit() çağırın")
        
        # Persistent regimes (P[i,i] > 0.5)
        persistence = np.diag(self.transition_matrix)
        persistent_regimes = np.where(persistence > 0.5)[0]
        
        # Most frequent transitions
        transitions = []
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                if i != j and self.transition_matrix[i, j] > 0.1:
                    transitions.append({
                        'from': int(i),
                        'to': int(j),
                        'prob': float(self.transition_matrix[i, j])
                    })
        
        transitions.sort(key=lambda x: x['prob'], reverse=True)
        
        return {
            'n_regimes': int(self.n_regimes),
            'stationary_distribution': self.stationary_dist.tolist(),
            'mixing_time': float(self.mixing_time),
            'persistence_diagonal': persistence.tolist(),
            'persistent_regimes': persistent_regimes.tolist(),
            'top_transitions': transitions[:5]
        }
    
    def is_test_set_adequate(
        self,
        test_labels: np.ndarray,
        min_coverage_ratio: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Test setinin rejim coverage'ının yeterliliğini kontrol et.
        
        Parameters
        ----------
        test_labels : np.ndarray
            Test set rejim etiketleri
        min_coverage_ratio : float, optional
            Minimum coverage oranı (varsayılan: 0.5)
            
        Returns
        -------
        Tuple[bool, str]
            (is_adequate, explanation)
        """
        if self.stationary_dist is None:
            raise ValueError("Önce fit() çağırın")
        
        # Test setindeki rejimler
        test_labels_clean = test_labels[test_labels != -1]
        test_regimes = set(test_labels_clean)
        
        # Train setindeki toplam rejim sayısı
        total_regimes = self.n_regimes
        
        # Coverage ratio
        coverage = len(test_regimes) / total_regimes
        
        # Her rejim için sample sayısı
        regime_counts = {
            i: np.sum(test_labels_clean == i)
            for i in range(total_regimes)
        }
        
        # Analiz
        issues = []
        
        if coverage < min_coverage_ratio:
            issues.append(
                f"Coverage yetersiz: {coverage:.1%} < {min_coverage_ratio:.1%}"
            )
        
        # Eksik rejimler
        missing_regimes = set(range(total_regimes)) - test_regimes
        if missing_regimes:
            # Ergodic distribution'da önemli olan rejimler mi?
            important_missing = [
                r for r in missing_regimes
                if self.stationary_dist[r] > 0.05
            ]
            
            if important_missing:
                issues.append(
                    f"Önemli rejimler eksik: {important_missing} "
                    f"(π > 0.05)"
                )
        
        # Sample size her rejim için yeterli mi?
        insufficient_samples = [
            (r, count) for r, count in regime_counts.items()
            if count < 20 and count > 0
        ]
        
        if insufficient_samples:
            issues.append(
                f"Yetersiz sample: {insufficient_samples}"
            )
        
        # Sonuç
        is_adequate = len(issues) == 0
        
        if is_adequate:
            explanation = f"✅ Test seti yeterli: {len(test_regimes)}/{total_regimes} rejim, coverage={coverage:.1%}"
        else:
            explanation = "⚠️ Test seti yetersiz:\n  " + "\n  ".join(issues)
        
        return is_adequate, explanation
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Markov analizi raporu oluştur.
        
        Parameters
        ----------
        output_file : Optional[str], optional
            Rapor dosyası (varsayılan: None)
            
        Returns
        -------
        str
            Rapor metni
        """
        if self.transition_matrix is None:
            raise ValueError("Önce fit() çağırın")
        
        lines = []
        lines.append("=" * 80)
        lines.append("MARKOV CHAIN ANALİZİ - REJİM GEÇİŞLERİ")
        lines.append("=" * 80)
        lines.append("")
        
        # 1. Genel bilgi
        lines.append("1️⃣ GENEL BİLGİ")
        lines.append("-" * 80)
        lines.append(f"Rejim sayısı: {self.n_regimes}")
        lines.append(f"Mixing time: {self.mixing_time:.2f}")
        lines.append("")
        
        # 2. Stationary distribution
        lines.append("2️⃣ STATIONARY DISTRIBUTION (Ergodic dağılım)")
        lines.append("-" * 80)
        for i, pi in enumerate(self.stationary_dist):
            lines.append(f"  Rejim {i}: π = {pi:.4f} ({pi*100:.2f}%)")
        lines.append("")
        
        # 3. Transition matrix
        lines.append("3️⃣ TRANSITION MATRIX (En olası geçişler)")
        lines.append("-" * 80)
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                if self.transition_matrix[i, j] > 0.1:
                    lines.append(
                        f"  {i} → {j}: {self.transition_matrix[i, j]:.3f}"
                    )
        lines.append("")
        
        # 4. Persistence
        lines.append("4️⃣ PERSISTENCE (Aynı rejimde kalma olasılığı)")
        lines.append("-" * 80)
        persistence = np.diag(self.transition_matrix)
        for i, p in enumerate(persistence):
            status = "Yüksek" if p > 0.5 else "Düşük"
            lines.append(f"  Rejim {i}: {p:.3f} ({status})")
        lines.append("")
        
        # 5. Öneriler
        lines.append("5️⃣ ÖNERİLER")
        lines.append("-" * 80)
        
        T_rec = self.recommend_test_size(coverage_confidence=0.95)
        lines.append(f"  Önerilen test size: {T_rec} gözlem")
        lines.append(f"  (95% güvenle tüm rejimleri örneklemek için)")
        
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


def analyze_regime_coverage(
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> Dict[str, any]:
    """
    Train ve test setlerindeki rejim coverage'ını analiz et.
    
    Parameters
    ----------
    train_labels : np.ndarray
        Train set rejim etiketleri
    test_labels : np.ndarray
        Test set rejim etiketleri
        
    Returns
    -------
    Dict[str, any]
        Coverage analizi sonuçları
    """
    # Markov analyzer
    analyzer = RegimeMarkovAnalyzer()
    analyzer.fit(train_labels)
    
    # Test adequacy
    is_adequate, explanation = analyzer.is_test_set_adequate(test_labels)
    
    # Statistics
    stats = analyzer.get_regime_statistics()
    
    return {
        'is_test_adequate': is_adequate,
        'explanation': explanation,
        'markov_stats': stats,
        'recommended_test_size': analyzer.recommend_test_size()
    }

