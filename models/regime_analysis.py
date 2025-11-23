"""
Rejim analizi ve karakterizasyonu.

Bu modül, Multi-Body GRM için tespit edilen rejimleri analiz eder
ve karakterize eder.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import DBSCAN
import warnings


class RegimeAnalyzer:
    """
    Rejim analizi ve karakterizasyonu sınıfı.
    
    Bu sınıf, tespit edilen rejimlerin özelliklerini analiz eder,
    istatistiksel karakteristiklerini çıkarır ve görselleştirir.
    
    Attributes
    ----------
    labels : np.ndarray
        Rejim etiketleri
    data : np.ndarray
        Zaman serisi verisi
    regimes : Dict
        Rejim bilgileri
    """
    
    def __init__(self):
        """RegimeAnalyzer başlatıcı."""
        self.labels = None
        self.data = None
        self.regimes = {}
    
    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        Rejimleri analiz et.
        
        Parameters
        ----------
        data : np.ndarray
            Zaman serisi verisi
        labels : np.ndarray
            Rejim etiketleri (DBSCAN çıktısı)
        """
        self.data = data
        self.labels = labels
        
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Outliers
        
        for label in unique_labels:
            self.regimes[label] = self._analyze_regime(label)
    
    def _analyze_regime(self, label: int) -> Dict:
        """
        Tek bir rejimi analiz et.
        
        Parameters
        ----------
        label : int
            Rejim etiketi
            
        Returns
        -------
        Dict
            Rejim özellikleri
        """
        mask = self.labels == label
        regime_data = self.data[mask]
        regime_indices = np.where(mask)[0]
        
        # Temel istatistikler
        stats_dict = {
            'label': int(label),
            'size': len(regime_data),
            'mean': float(np.mean(regime_data)),
            'std': float(np.std(regime_data)),
            'min': float(np.min(regime_data)),
            'max': float(np.max(regime_data)),
            'median': float(np.median(regime_data)),
            'skewness': float(stats.skew(regime_data)),
            'kurtosis': float(stats.kurtosis(regime_data)),
            'indices': regime_indices
        }
        
        # Volatilite karakterizasyonu
        if stats_dict['std'] < 0.5:
            stats_dict['volatility_regime'] = 'Low'
        elif stats_dict['std'] < 1.5:
            stats_dict['volatility_regime'] = 'Medium'
        else:
            stats_dict['volatility_regime'] = 'High'
        
        # Trend karakterizasyonu
        if len(regime_data) > 1:
            # Linear regression
            x = np.arange(len(regime_data))
            slope, intercept, r_value, _, _ = stats.linregress(x, regime_data)
            stats_dict['trend_slope'] = float(slope)
            stats_dict['trend_r2'] = float(r_value ** 2)
            
            if abs(slope) < 0.01:
                stats_dict['trend_type'] = 'Stationary'
            elif slope > 0:
                stats_dict['trend_type'] = 'Upward'
            else:
                stats_dict['trend_type'] = 'Downward'
        else:
            stats_dict['trend_slope'] = 0.0
            stats_dict['trend_r2'] = 0.0
            stats_dict['trend_type'] = 'Unknown'
        
        # Autokorelasyon
        if len(regime_data) > 10:
            acf = self._compute_acf(regime_data, nlags=5)
            stats_dict['acf_lag1'] = float(acf[1]) if len(acf) > 1 else 0.0
            stats_dict['persistence'] = 'High' if abs(acf[1]) > 0.5 else 'Low'
        else:
            stats_dict['acf_lag1'] = 0.0
            stats_dict['persistence'] = 'Unknown'
        
        return stats_dict
    
    def _compute_acf(self, data: np.ndarray, nlags: int = 5) -> np.ndarray:
        """
        Autokorelasyon fonksiyonu hesapla.
        
        Parameters
        ----------
        data : np.ndarray
            Zaman serisi
        nlags : int, optional
            Lag sayısı (varsayılan: 5)
            
        Returns
        -------
        np.ndarray
            ACF değerleri
        """
        data_centered = data - np.mean(data)
        c0 = np.dot(data_centered, data_centered) / len(data)
        
        acf = np.ones(nlags + 1)
        for lag in range(1, nlags + 1):
            if lag < len(data):
                c_lag = np.dot(data_centered[:-lag], data_centered[lag:]) / len(data)
                acf[lag] = c_lag / c0
        
        return acf
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Rejim özet tablosu.
        
        Returns
        -------
        pd.DataFrame
            Rejim özeti
        """
        rows = []
        for label, info in self.regimes.items():
            rows.append({
                'Regime': label,
                'Size': info['size'],
                'Mean': info['mean'],
                'Std': info['std'],
                'Volatility': info['volatility_regime'],
                'Trend': info['trend_type'],
                'Skewness': info['skewness'],
                'Kurtosis': info['kurtosis'],
                'Persistence': info['persistence']
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('Regime')
    
    def get_regime_transitions(self) -> Dict[str, int]:
        """
        Rejim geçişlerini analiz et.
        
        Returns
        -------
        Dict[str, int]
            Geçiş matrisi (rejim1->rejim2: count)
        """
        transitions = {}
        
        for i in range(len(self.labels) - 1):
            current = self.labels[i]
            next_label = self.labels[i + 1]
            
            if current != -1 and next_label != -1:
                key = f"{current}->{next_label}"
                transitions[key] = transitions.get(key, 0) + 1
        
        return transitions
    
    def get_regime_durations(self) -> Dict[int, List[int]]:
        """
        Her rejimin süre dağılımını hesapla.
        
        Returns
        -------
        Dict[int, List[int]]
            Rejim -> süre listesi
        """
        durations = {}
        
        if len(self.labels) == 0:
            return durations
        
        current_label = self.labels[0]
        current_duration = 1
        
        for i in range(1, len(self.labels)):
            if self.labels[i] == current_label:
                current_duration += 1
            else:
                if current_label != -1:
                    if current_label not in durations:
                        durations[current_label] = []
                    durations[current_label].append(current_duration)
                
                current_label = self.labels[i]
                current_duration = 1
        
        # Son rejim
        if current_label != -1:
            if current_label not in durations:
                durations[current_label] = []
            durations[current_label].append(current_duration)
        
        return durations
    
    def characterize_dataset(self) -> Dict[str, any]:
        """
        Veri seti rejim karakterizasyonu.
        
        Returns
        -------
        Dict[str, any]
            Veri seti özellikleri
        """
        n_regimes = len(self.regimes)
        n_outliers = np.sum(self.labels == -1)
        
        # Dominant rejim
        regime_sizes = [(label, info['size']) for label, info in self.regimes.items()]
        regime_sizes.sort(key=lambda x: x[1], reverse=True)
        dominant_regime = regime_sizes[0][0] if regime_sizes else None
        
        # Rejim geçişleri
        transitions = self.get_regime_transitions()
        n_transitions = sum(transitions.values())
        
        # Rejim süreleri
        durations = self.get_regime_durations()
        avg_durations = {
            label: np.mean(durs) for label, durs in durations.items()
        }
        
        return {
            'n_regimes': n_regimes,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(self.labels) if len(self.labels) > 0 else 0,
            'dominant_regime': dominant_regime,
            'regime_sizes': regime_sizes,
            'n_transitions': n_transitions,
            'avg_regime_durations': avg_durations
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Rejim analizi raporu oluştur.
        
        Parameters
        ----------
        output_file : Optional[str], optional
            Rapor dosyası (varsayılan: None)
            
        Returns
        -------
        str
            Rapor metni
        """
        lines = []
        lines.append("=" * 80)
        lines.append("📊 REJİM ANALİZİ RAPORU")
        lines.append("=" * 80)
        lines.append("")
        
        # 1. Genel özet
        dataset_char = self.characterize_dataset()
        lines.append("1️⃣ VERİ SETİ ÖZETİ")
        lines.append("-" * 80)
        lines.append(f"   Toplam Rejim Sayısı: {dataset_char['n_regimes']}")
        lines.append(f"   Outlier Sayısı: {dataset_char['n_outliers']} ({dataset_char['outlier_ratio']*100:.1f}%)")
        lines.append(f"   Dominant Rejim: Regime {dataset_char['dominant_regime']}")
        lines.append(f"   Toplam Geçiş Sayısı: {dataset_char['n_transitions']}")
        lines.append("")
        
        # 2. Rejim detayları
        lines.append("2️⃣ REJİM DETAYLARI")
        lines.append("-" * 80)
        summary_df = self.get_regime_summary()
        lines.append(summary_df.to_string(index=False))
        lines.append("")
        
        # 3. Rejim geçişleri
        lines.append("3️⃣ REJİM GEÇİŞLERİ")
        lines.append("-" * 80)
        transitions = self.get_regime_transitions()
        for key, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"   {key}: {count} kez")
        lines.append("")
        
        # 4. Rejim süreleri
        lines.append("4️⃣ REJİM SÜRELERİ (Ortalama)")
        lines.append("-" * 80)
        for label, avg_dur in dataset_char['avg_regime_durations'].items():
            lines.append(f"   Regime {label}: {avg_dur:.1f} gözlem")
        lines.append("")
        
        # 5. Öneriler
        lines.append("5️⃣ ANALİZ ÖNERİLERİ")
        lines.append("-" * 80)
        
        if dataset_char['n_regimes'] < 2:
            lines.append("   ⚠️  UYARI: Çok az rejim tespit edildi. DBSCAN parametrelerini gözden geçirin.")
        
        if dataset_char['outlier_ratio'] > 0.3:
            lines.append("   ⚠️  UYARI: Yüksek outlier oranı. Veri ön işleme gerekebilir.")
        
        if dataset_char['n_transitions'] < 5:
            lines.append("   ⚠️  UYARI: Çok az rejim geçişi. Multi-Body GRM'in faydası sınırlı olabilir.")
        
        lines.append("")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"[OK] Rejim analizi raporu kaydedildi: {output_file}")
        
        return report_text


def analyze_regime_diversity(
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> Dict[str, any]:
    """
    Train ve test setlerindeki rejim çeşitliliğini analiz et.
    
    Parameters
    ----------
    train_labels : np.ndarray
        Train set rejim etiketleri
    test_labels : np.ndarray
        Test set rejim etiketleri
        
    Returns
    -------
    Dict[str, any]
        Rejim çeşitliliği analizi
    """
    train_regimes = set(train_labels)
    train_regimes.discard(-1)
    
    test_regimes = set(test_labels)
    test_regimes.discard(-1)
    
    common_regimes = train_regimes & test_regimes
    train_only = train_regimes - test_regimes
    test_only = test_regimes - train_regimes
    
    coverage = len(common_regimes) / len(train_regimes) if len(train_regimes) > 0 else 0
    
    return {
        'train_n_regimes': len(train_regimes),
        'test_n_regimes': len(test_regimes),
        'common_regimes': len(common_regimes),
        'train_only_regimes': len(train_only),
        'test_only_regimes': len(test_only),
        'coverage_ratio': coverage,
        'is_sufficient': coverage >= 0.5  # En az %50 kapsama
    }


def recommend_dbscan_params(
    data: np.ndarray,
    feature_matrix: np.ndarray,
    eps_range: List[float] = None,
    min_samples_range: List[int] = None
) -> Tuple[float, int]:
    """
    DBSCAN için optimal parametreleri öner.
    
    Silhouette score kullanarak en iyi eps ve min_samples değerlerini bulur.
    
    Parameters
    ----------
    data : np.ndarray
        Zaman serisi verisi
    feature_matrix : np.ndarray
        Özellik matrisi (DBSCAN için)
    eps_range : List[float], optional
        Test edilecek eps değerleri
    min_samples_range : List[int], optional
        Test edilecek min_samples değerleri
        
    Returns
    -------
    Tuple[float, int]
        Optimal (eps, min_samples)
    """
    from sklearn.metrics import silhouette_score
    
    if eps_range is None:
        # Feature matrix'in scale'ine göre
        distances = []
        for i in range(min(100, len(feature_matrix))):
            for j in range(i + 1, min(100, len(feature_matrix))):
                dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances.append(dist)
        
        median_dist = np.median(distances)
        eps_range = [median_dist * 0.5, median_dist, median_dist * 1.5]
    
    if min_samples_range is None:
        min_samples_range = [3, 5, 10]
    
    best_score = -1
    best_eps = eps_range[0]
    best_min_samples = min_samples_range[0]
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(feature_matrix)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # En az 2 cluster olmalı
                    if n_clusters < 2:
                        continue
                    
                    # Silhouette score
                    score = silhouette_score(feature_matrix, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                
                except Exception:
                    continue
    
    return best_eps, best_min_samples

