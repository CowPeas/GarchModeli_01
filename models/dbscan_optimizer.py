"""
DBSCAN parametre optimizasyonu modülü.

Bu modül, DBSCAN clustering algoritması için optimal (ε, minPts)
parametrelerini bulmak üzere k-distance analizi ve grid search sağlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings


class DBSCANOptimizer:
    """
    DBSCAN parametrelerini otomatik optimize eder.
    
    Bu sınıf, k-distance graph analizi, silhouette score optimizasyonu
    ve Hopkins statistic ile clustering quality değerlendirmesi yapar.
    
    Methods
    -------
    compute_k_distances
        Her nokta için k-NN mesafesi
    find_elbow_point
        K-distance grafiğinde elbow noktası
    optimize_eps_minpts_grid
        Grid search ile optimal parametreler
    """
    
    @staticmethod
    def compute_k_distances(X: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Her nokta için k-nearest neighbor mesafesini hesapla.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        k : int, optional
            Komşu sayısı (varsayılan: 5)
            
        Returns
        -------
        np.ndarray
            K-distances (descending order)
            
        Examples
        --------
        >>> X = np.random.randn(100, 3)
        >>> k_dists = DBSCANOptimizer.compute_k_distances(X, k=5)
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # k-distance (ilk mesafe 0 olduğu için k+1'inci)
        k_distances = distances[:, k]
        
        # Descending order
        k_distances_sorted = np.sort(k_distances)[::-1]
        
        return k_distances_sorted
    
    @staticmethod
    def find_elbow_point(k_distances: np.ndarray) -> Tuple[float, int]:
        """
        K-distance grafiğindeki "elbow point"i bul (optimal ε).
        
        Method: Maximum curvature (2nd derivative)
        
        Parameters
        ----------
        k_distances : np.ndarray
            K-distance values (sorted)
            
        Returns
        -------
        Tuple[float, int]
            (epsilon, elbow_index)
        """
        if len(k_distances) < 3:
            return k_distances[0], 0
        
        # 2. türev (discrete approximation)
        second_derivative = np.abs(np.diff(k_distances, 2))
        
        # Maksimum curvature noktası
        elbow_idx = np.argmax(second_derivative) + 1
        epsilon = k_distances[elbow_idx]
        
        return epsilon, elbow_idx
    
    @staticmethod
    def optimize_eps_minpts_grid(
        X: np.ndarray,
        eps_range: Optional[np.ndarray] = None,
        minpts_range: Optional[np.ndarray] = None,
        K_desired: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[float, int, Dict]:
        """
        Grid search ile optimal (ε, minPts) bulunması.
        
        Objective: Maximum silhouette score
        Constraints:
          - K_min ≤ n_clusters ≤ K_max
          - outlier_ratio < 0.3
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        eps_range : Optional[np.ndarray], optional
            Test edilecek ε değerleri (None ise otomatik)
        minpts_range : Optional[np.ndarray], optional
            Test edilecek minPts değerleri (None ise otomatik)
        K_desired : Optional[int], optional
            İstenen cluster sayısı (None ise constraint yok)
        verbose : bool, optional
            İlerleme göster
            
        Returns
        -------
        Tuple[float, int, Dict]
            (optimal_eps, optimal_minpts, results_dict)
        """
        # Otomatik range belirleme
        if eps_range is None:
            k_dists = DBSCANOptimizer.compute_k_distances(X, k=5)
            eps_baseline, _ = DBSCANOptimizer.find_elbow_point(k_dists)
            eps_range = np.linspace(eps_baseline * 0.5, eps_baseline * 1.5, 10)
        
        if minpts_range is None:
            D = X.shape[1]
            minpts_range = np.arange(max(D + 1, 3), max(D + 1, 10))
        
        best_score = -1
        best_params = (eps_range[0], int(minpts_range[0]))
        results = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            for eps in eps_range:
                for minpts in minpts_range:
                    dbscan = DBSCAN(eps=eps, min_samples=int(minpts))
                    labels = dbscan.fit_predict(X)
                    
                    # Metrics
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    outlier_ratio = np.sum(labels == -1) / len(labels)
                    
                    # Constraints
                    if n_clusters < 2:
                        continue
                    
                    if outlier_ratio > 0.3:
                        continue
                    
                    if K_desired is not None:
                        if abs(n_clusters - K_desired) > 3:
                            continue
                    
                    # Silhouette score
                    try:
                        # Outlier'ları çıkar
                        mask = labels != -1
                        if np.sum(mask) < 2:
                            score = -1
                        else:
                            score = silhouette_score(X[mask], labels[mask])
                    except:
                        score = -1
                    
                    results.append({
                        'eps': float(eps),
                        'minpts': int(minpts),
                        'n_clusters': int(n_clusters),
                        'outlier_ratio': float(outlier_ratio),
                        'silhouette': float(score)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, int(minpts))
                    
                    if verbose and len(results) % 10 == 0:
                        print(f"  Tested {len(results)} combinations...")
        
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        return best_params[0], best_params[1], {
            'best_score': best_score,
            'all_results': results_df,
            'n_tested': len(results)
        }
    
    @staticmethod
    def hopkins_statistic(X: np.ndarray, sample_size: int = None) -> float:
        """
        Hopkins statistic ile clustering tendency'yi ölç.
        
        H ≈ 1 → data clusterable
        H ≈ 0.5 → random (uniform)
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        sample_size : int, optional
            Sample boyutu (None ise 0.1 * n)
            
        Returns
        -------
        float
            Hopkins statistic (0-1)
        """
        n = len(X)
        
        if sample_size is None:
            sample_size = min(int(0.1 * n), 100)
        
        sample_size = min(sample_size, n)
        
        # Random sample from data
        indices = np.random.choice(n, sample_size, replace=False)
        X_sample = X[indices]
        
        # Random uniform sample
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_random = np.random.uniform(X_min, X_max, (sample_size, X.shape[1]))
        
        # Distance to nearest neighbor
        u_dists = cdist(X_random, X).min(axis=1)
        w_dists = cdist(X_sample, np.delete(X, indices, axis=0)).min(axis=1)
        
        u_sum = u_dists.sum()
        w_sum = w_dists.sum()
        
        if u_sum + w_sum == 0:
            return 0.5
        
        H = u_sum / (u_sum + w_sum)
        
        return H
    
    @staticmethod
    def recommend_params(
        X: np.ndarray,
        method: str = 'elbow',
        K_desired: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Önerilen DBSCAN parametrelerini döndür.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        method : str, optional
            Metod: 'elbow', 'grid_search' (varsayılan: 'elbow')
        K_desired : Optional[int], optional
            Hedef cluster sayısı
            
        Returns
        -------
        Tuple[float, int]
            (recommended_eps, recommended_minpts)
        """
        D = X.shape[1]
        n = X.shape[0]
        
        if method == 'elbow':
            # K-distance analysis
            k = max(D + 1, 5)
            k_dists = DBSCANOptimizer.compute_k_distances(X, k=k)
            eps, _ = DBSCANOptimizer.find_elbow_point(k_dists)
            
            # minPts heuristic
            minpts = max(D + 1, int(np.log(n)))
            
            return eps, minpts
        
        elif method == 'grid_search':
            eps, minpts, _ = DBSCANOptimizer.optimize_eps_minpts_grid(
                X, K_desired=K_desired
            )
            return eps, minpts
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def visualize_k_distance_plot(
        X: np.ndarray,
        k: int = 5,
        output_file: Optional[str] = None
    ):
        """
        K-distance plot oluştur (elbow detection için).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        k : int, optional
            K value
        output_file : Optional[str], optional
            Çıktı dosyası
        """
        try:
            import matplotlib.pyplot as plt
            
            k_dists = DBSCANOptimizer.compute_k_distances(X, k=k)
            eps, elbow_idx = DBSCANOptimizer.find_elbow_point(k_dists)
            
            plt.figure(figsize=(10, 6))
            plt.plot(k_dists, 'b-', linewidth=2)
            plt.axhline(y=eps, color='r', linestyle='--', 
                       label=f'Recommended ε = {eps:.4f}')
            plt.scatter([elbow_idx], [eps], color='red', s=200, 
                       zorder=5, label='Elbow Point')
            
            plt.xlabel('Points sorted by distance', fontsize=12)
            plt.ylabel(f'{k}-distance', fontsize=12)
            plt.title(f'K-Distance Graph (k={k})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"[OK] K-distance plot kaydedildi: {output_file}")
            else:
                plt.show()
            
            plt.close()
        
        except ImportError:
            print("[WARN] Matplotlib bulunamadı, plot oluşturulamadı")


def auto_tune_dbscan(
    X: np.ndarray,
    K_desired: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    DBSCAN'i otomatik tune et ve sonuçları döndür.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    K_desired : Optional[int], optional
        Hedef cluster sayısı
    verbose : bool, optional
        İlerleme göster
        
    Returns
    -------
    Dict[str, any]
        Tuning sonuçları
    """
    if verbose:
        print("\n[DBSCAN AUTO-TUNING]")
        print("=" * 60)
    
    # Hopkins statistic
    H = DBSCANOptimizer.hopkins_statistic(X)
    
    if verbose:
        print(f"Hopkins Statistic: {H:.4f}")
        if H > 0.7:
            print("  ✅ Data clusterable görünüyor")
        elif H > 0.5:
            print("  ⚠️  Orta düzeyde clustering tendency")
        else:
            print("  ❌ Data uniform/random görünüyor")
    
    # Grid search
    if verbose:
        print("\nGrid search başlatılıyor...")
    
    eps, minpts, results = DBSCANOptimizer.optimize_eps_minpts_grid(
        X, K_desired=K_desired, verbose=verbose
    )
    
    # Test
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    outlier_ratio = np.sum(labels == -1) / len(labels)
    
    if verbose:
        print(f"\n[SONUÇ]")
        print(f"  Optimal ε: {eps:.4f}")
        print(f"  Optimal minPts: {minpts}")
        print(f"  Cluster sayısı: {n_clusters}")
        print(f"  Outlier oranı: {outlier_ratio*100:.1f}%")
        print(f"  Silhouette score: {results['best_score']:.4f}")
        print("=" * 60)
    
    return {
        'eps': eps,
        'minpts': minpts,
        'n_clusters': n_clusters,
        'outlier_ratio': outlier_ratio,
        'silhouette_score': results['best_score'],
        'hopkins_statistic': H,
        'is_clusterable': H > 0.6,
        'all_results': results['all_results']
    }

