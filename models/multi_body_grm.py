"""
Multi-Body GRM - Multiple Gravitational Anomalies.

Bu modül, birden fazla "black hole" (rejim) modelleyen sistemdir.
Her rejim, farklı bir GRM parametresi seti ile temsil edilir.

FAZE 6: PIML İLERİ SEVİYE
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Local imports
from models.grm_model import SchwarzschildGRM


class MultiBodyGRM:
    """
    Multi-Body GRM Model.
    
    Bu model, rezidüelleri farklı rejimlere (regime) ayırır ve
    her rejim için ayrı bir GRM modeli eğitir. Tahmin sırasında,
    mevcut rejime göre uygun GRM modeli kullanılır.
    
    Yaklaşım:
        1. Rezidüelleri DBSCAN ile cluster'la (rejim tespiti)
        2. Her rejim için ayrı SchwarzschildGRM eğit
        3. Tahmin sırasında, mevcut rejime göre weighted sum
    
    Attributes
    ----------
    n_bodies : int
        Maksimum body (rejim) sayısı
    window_size : int
        Pencere boyutu
    body_params : List[Dict]
        Her body için parametreler
    clusterer : DBSCAN
        Rejim tespiti için clusterer
    """
    
    def __init__(
        self,
        window_size: int = 20,
        eps: float = 0.5,
        min_samples: int = 10,
        use_decay: bool = True
    ):
        """
        MultiBodyGRM sınıfını başlatır.
        
        Parameters
        ----------
        window_size : int, optional
            Pencere boyutu (varsayılan: 20)
        eps : float, optional
            DBSCAN eps parametresi (varsayılan: 0.5)
        min_samples : int, optional
            DBSCAN min_samples parametresi (varsayılan: 10)
        use_decay : bool, optional
            Decay factor kullan (varsayılan: True)
        """
        self.window_size = window_size
        self.eps = eps
        self.min_samples = min_samples
        self.use_decay = use_decay
        self.body_params: List[Dict] = []
        self.clusterer: Optional[DBSCAN] = None
        self.regime_labels: Optional[np.ndarray] = None
    
    def compute_autocorr(
        self,
        window: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Otokorelasyon hesapla.
        
        Parameters
        ----------
        window : np.ndarray
            Pencere verisi
        lag : int, optional
            Lag değeri (varsayılan: 1)
            
        Returns
        -------
        float
            Otokorelasyon değeri
        """
        if len(window) <= lag:
            return 0.0
        
        x_t = window[:-lag]
        x_t_lag = window[lag:]
        
        if np.std(x_t) < 1e-8 or np.std(x_t_lag) < 1e-8:
            return 0.0
        
        corr = np.corrcoef(x_t, x_t_lag)[0, 1]
        
        if np.isnan(corr):
            return 0.0
        
        return float(np.clip(corr, -1.0, 1.0))
    
    def cluster_residuals(
        self,
        residuals: np.ndarray
    ) -> np.ndarray:
        """
        Rezidüelleri cluster'la (rejim tespiti).
        
        Her pencere için feature'lar:
        - mean, std, max, min, autocorr
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
            
        Returns
        -------
        np.ndarray
            Rejim etiketleri (her pencere için)
        """
        features = []
        
        for t in range(self.window_size, len(residuals)):
            window = residuals[t - self.window_size:t]
            
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                self.compute_autocorr(window)
            ])
        
        features = np.array(features)
        
        # Normalize features
        if len(features) > 0:
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0) + 1e-8
            features = (features - feature_mean) / feature_std
        
        # DBSCAN clustering
        self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        regime_labels = self.clusterer.fit_predict(features)
        
        # Pencere öncesi için -1 (noise) ata
        full_labels = np.full(len(residuals), -1, dtype=int)
        full_labels[self.window_size:] = regime_labels
        
        return full_labels
    
    def fit(self, residuals: np.ndarray) -> None:
        """
        Multi-body GRM modelini eğit.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        """
        # Rejim tespiti
        self.regime_labels = self.cluster_residuals(residuals)
        unique_labels = np.unique(self.regime_labels)
        
        # Noise label'ı (-1) çıkar
        unique_labels = unique_labels[unique_labels != -1]
        
        print(f"[MultiBodyGRM] {len(unique_labels)} rejim tespit edildi")
        
        # Her rejim için ayrı GRM eğit
        self.body_params = []
        
        for regime_id in unique_labels:
            regime_mask = self.regime_labels == regime_id
            regime_indices = np.where(regime_mask)[0]
            
            # Pencere boyutu kadar önceki verileri de al
            regime_residuals = []
            for idx in regime_indices:
                if idx >= self.window_size:
                    window = residuals[idx - self.window_size:idx + 1]
                    regime_residuals.extend(window)
            
            if len(regime_residuals) < self.window_size * 2:
                continue
            
            regime_residuals = np.array(regime_residuals)
            
            # Her rejim için ayrı GRM fit
            grm = SchwarzschildGRM(
                window_size=self.window_size,
                use_decay=self.use_decay
            )
            grm.fit(regime_residuals)
            
            self.body_params.append({
                'body_id': int(regime_id),
                'alpha': grm.alpha,
                'beta': grm.beta if self.use_decay else None,
                'n_samples': len(regime_residuals),
                'grm_model': grm  # Full model'i sakla
            })
            
            beta_str = f"{grm.beta:.4f}" if self.use_decay else "N/A"
            print(f"  Rejim {regime_id}: α={grm.alpha:.4f}, "
                  f"β={beta_str}, "
                  f"n={len(regime_residuals)}")
        
        print(f"[MultiBodyGRM] {len(self.body_params)} body eğitildi\n")
    
    def predict_regime(
        self,
        residuals: np.ndarray,
        current_time: int
    ) -> int:
        """
        Mevcut rejimi tahmin et.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        current_time : int
            Mevcut zaman indeksi
            
        Returns
        -------
        int
            Tahmin edilen rejim ID
        """
        if current_time < self.window_size:
            # İlk pencere için, en yakın rejimi bul
            if self.regime_labels is not None and len(self.regime_labels) > 0:
                # İlk geçerli rejimi kullan
                valid_labels = self.regime_labels[self.regime_labels != -1]
                if len(valid_labels) > 0:
                    return int(valid_labels[0])
            return 0
        
        # Mevcut pencere için feature hesapla
        window = residuals[current_time - self.window_size:current_time]
        
        # Boş pencere kontrolü
        if len(window) == 0:
            return 0
        
        # NaN içeren değerleri filtrele
        window_clean = window[~np.isnan(window)]
        if len(window_clean) == 0:
            return 0
        
        features = np.array([
            np.mean(window_clean),
            np.std(window_clean) if len(window_clean) > 1 else 0.0,
            np.max(window_clean),
            np.min(window_clean),
            self.compute_autocorr(window_clean)
        ])
        
        # Normalize
        if self.clusterer is not None:
            # DBSCAN'ın normalize ettiği feature'ları kullan
            # Basit yaklaşım: en yakın rejimi bul
            if self.regime_labels is not None:
                # Son görülen rejimi kullan
                recent_labels = self.regime_labels[
                    max(0, current_time - 10):current_time
                ]
                valid_labels = recent_labels[recent_labels != -1]
                if len(valid_labels) > 0:
                    # En sık görülen rejim
                    unique, counts = np.unique(valid_labels, return_counts=True)
                    return int(unique[np.argmax(counts)])
        
        return 0
    
    def compute_curvature(
        self,
        residuals: np.ndarray,
        current_time: int,
        current_regime: Optional[int] = None
    ) -> float:
        """
        Multi-body curvature hesapla.
        
        Her body'nin katkısı, mevcut rejime göre weighted sum.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        current_time : int
            Mevcut zaman indeksi
        current_regime : int, optional
            Mevcut rejim, None ise tahmin edilir
            
        Returns
        -------
        float
            Toplam bükülme
        """
        if len(self.body_params) == 0:
            return 0.0
        
        if current_time < self.window_size:
            return 0.0
        
        # Mevcut rejimi tahmin et
        if current_regime is None:
            current_regime = self.predict_regime(residuals, current_time)
        
        # Son pencere
        recent_residuals = residuals[current_time - self.window_size:current_time]
        
        if len(recent_residuals) == 0:
            return 0.0
        
        # NaN temizle
        recent_residuals = recent_residuals[~np.isnan(recent_residuals)]
        if len(recent_residuals) == 0:
            return 0.0
        
        total_curvature = 0.0
        
        for params in self.body_params:
            body_id = params['body_id']
            grm_model = params['grm_model']
            
            # Weight: mevcut rejim ise 1.0, değilse 0.1
            if body_id == current_regime:
                weight = 1.0
            else:
                weight = 0.1
            
            # Bu body'nin katkısı
            mass_arr = grm_model.compute_mass(recent_residuals)
            if len(mass_arr) == 0:
                continue
                
            mass = mass_arr[-1]
            last_residual = recent_residuals[-1]
            
            # NaN kontrolü
            if np.isnan(mass) or np.isnan(last_residual):
                continue
            
            # Basit curvature (Schwarzschild)
            gamma_i = grm_model.alpha * mass * np.sign(last_residual)
            
            # Decay factor (eğer kullanılıyorsa)
            if self.use_decay and params['beta'] is not None:
                # Time since shock hesapla
                tau = 5.0  # Simplified
                decay = 1.0 / (1.0 + params['beta'] * tau)
                gamma_i *= decay
            
            # NaN kontrolü
            if not np.isnan(gamma_i):
                total_curvature += weight * gamma_i
        
        return float(total_curvature)
    
    def predict(
        self,
        residuals: np.ndarray,
        current_time: int,
        baseline_pred: float
    ) -> Tuple[float, float, int]:
        """
        Tahmin yap.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        current_time : int
            Mevcut zaman indeksi
        baseline_pred : float
            Baseline tahmin
            
        Returns
        -------
        Tuple[float, float, float]
            (baseline_pred, grm_correction, final_pred, regime_id)
        """
        # Rejim tahmini
        regime_id = self.predict_regime(residuals, current_time)
        
        # GRM correction
        grm_correction = self.compute_curvature(
            residuals, current_time, current_regime=regime_id
        )
        
        # Final prediction
        final_pred = baseline_pred + grm_correction
        
        return baseline_pred, grm_correction, final_pred, regime_id

