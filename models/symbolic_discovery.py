"""
Symbolic Regression Modülü - Dinamik Keşfi.

Bu modül, veriden optimal bükülme fonksiyonunu otomatik keşfeder.

FAZE 5: PIML TEMEL ENTEGRASYONU
"""

import numpy as np
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# PySR import (optional, check if available)
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("[UYARI] PySR kurulu değil. Symbolic regression kullanılamaz.")
    print("[UYARI] Kurulum: pip install pysr")


class SymbolicGRM:
    """
    Symbolic regression ile bükülme fonksiyonunu keşfet.
    
    PySR kullanarak en iyi sembolik denklemi bul:
    Γ(t) = f(M(t), a(t), τ(t), ε(t))
    """
    
    def __init__(
        self,
        niterations: int = 100,
        binary_operators: Optional[list] = None,
        unary_operators: Optional[list] = None,
        maxsize: int = 20,
        populations: int = 15
    ):
        """
        SymbolicGRM sınıfını başlatır.
        
        Parameters
        ----------
        niterations : int, optional
            İterasyon sayısı (varsayılan: 100)
        binary_operators : list, optional
            İkili operatörler, None ise varsayılan kullanılır
        unary_operators : list, optional
            Tekli operatörler, None ise varsayılan kullanılır
        maxsize : int, optional
            Maksimum formül boyutu (varsayılan: 20)
        populations : int, optional
            Popülasyon sayısı (varsayılan: 15)
        """
        if not PYSR_AVAILABLE:
            raise ImportError(
                "PySR kurulu değil. Lütfen 'pip install pysr' komutu ile kurun."
            )
        
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        
        if unary_operators is None:
            unary_operators = ["exp", "log", "sqrt", "tanh", "abs"]
        
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            model_selection="best",
            loss="loss(x, y) = (x - y)^2",
            maxsize=maxsize,
            populations=populations
        )
        
        self.best_formula = None
        self.r2_score = None
    
    def prepare_features(
        self,
        residuals: np.ndarray,
        window_size: int = 20,
        shock_threshold_quantile: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GRM feature'larını hazırla.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        window_size : int, optional
            Pencere boyutu (varsayılan: 20)
        shock_threshold_quantile : float, optional
            Şok eşiği quantile (varsayılan: 0.95)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X, y) - Features ve hedefler
        """
        n = len(residuals)
        X = []
        y = []
        
        # Şok eşiği
        abs_residuals = np.abs(residuals)
        shock_threshold = np.quantile(abs_residuals, shock_threshold_quantile)
        
        for t in range(window_size, n - 1):
            window = residuals[t - window_size:t]
            
            # Features
            mass = np.var(window)
            
            # Spin (otokorelasyon)
            if len(window) > 1 and np.std(window) > 1e-8:
                spin = np.corrcoef(window[1:], window[:-1])[0, 1]
                spin = np.clip(spin, -1, 1)
            else:
                spin = 0.0
            
            # Tau (time since last shock)
            abs_res = np.abs(residuals[:t])
            shock_indices = np.where(abs_res > shock_threshold)[0]
            if len(shock_indices) == 0:
                tau = float(len(residuals[:t]))
            else:
                last_shock = shock_indices[-1]
                tau = float(len(residuals[:t]) - last_shock)
            
            epsilon = residuals[t]
            
            X.append([mass, spin, tau, epsilon])
            
            # Target: next residual or ideal correction
            y.append(residuals[t + 1])
        
        return np.array(X), np.array(y)
    
    def discover_formula(
        self,
        residuals: np.ndarray,
        window_size: int = 20,
        verbose: bool = True
    ) -> str:
        """
        Sembolik denklemi keşfet.
        
        Parameters
        ----------
        residuals : np.ndarray
            Artık dizisi
        window_size : int, optional
            Pencere boyutu (varsayılan: 20)
        verbose : bool, optional
            İlerleme göster (varsayılan: True)
            
        Returns
        -------
        str
            En iyi sembolik denklem
        """
        # Feature hazırlama
        X, y = self.prepare_features(residuals, window_size)
        
        # Feature isimleri
        feature_names = ["M", "a", "tau", "epsilon"]
        
        if verbose:
            print("\n" + "=" * 80)
            print("SEMBOLİK REGRESYON BAŞLATILIYOR")
            print("=" * 80)
            print(f"Örnek sayısı: {len(X)}")
            print(f"Feature sayısı: {X.shape[1]}")
            print("(Bu işlem 10-30 dakika sürebilir)")
            print("=" * 80 + "\n")
        
        # Symbolic regression
        self.model.fit(X, y, variable_names=feature_names)
        
        # En iyi formül
        self.best_formula = self.model.get_best()
        self.r2_score = self.model.score(X, y)
        
        if verbose:
            print("\n" + "=" * 80)
            print("KEŞFEDİLEN FORMÜL")
            print("=" * 80)
            print(f"Γ(t) = {self.best_formula}")
            print(f"R² Score: {self.r2_score:.4f}")
            print("=" * 80)
            
            # Tüm adayları göster
            if hasattr(self.model, 'equations_'):
                print("\nTÜM ADAY FORMÜLLER (Complexity vs Accuracy):")
                print(self.model.equations_)
        
        return self.best_formula
    
    def predict(
        self,
        M: np.ndarray,
        a: np.ndarray,
        tau: np.ndarray,
        epsilon: np.ndarray
    ) -> np.ndarray:
        """
        Keşfedilen formülü kullanarak tahmin yap.
        
        Parameters
        ----------
        M : np.ndarray
            Kütle dizisi
        a : np.ndarray
            Dönme dizisi
        tau : np.ndarray
            Time since shock dizisi
        epsilon : np.ndarray
            Artık dizisi
            
        Returns
        -------
        np.ndarray
            Tahminler
        """
        if self.best_formula is None:
            raise ValueError(
                "Formül henüz keşfedilmedi. Önce discover_formula() çağırın."
            )
        
        X = np.column_stack([M, a, tau, epsilon])
        return self.model.predict(X)
    
    def get_formula_info(self) -> dict:
        """
        Keşfedilen formül hakkında bilgi döndürür.
        
        Returns
        -------
        dict
            Formül bilgileri
        """
        if self.best_formula is None:
            return {
                'formula': None,
                'r2_score': None,
                'status': 'not_discovered'
            }
        
        info = {
            'formula': self.best_formula,
            'r2_score': self.r2_score,
            'status': 'discovered'
        }
        
        if hasattr(self.model, 'equations_'):
            info['all_equations'] = self.model.equations_
        
        return info

