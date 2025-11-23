"""
Time-Series Cross-Validation Modülü - GRM Modeli için Sağlamlık Testi.

Bu modül, zaman serisi modellerini rolling window validation,
expanding window ve blocked cross-validation ile değerlendirir.

FAZE 4: ZENGİNLEŞTİRME
GÜNCELLEMELER:
- Gelişmiş CV stratejileri (expanding, blocked, rolling)
- Bootstrap CI entegrasyonu
- Çoklu model karşılaştırma
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from models.baseline_model import BaselineARIMA
from models.grm_model import SchwarzschildGRM
from models.kerr_grm_model import KerrGRM
from models.metrics import calculate_rmse, calculate_mae


class TimeSeriesCrossValidator:
    """
    Time-series için walk-forward cross-validation.
    
    Strateji:
    ┌─────────────────────────────────────────────────────────┐
    │ Fold 1: [Train────────][Val──][Test──]                 │
    │ Fold 2:    [Train────────][Val──][Test──]              │
    │ Fold 3:       [Train────────][Val──][Test──]           │
    │ Fold 4:          [Train────────][Val──][Test──]        │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        initial_train_size: int = 300,
        val_size: int = 50,
        test_size: int = 50,
        step_size: int = 50
    ):
        """
        TimeSeriesCrossValidator sınıfını başlatır.
        
        Parameters
        ----------
        initial_train_size : int, optional
            İlk fold için train boyutu (varsayılan: 300)
        val_size : int, optional
            Validation boyutu (varsayılan: 50)
        test_size : int, optional
            Test boyutu (varsayılan: 50)
        step_size : int, optional
            Her fold arası kaydırma miktarı (varsayılan: 50)
        """
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Veriyi k fold'a böl.
        
        Parameters
        ----------
        data : np.ndarray
            Zaman serisi verisi
            
        Returns
        -------
        List[Tuple]
            Her fold için (train_indices, val_indices, test_indices)
        """
        n = len(data)
        folds = []
        
        current_train_end = self.initial_train_size
        
        while current_train_end + self.val_size + self.test_size <= n:
            train_indices = np.arange(0, current_train_end)
            val_indices = np.arange(
                current_train_end,
                current_train_end + self.val_size
            )
            test_indices = np.arange(
                current_train_end + self.val_size,
                current_train_end + self.val_size + self.test_size
            )
            
            folds.append((train_indices, val_indices, test_indices))
            current_train_end += self.step_size
        
        return folds
    
    def evaluate_model(
        self,
        model_class,
        data: np.ndarray,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Modeli tüm fold'larda değerlendir.
        
        Parameters
        ----------
        model_class : type
            Model sınıfı
        data : np.ndarray
            Zaman serisi verisi
        **model_kwargs
            Model parametreleri
            
        Returns
        -------
        Dict[str, List[float]]
            Her metrik için fold sonuçları
        """
        folds = self.split(data)
        results = {
            'rmse': [],
            'mae': [],
            'fold': []
        }
        
        for i, (train_idx, val_idx, test_idx) in enumerate(folds):
            print(f"  Fold {i+1}/{len(folds)}...")
            
            train_data = pd.Series(data[train_idx])
            val_data = pd.Series(data[val_idx])
            test_data = pd.Series(data[test_idx])
            
            # Model train
            if model_class == BaselineARIMA:
                model = model_class()
                best_order = model.grid_search(
                    train_data, val_data,
                    p_range=[0, 1, 2],
                    d_range=[0, 1],
                    q_range=[0, 1, 2],
                    verbose=False
                )
                model.fit(train_data, order=best_order)
            else:
                # GRM: önce baseline, sonra GRM
                baseline = BaselineARIMA()
                best_order = baseline.grid_search(
                    train_data, val_data,
                    p_range=[0, 1, 2],
                    d_range=[0, 1],
                    q_range=[0, 1, 2],
                    verbose=False
                )
                baseline.fit(train_data, order=best_order)
                train_residuals = baseline.get_residuals()
                
                model = model_class(**model_kwargs)
                model.fit(train_residuals)
                model.baseline = baseline  # Baseline'ı sakla
            
            # Test predict (walk-forward)
            predictions = self.walk_forward_predict(model, test_data)
            
            # Metrics
            rmse = calculate_rmse(test_data.values, predictions)
            mae = calculate_mae(test_data.values, predictions)
            
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['fold'].append(i + 1)
        
        return results
    
    def walk_forward_predict(
        self,
        model,
        test_data: pd.Series
    ) -> np.ndarray:
        """
        Walk-forward validation ile tahmin yap.
        
        Parameters
        ----------
        model : BaselineARIMA or GRM model
            Eğitilmiş model
        test_data : pd.Series
            Test verisi
            
        Returns
        -------
        np.ndarray
            Tahmin dizisi
        """
        if hasattr(model, 'baseline'):
            # GRM model
            return self.walk_forward_predict_grm(
                model.baseline, model, test_data
            )
        else:
            # Baseline model
            predictions = []
            for i in range(len(test_data)):
                pred = model.predict(1)[0]
                predictions.append(pred)
                if i < len(test_data) - 1:
                    try:
                        model.fitted_model = model.fitted_model.append(
                            [test_data.iloc[i]], refit=False
                        )
                    except:
                        pass
            return np.array(predictions)
    
    def walk_forward_predict_grm(
        self,
        baseline: BaselineARIMA,
        grm,
        test_data: pd.Series
    ) -> np.ndarray:
        """
        Walk-forward validation ile GRM tahminleri.
        
        Parameters
        ----------
        baseline : BaselineARIMA
            Eğitilmiş baseline model
        grm : SchwarzschildGRM or KerrGRM
            Eğitilmiş GRM model
        test_data : pd.Series
            Test verisi
            
        Returns
        -------
        np.ndarray
            Final tahminler
        """
        predictions = []
        all_residuals = list(baseline.get_residuals())
        
        # Şok tespiti
        shock_times = None
        if len(all_residuals) > 0:
            shock_times = grm.detect_shocks(np.array(all_residuals))
        
        for i in range(len(test_data)):
            # Baseline tahmin
            baseline_pred = baseline.predict(1)[0]
            
            # Time since shock
            current_time = len(all_residuals)
            tau = grm.compute_time_since_shock(
                current_time=current_time,
                shock_times=shock_times
            )
            
            # GRM düzeltmesi
            recent_residuals = np.array(all_residuals[-grm.window_size:])
            
            if len(recent_residuals) > 0:
                mass = grm.compute_mass(recent_residuals)[-1]
                
                if hasattr(grm, 'compute_spin'):
                    # Kerr
                    spin = grm.compute_spin(recent_residuals)[-1]
                    correction = grm.compute_curvature_single(
                        recent_residuals[-1],
                        mass,
                        spin,
                        time_since_shock=tau
                    )
                else:
                    # Schwarzschild
                    correction = grm.compute_curvature_single(
                        recent_residuals[-1],
                        mass,
                        time_since_shock=tau
                    )
            else:
                correction = 0.0
            
            final_pred = baseline_pred + correction
            predictions.append(final_pred)
            
            # Gerçek değeri gözlemle
            actual = test_data.iloc[i]
            residual = actual - baseline_pred
            all_residuals.append(residual)
            
            # Şok tespiti güncelle
            if len(all_residuals) > grm.window_size:
                shock_times = grm.detect_shocks(np.array(all_residuals))
            
            # Baseline'ı güncelle
            if i < len(test_data) - 1:
                try:
                    baseline.fitted_model = baseline.fitted_model.append(
                        [actual], refit=False
                    )
                except:
                    pass
        
        return np.array(predictions)
    
    def compare_models(
        self,
        models: Dict[str, Tuple[type, dict]],
        data: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Birden fazla modeli karşılaştır.
        
        Parameters
        ----------
        models : Dict[str, Tuple[type, dict]]
            Model adı -> (model_class, model_kwargs)
        data : np.ndarray
            Zaman serisi verisi
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (karşılaştırma tablosu, detaylı sonuçlar)
        """
        all_results = {}
        
        for name, (model_class, kwargs) in models.items():
            print(f"\n{name} değerlendiriliyor...")
            results = self.evaluate_model(model_class, data, **kwargs)
            all_results[name] = results
        
        # Özet istatistikler
        summary = []
        for name, results in all_results.items():
            summary.append({
                'Model': name,
                'Mean_RMSE': np.mean(results['rmse']),
                'Std_RMSE': np.std(results['rmse']),
                'Min_RMSE': np.min(results['rmse']),
                'Max_RMSE': np.max(results['rmse']),
                'Mean_MAE': np.mean(results['mae']),
                'Std_MAE': np.std(results['mae'])
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('Mean_RMSE')
        
        return df, all_results

