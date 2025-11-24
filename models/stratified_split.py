"""
Stratified time series split modülü.

Bu modül, rejim-aware sampling ile train/val/test split yapar,
her rejimden proportional sample alarak test set coverage'ını garantiler.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import DBSCAN


class StratifiedTimeSeriesSplit:
    """
    Rejim-aware time series splitting.
    
    Geleneksel temporal split yerine, her rejimden proportional
    sample alarak daha representative test seti oluşturur.
    
    Attributes
    ----------
    train_ratio : float
        Train set oranı
    val_ratio : float
        Validation set oranı
    test_ratio : float
        Test set oranı
    preserve_temporal_order : bool
        Temporal order korunsun mu
    """
    
    def __init__(
        self,
        train_ratio: float = 0.50,
        val_ratio: float = 0.15,
        test_ratio: float = 0.35,
        preserve_temporal_order: bool = True
    ):
        """
        StratifiedTimeSeriesSplit başlatıcı.
        
        Parameters
        ----------
        train_ratio : float, optional
            Train set oranı (varsayılan: 0.50)
        val_ratio : float, optional
            Validation set oranı (varsayılan: 0.15)
        test_ratio : float, optional
            Test set oranı (varsayılan: 0.35)
        preserve_temporal_order : bool, optional
            Temporal order korunsun mu (varsayılan: True)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Oranlar toplamı 1.0 olmalı")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.preserve_temporal_order = preserve_temporal_order
        
        self.regime_labels_ = None
        self.split_indices_ = None
    
    def fit_split(
        self,
        data: pd.Series,
        regime_labels: np.ndarray
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Rejim-aware split yap.
        
        Her rejimden proportional sample alarak train/val/test oluştur.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        regime_labels : np.ndarray
            Rejim etiketleri (-1: outlier)
            
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (train_data, val_data, test_data)
        """
        self.regime_labels_ = regime_labels
        n = len(data)
        
        if len(regime_labels) != n:
            raise ValueError("Data ve regime_labels uzunlukları eşit olmalı")
        
        # Outlier'ları filtrele
        valid_mask = regime_labels != -1
        valid_indices = np.where(valid_mask)[0]
        valid_labels = regime_labels[valid_mask]
        
        # Her rejim için indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        unique_regimes = np.unique(valid_labels)
        
        for regime in unique_regimes:
            regime_idx = valid_indices[valid_labels == regime]
            n_regime = len(regime_idx)
            
            if n_regime < 10:
                # Çok az sample → tümünü train'e at
                train_indices.extend(regime_idx.tolist())
                continue
            
            # Calculate splits
            n_train = int(n_regime * self.train_ratio)
            n_val = int(n_regime * self.val_ratio)
            n_test = n_regime - n_train - n_val
            
            if self.preserve_temporal_order:
                # Temporal order: ilk → train, orta → val, son → test
                train_indices.extend(regime_idx[:n_train].tolist())
                val_indices.extend(regime_idx[n_train:n_train + n_val].tolist())
                test_indices.extend(regime_idx[n_train + n_val:].tolist())
            else:
                # Random split (shuffle)
                np.random.shuffle(regime_idx)
                train_indices.extend(regime_idx[:n_train].tolist())
                val_indices.extend(regime_idx[n_train:n_train + n_val].tolist())
                test_indices.extend(regime_idx[n_train + n_val:].tolist())
        
        # Outlier'ları train'e ekle
        outlier_indices = np.where(regime_labels == -1)[0]
        train_indices.extend(outlier_indices.tolist())
        
        # Sort (temporal order için)
        if self.preserve_temporal_order:
            train_indices = sorted(train_indices)
            val_indices = sorted(val_indices)
            test_indices = sorted(test_indices)
        
        self.split_indices_ = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        # Create data splits
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        test_data = data.iloc[test_indices]
        
        return train_data, val_data, test_data
    
    def get_regime_distribution(self) -> Dict[str, Dict[int, int]]:
        """
        Her split'teki rejim dağılımını döndür.
        
        Returns
        -------
        Dict[str, Dict[int, int]]
            {'train': {regime_id: count, ...}, ...}
        """
        if self.split_indices_ is None or self.regime_labels_ is None:
            raise ValueError("Önce fit_split() çağırın")
        
        distribution = {}
        
        for split_name, indices in self.split_indices_.items():
            labels = self.regime_labels_[indices]
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            distribution[split_name] = dict(zip(unique.tolist(), counts.tolist()))
        
        return distribution
    
    def validate_coverage(
        self,
        min_regime_samples: int = 10
    ) -> Tuple[bool, str]:
        """
        Test set coverage'ını validate et.
        
        Parameters
        ----------
        min_regime_samples : int, optional
            Her rejimde minimum sample sayısı
            
        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        if self.split_indices_ is None:
            raise ValueError("Önce fit_split() çağırın")
        
        distribution = self.get_regime_distribution()
        
        train_regimes = set(distribution['train'].keys())
        test_regimes = set(distribution['test'].keys())
        
        # Coverage ratio
        coverage = len(test_regimes) / len(train_regimes) if train_regimes else 0
        
        # Eksik rejimler
        missing_regimes = train_regimes - test_regimes
        
        # Yetersiz sample'lı rejimler
        insufficient_regimes = [
            (r, count) for r, count in distribution['test'].items()
            if count < min_regime_samples
        ]
        
        issues = []
        
        if coverage < 0.5:
            issues.append(
                f"Coverage yetersiz: {coverage:.1%} "
                f"(Test'te {len(test_regimes)}/{len(train_regimes)} rejim)"
            )
        
        if missing_regimes:
            issues.append(f"Eksik rejimler: {sorted(missing_regimes)}")
        
        if insufficient_regimes:
            issues.append(
                f"Yetersiz sample: {insufficient_regimes}"
            )
        
        is_valid = len(issues) == 0
        
        if is_valid:
            message = (
                f"✅ Test coverage yeterli: "
                f"{len(test_regimes)}/{len(train_regimes)} rejim "
                f"({coverage:.1%})"
            )
        else:
            message = "⚠️ Coverage sorunları:\n  " + "\n  ".join(issues)
        
        return is_valid, message
    
    def generate_report(self) -> str:
        """
        Detaylı split raporu oluştur.
        
        Returns
        -------
        str
            Rapor metni
        """
        if self.split_indices_ is None:
            raise ValueError("Önce fit_split() çağırın")
        
        distribution = self.get_regime_distribution()
        
        lines = []
        lines.append("=" * 80)
        lines.append("STRATIFIED TIME SERIES SPLIT RAPORU")
        lines.append("=" * 80)
        lines.append("")
        
        # Genel bilgi
        lines.append("GENEL BİLGİ")
        lines.append("-" * 80)
        for split_name, indices in self.split_indices_.items():
            n = len(indices)
            ratio = n / sum(len(idx) for idx in self.split_indices_.values())
            lines.append(f"  {split_name.capitalize()}: {n} samples ({ratio:.1%})")
        lines.append("")
        
        # Rejim dağılımları
        lines.append("REJİM DAĞILIMLARI")
        lines.append("-" * 80)
        
        all_regimes = set()
        for dist in distribution.values():
            all_regimes.update(dist.keys())
        
        lines.append(f"{'Regime':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
        lines.append("-" * 40)
        
        for regime in sorted(all_regimes):
            train_count = distribution['train'].get(regime, 0)
            val_count = distribution['val'].get(regime, 0)
            test_count = distribution['test'].get(regime, 0)
            lines.append(
                f"{regime:<10} {train_count:<10} {val_count:<10} {test_count:<10}"
            )
        
        lines.append("")
        
        # Validation
        is_valid, validation_msg = self.validate_coverage()
        lines.append("VALIDATION")
        lines.append("-" * 80)
        lines.append(validation_msg)
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def compare_split_strategies(
    data: pd.Series,
    regime_labels: np.ndarray,
    strategies: List[Dict]
) -> pd.DataFrame:
    """
    Farklı split stratejilerini karşılaştır.
    
    Parameters
    ----------
    data : pd.Series
        Time series data
    regime_labels : np.ndarray
        Rejim etiketleri
    strategies : List[Dict]
        Her bir strateji: {'name': str, 'params': dict}
        
    Returns
    -------
    pd.DataFrame
        Karşılaştırma tablosu
    """
    results = []
    
    for strategy in strategies:
        name = strategy['name']
        params = strategy.get('params', {})
        
        splitter = StratifiedTimeSeriesSplit(**params)
        train, val, test = splitter.fit_split(data, regime_labels)
        
        distribution = splitter.get_regime_distribution()
        is_valid, _ = splitter.validate_coverage()
        
        results.append({
            'Strategy': name,
            'Train_Size': len(train),
            'Val_Size': len(val),
            'Test_Size': len(test),
            'Test_Regimes': len(distribution['test']),
            'Train_Regimes': len(distribution['train']),
            'Coverage': len(distribution['test']) / len(distribution['train']),
            'Valid': is_valid
        })
    
    return pd.DataFrame(results)

