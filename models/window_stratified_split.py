"""Window-based stratified time series split module.

This module provides an improved stratified split that preserves regime
distribution while maintaining temporal order. Unlike the standard stratified
split, this uses time windows to avoid sample loss in minority regimes.

PEP8 compliant | PEP257 compliant
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy.stats import entropy


class WindowStratifiedSplit:
    """Window-based stratified time series splitter.
    
    This class implements a time window-based approach to stratified splitting,
    ensuring that minority regimes are adequately represented in train, val,
    and test sets.
    
    The algorithm:
    1. Divides the time series into fixed-size windows
    2. Computes regime distribution and entropy for each window
    3. Allocates windows to train/val/test based on diversity
    4. Preserves temporal order within each split
    
    Attributes
    ----------
    window_size : int
        Size of each time window (default: 30 days).
    train_ratio : float
        Proportion of data for training (default: 0.60).
    val_ratio : float
        Proportion of data for validation (default: 0.15).
    test_ratio : float
        Proportion of data for testing (default: 0.25).
    preserve_diversity : bool
        If True, prioritize high-entropy windows for train set.
    
    Examples
    --------
    >>> splitter = WindowStratifiedSplit(window_size=30)
    >>> train, val, test = splitter.split(data, regime_labels)
    >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    
    def __init__(
        self,
        window_size: int = 30,
        train_ratio: float = 0.60,
        val_ratio: float = 0.15,
        test_ratio: float = 0.25,
        preserve_diversity: bool = True
    ):
        """Initialize window-based stratified splitter.
        
        Parameters
        ----------
        window_size : int, optional
            Number of days per window, by default 30.
        train_ratio : float, optional
            Training set proportion, by default 0.60.
        val_ratio : float, optional
            Validation set proportion, by default 0.15.
        test_ratio : float, optional
            Test set proportion, by default 0.25.
        preserve_diversity : bool, optional
            Prioritize diverse windows for training, by default True.
        
        Raises
        ------
        ValueError
            If ratios don't sum to 1.0 or window_size < 1.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )
        
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.preserve_diversity = preserve_diversity
        
        self.windows_: List[Dict] = []
        self.train_windows_: List[int] = []
        self.val_windows_: List[int] = []
        self.test_windows_: List[int] = []
    
    def _compute_window_metrics(
        self,
        regime_labels: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> Dict:
        """Compute metrics for a single time window.
        
        Parameters
        ----------
        regime_labels : np.ndarray
            Regime labels for entire dataset.
        start_idx : int
            Window start index.
        end_idx : int
            Window end index (exclusive).
        
        Returns
        -------
        dict
            Window metrics including entropy, regime counts, and indices.
        """
        window_regimes = regime_labels[start_idx:end_idx]
        
        # Filter out outliers (-1)
        valid_regimes = window_regimes[window_regimes != -1]
        
        if len(valid_regimes) == 0:
            # Empty window (all outliers)
            return {
                'start': start_idx,
                'end': end_idx,
                'entropy': 0.0,
                'regime_counts': {},
                'n_regimes': 0,
                'outlier_ratio': 1.0
            }
        
        # Compute regime distribution
        unique_regimes, counts = np.unique(valid_regimes, return_counts=True)
        regime_counts = dict(zip(unique_regimes, counts))
        
        # Compute Shannon entropy (diversity measure)
        probs = counts / counts.sum()
        shannon_entropy = entropy(probs, base=2)
        
        # Outlier ratio
        outlier_ratio = (window_regimes == -1).sum() / len(window_regimes)
        
        return {
            'start': start_idx,
            'end': end_idx,
            'entropy': shannon_entropy,
            'regime_counts': regime_counts,
            'n_regimes': len(unique_regimes),
            'outlier_ratio': outlier_ratio
        }
    
    def _create_windows(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> List[Dict]:
        """Divide time series into windows and compute metrics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data.
        regime_labels : np.ndarray
            Regime labels corresponding to data.
        
        Returns
        -------
        list of dict
            List of window metrics.
        """
        n = len(data)
        n_windows = (n + self.window_size - 1) // self.window_size
        
        windows = []
        for i in range(n_windows):
            start = i * self.window_size
            end = min((i + 1) * self.window_size, n)
            
            window_metrics = self._compute_window_metrics(
                regime_labels, start, end
            )
            windows.append(window_metrics)
        
        return windows
    
    def _allocate_windows(
        self,
        windows: List[Dict]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Allocate windows to train, val, test sets.
        
        If preserve_diversity is True, high-entropy windows are prioritized
        for the training set to ensure diverse regime representation.
        
        Parameters
        ----------
        windows : list of dict
            Window metrics.
        
        Returns
        -------
        tuple of (list, list, list)
            Indices of windows for train, val, test sets.
        """
        n_windows = len(windows)
        
        # Sort by entropy (descending) if preserving diversity
        if self.preserve_diversity:
            window_indices = np.argsort(
                [-w['entropy'] for w in windows]
            )
        else:
            # Temporal order
            window_indices = np.arange(n_windows)
        
        # Allocate windows
        n_train = int(n_windows * self.train_ratio)
        n_val = int(n_windows * self.val_ratio)
        
        train_windows = window_indices[:n_train].tolist()
        val_windows = window_indices[n_train:n_train + n_val].tolist()
        test_windows = window_indices[n_train + n_val:].tolist()
        
        # Sort to maintain temporal order within each set
        train_windows.sort()
        val_windows.sort()
        test_windows.sort()
        
        return train_windows, val_windows, test_windows
    
    def split(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to split.
        regime_labels : np.ndarray
            Regime labels for each time point.
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            Train, validation, and test DataFrames.
        
        Raises
        ------
        ValueError
            If data and regime_labels have different lengths.
        
        Examples
        --------
        >>> data = pd.DataFrame({'y': np.random.randn(1000)})
        >>> labels = np.random.randint(0, 3, 1000)
        >>> splitter = WindowStratifiedSplit(window_size=50)
        >>> train, val, test = splitter.split(data, labels)
        """
        if len(data) != len(regime_labels):
            raise ValueError(
                f"data and regime_labels must have same length: "
                f"{len(data)} != {len(regime_labels)}"
            )
        
        # Create windows
        self.windows_ = self._create_windows(data, regime_labels)
        
        # Allocate windows to splits
        train_wins, val_wins, test_wins = self._allocate_windows(self.windows_)
        
        self.train_windows_ = train_wins
        self.val_windows_ = val_wins
        self.test_windows_ = test_wins
        
        # Extract indices
        train_indices = []
        for w_idx in train_wins:
            window = self.windows_[w_idx]
            train_indices.extend(range(window['start'], window['end']))
        
        val_indices = []
        for w_idx in val_wins:
            window = self.windows_[w_idx]
            val_indices.extend(range(window['start'], window['end']))
        
        test_indices = []
        for w_idx in test_wins:
            window = self.windows_[w_idx]
            test_indices.extend(range(window['start'], window['end']))
        
        # Create DataFrames
        train_df = data.iloc[train_indices].copy()
        val_df = data.iloc[val_indices].copy()
        test_df = data.iloc[test_indices].copy()
        
        return train_df, val_df, test_df
    
    def get_regime_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get regime distribution across train, val, test sets.
        
        Returns
        -------
        dict
            Regime counts for each split.
        
        Raises
        ------
        RuntimeError
            If split() hasn't been called yet.
        
        Examples
        --------
        >>> splitter.split(data, labels)
        >>> dist = splitter.get_regime_distribution()
        >>> print(dist['train'])
        {0: 500, 1: 30, 2: 20}
        """
        if not hasattr(self, 'windows_'):
            raise RuntimeError("Must call split() before get_regime_distribution()")
        
        def aggregate_regimes(window_indices: List[int]) -> Dict[int, int]:
            """Aggregate regime counts across windows."""
            total_counts = {}
            for w_idx in window_indices:
                window = self.windows_[w_idx]
                for regime, count in window['regime_counts'].items():
                    total_counts[regime] = total_counts.get(regime, 0) + count
            return total_counts
        
        return {
            'train': aggregate_regimes(self.train_windows_),
            'val': aggregate_regimes(self.val_windows_),
            'test': aggregate_regimes(self.test_windows_)
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate detailed split report.
        
        Parameters
        ----------
        output_file : str, optional
            If provided, save report to this file.
        
        Returns
        -------
        str
            Formatted report string.
        
        Raises
        ------
        RuntimeError
            If split() hasn't been called yet.
        """
        if not hasattr(self, 'windows_'):
            raise RuntimeError("Must call split() before generate_report()")
        
        # Get distributions
        dist = self.get_regime_distribution()
        
        # Compute total samples
        train_total = sum(dist['train'].values())
        val_total = sum(dist['val'].values())
        test_total = sum(dist['test'].values())
        total = train_total + val_total + test_total
        
        # Build report
        lines = [
            "=" * 80,
            "WINDOW-BASED STRATIFIED SPLIT REPORT",
            "=" * 80,
            "",
            "CONFIGURATION",
            "-" * 80,
            f"  Window size: {self.window_size}",
            f"  Preserve diversity: {self.preserve_diversity}",
            f"  Train ratio: {self.train_ratio:.2f}",
            f"  Val ratio: {self.val_ratio:.2f}",
            f"  Test ratio: {self.test_ratio:.2f}",
            "",
            "SPLIT SIZES",
            "-" * 80,
            f"  Train: {train_total} samples ({train_total/total*100:.1f}%)",
            f"  Val:   {val_total} samples ({val_total/total*100:.1f}%)",
            f"  Test:  {test_total} samples ({test_total/total*100:.1f}%)",
            "",
            "REGIME DISTRIBUTION",
            "-" * 80,
            f"{'Regime':<10} {'Train':<15} {'Val':<15} {'Test':<15}",
            "-" * 55
        ]
        
        # Get all unique regimes
        all_regimes = set()
        for split_dist in dist.values():
            all_regimes.update(split_dist.keys())
        
        for regime in sorted(all_regimes):
            train_count = dist['train'].get(regime, 0)
            val_count = dist['val'].get(regime, 0)
            test_count = dist['test'].get(regime, 0)
            
            lines.append(
                f"{regime:<10} {train_count:<15} {val_count:<15} {test_count:<15}"
            )
        
        lines.extend([
            "",
            "WINDOW STATISTICS",
            "-" * 80,
            f"  Total windows: {len(self.windows_)}",
            f"  Train windows: {len(self.train_windows_)}",
            f"  Val windows: {len(self.val_windows_)}",
            f"  Test windows: {len(self.test_windows_)}",
            "",
            f"  Avg entropy: {np.mean([w['entropy'] for w in self.windows_]):.3f}",
            f"  Max entropy: {np.max([w['entropy'] for w in self.windows_]):.3f}",
            "",
            "=" * 80
        ])
        
        report = "\n".join(lines)
        
        # Save if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


def quick_window_split(
    data: pd.DataFrame,
    regime_labels: np.ndarray,
    window_size: int = 30,
    train_ratio: float = 0.60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quick window-based stratified split with default parameters.
    
    Convenience function for common use case.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data.
    regime_labels : np.ndarray
        Regime labels.
    window_size : int, optional
        Window size in days, by default 30.
    train_ratio : float, optional
        Training set proportion, by default 0.60.
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Train, validation, test sets.
    
    Examples
    --------
    >>> train, val, test = quick_window_split(data, labels, window_size=50)
    """
    splitter = WindowStratifiedSplit(
        window_size=window_size,
        train_ratio=train_ratio
    )
    return splitter.split(data, regime_labels)

