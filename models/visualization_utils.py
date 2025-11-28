"""Visualization Utilities for GRM Analysis.

This module provides comprehensive visualization functions for GRM models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GRMVisualizer:
    """Comprehensive visualization suite for GRM models."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save visualizations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")
    
    def plot_time_series_comparison(
        self,
        test_df: pd.DataFrame,
        baseline_pred: np.ndarray,
        ensemble_pred: np.ndarray,
        adaptive_pred: np.ndarray,
        ticker: str = 'Asset'
    ):
        """Plot time series comparison: Actual vs Models.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data with actual returns.
        baseline_pred : np.ndarray
            Baseline model predictions.
        ensemble_pred : np.ndarray
            Ensemble GRM predictions.
        adaptive_pred : np.ndarray
            Adaptive GRM predictions.
        ticker : str
            Asset ticker for title.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        actual = test_df['returns'].values
        time_index = range(len(actual))
        
        # Plot 1: Full comparison
        axes[0].plot(time_index, actual, label='Actual', alpha=0.7, linewidth=1.5, color='black')
        axes[0].plot(time_index, baseline_pred, label='Baseline', alpha=0.6, linewidth=1, linestyle='--')
        axes[0].plot(time_index, ensemble_pred, label='Ensemble GRM', alpha=0.7, linewidth=1.2)
        axes[0].plot(time_index, adaptive_pred, label='Adaptive GRM', alpha=0.7, linewidth=1.2)
        axes[0].set_title(f'{ticker} - Time Series Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Returns', fontsize=11)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction errors
        baseline_error = actual - baseline_pred
        ensemble_error = actual - ensemble_pred
        adaptive_error = actual - adaptive_pred
        
        axes[1].plot(time_index, baseline_error, label='Baseline Error', alpha=0.6, linewidth=1)
        axes[1].plot(time_index, ensemble_error, label='Ensemble Error', alpha=0.7, linewidth=1.2)
        axes[1].plot(time_index, adaptive_error, label='Adaptive Error', alpha=0.7, linewidth=1.2)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Error', fontsize=11)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative errors (squared)
        axes[2].plot(time_index, np.cumsum(baseline_error**2), label='Baseline Cum. Sq. Error', linewidth=1.5)
        axes[2].plot(time_index, np.cumsum(ensemble_error**2), label='Ensemble Cum. Sq. Error', linewidth=1.5)
        axes[2].plot(time_index, np.cumsum(adaptive_error**2), label='Adaptive Cum. Sq. Error', linewidth=1.5)
        axes[2].set_title('Cumulative Squared Errors', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time Step', fontsize=11)
        axes[2].set_ylabel('Cumulative Squared Error', fontsize=11)
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_time_series_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_regime_distribution(
        self,
        regime_labels: np.ndarray,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ticker: str = 'Asset'
    ):
        """Plot regime distribution across splits.
        
        Parameters
        ----------
        regime_labels : np.ndarray
            All regime labels.
        train_df, val_df, test_df : pd.DataFrame
            Data splits.
        ticker : str
            Asset ticker.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get split sizes
        train_size = len(train_df)
        val_size = len(val_df)
        test_size = len(test_df)
        
        # Map regime labels (assuming continuous indexing)
        train_regimes = regime_labels[:train_size]
        val_regimes = regime_labels[train_size:train_size+val_size]
        test_regimes = regime_labels[train_size+val_size:]
        
        # Plot 1: Overall regime distribution
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        axes[0, 0].bar(unique_regimes, counts, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Overall Regime Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Regime ID', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Train/Val/Test split comparison
        train_counts = np.bincount(train_regimes[train_regimes >= 0], minlength=len(unique_regimes))
        val_counts = np.bincount(val_regimes[val_regimes >= 0], minlength=len(unique_regimes))
        test_counts = np.bincount(test_regimes[test_regimes >= 0], minlength=len(unique_regimes))
        
        x = np.arange(len(unique_regimes))
        width = 0.25
        
        axes[0, 1].bar(x - width, train_counts, width, label='Train', alpha=0.7)
        axes[0, 1].bar(x, val_counts, width, label='Val', alpha=0.7)
        axes[0, 1].bar(x + width, test_counts, width, label='Test', alpha=0.7)
        axes[0, 1].set_title('Regime Distribution by Split', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Regime ID', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Regime timeline
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_regimes)))
        for i, regime in enumerate(unique_regimes):
            regime_mask = regime_labels == regime
            axes[1, 0].scatter(
                np.where(regime_mask)[0],
                [regime] * np.sum(regime_mask),
                c=[colors[i]],
                s=10,
                alpha=0.6,
                label=f'Regime {regime}'
            )
        
        # Mark split boundaries
        axes[1, 0].axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='Train|Val')
        axes[1, 0].axvline(x=train_size+val_size, color='blue', linestyle='--', linewidth=2, label='Val|Test')
        
        axes[1, 0].set_title('Regime Timeline', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Step', fontsize=11)
        axes[1, 0].set_ylabel('Regime ID', fontsize=11)
        axes[1, 0].legend(loc='upper right', fontsize=8, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Regime transition matrix (heatmap)
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            if regime_labels[i] >= 0 and regime_labels[i+1] >= 0:
                transition_matrix[regime_labels[i], regime_labels[i+1]] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        im = axes[1, 1].imshow(transition_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        axes[1, 1].set_title('Regime Transition Probabilities', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('To Regime', fontsize=11)
        axes[1, 1].set_ylabel('From Regime', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Probability', fontsize=10)
        
        # Add text annotations for significant transitions
        for i in range(n_regimes):
            for j in range(n_regimes):
                if transition_matrix[i, j] > 0.1:
                    axes[1, 1].text(j, i, f'{transition_matrix[i, j]:.2f}',
                                   ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_regime_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_adaptive_alpha_evolution(
        self,
        alpha_history: np.ndarray,
        volatility_history: np.ndarray,
        test_df: pd.DataFrame,
        ticker: str = 'Asset'
    ):
        """Plot adaptive alpha evolution and correlation with volatility.
        
        Parameters
        ----------
        alpha_history : np.ndarray
            History of alpha values.
        volatility_history : np.ndarray
            History of volatility values.
        test_df : pd.DataFrame
            Test data.
        ticker : str
            Asset ticker.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        time_index = range(len(alpha_history))
        
        # Plot 1: Alpha evolution
        axes[0].plot(time_index, alpha_history, label='Alpha', color='purple', linewidth=1.5)
        axes[0].axhline(y=np.mean(alpha_history), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(alpha_history):.3f}', linewidth=1.5)
        axes[0].fill_between(time_index, 
                            np.min(alpha_history), 
                            alpha_history, 
                            alpha=0.3, 
                            color='purple')
        axes[0].set_title(f'{ticker} - Adaptive Alpha Evolution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Alpha Value', fontsize=11)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Volatility evolution
        axes[1].plot(time_index, volatility_history, label='Volatility', color='orange', linewidth=1.5)
        axes[1].axhline(y=np.mean(volatility_history), color='red', linestyle='--',
                       label=f'Mean: {np.mean(volatility_history):.6f}', linewidth=1.5)
        axes[1].fill_between(time_index, 
                            np.min(volatility_history), 
                            volatility_history, 
                            alpha=0.3, 
                            color='orange')
        axes[1].set_title('Volatility (Mass) Evolution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volatility', fontsize=11)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation scatter
        correlation = np.corrcoef(alpha_history, volatility_history)[0, 1]
        
        axes[2].scatter(volatility_history, alpha_history, alpha=0.5, s=30, c=time_index, cmap='viridis')
        
        # Add regression line
        z = np.polyfit(volatility_history, alpha_history, 1)
        p = np.poly1d(z)
        vol_sorted = np.sort(volatility_history)
        axes[2].plot(vol_sorted, p(vol_sorted), "r--", linewidth=2, 
                    label=f'Linear fit (r={correlation:.3f})')
        
        axes[2].set_title('Alpha-Volatility Correlation', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Volatility', fontsize=11)
        axes[2].set_ylabel('Alpha', fontsize=11)
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
        cbar.set_label('Time Step', fontsize=10)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_adaptive_alpha_evolution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_correction_analysis(
        self,
        ensemble_corrections: np.ndarray,
        adaptive_corrections: np.ndarray,
        test_df: pd.DataFrame,
        ticker: str = 'Asset'
    ):
        """Plot correction magnitude and distribution analysis.
        
        Parameters
        ----------
        ensemble_corrections : np.ndarray
            Ensemble GRM corrections.
        adaptive_corrections : np.ndarray
            Adaptive GRM corrections.
        test_df : pd.DataFrame
            Test data.
        ticker : str
            Asset ticker.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        time_index = range(len(ensemble_corrections))
        
        # Plot 1: Correction magnitude over time
        axes[0, 0].plot(time_index, ensemble_corrections, label='Ensemble', alpha=0.7, linewidth=1)
        axes[0, 0].plot(time_index, adaptive_corrections, label='Adaptive', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[0, 0].set_title('GRM Corrections Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Step', fontsize=11)
        axes[0, 0].set_ylabel('Correction', fontsize=11)
        axes[0, 0].legend(loc='best', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Correction distribution
        axes[0, 1].hist(ensemble_corrections, bins=50, alpha=0.6, label='Ensemble', density=True)
        axes[0, 1].hist(adaptive_corrections, bins=50, alpha=0.6, label='Adaptive', density=True)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Correction Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Correction Magnitude', fontsize=11)
        axes[0, 1].set_ylabel('Density', fontsize=11)
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Absolute correction magnitude
        axes[1, 0].plot(time_index, np.abs(ensemble_corrections), 
                       label='|Ensemble|', alpha=0.7, linewidth=1.5)
        axes[1, 0].plot(time_index, np.abs(adaptive_corrections),
                       label='|Adaptive|', alpha=0.7, linewidth=1.5)
        axes[1, 0].set_title('Absolute Correction Magnitude', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Step', fontsize=11)
        axes[1, 0].set_ylabel('|Correction|', fontsize=11)
        axes[1, 0].legend(loc='best', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correction effectiveness (scatter)
        actual_error = test_df['returns'].values - (test_df['returns'].values - ensemble_corrections)
        
        axes[1, 1].scatter(ensemble_corrections, actual_error, alpha=0.5, s=20, label='Ensemble')
        axes[1, 1].scatter(adaptive_corrections, actual_error, alpha=0.5, s=20, label='Adaptive')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Correction vs Error', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Correction Applied', fontsize=11)
        axes[1, 1].set_ylabel('Actual Error', fontsize=11)
        axes[1, 1].legend(loc='best', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_correction_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_performance_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        ticker: str = 'Asset'
    ):
        """Plot comprehensive performance metrics comparison.
        
        Parameters
        ----------
        metrics : Dict
            Dictionary with model names as keys and metric dicts as values.
        ticker : str
            Asset ticker.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        models = list(metrics.keys())
        
        # Extract metrics
        rmse_values = [metrics[m]['rmse'] for m in models]
        mae_values = [metrics[m]['mae'] for m in models]
        improvement_values = [metrics[m].get('improvement', 0) for m in models]
        
        # Plot 1: RMSE comparison
        colors = ['gray', 'skyblue', 'orange']
        axes[0, 0].bar(models, rmse_values, color=colors, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('RMSE', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(rmse_values):
            axes[0, 0].text(i, v + 0.0001, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: MAE comparison
        axes[0, 1].bar(models, mae_values, color=colors, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('MAE', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(mae_values):
            axes[0, 1].text(i, v + 0.0001, f'{v:.6f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Improvement percentage
        axes[1, 0].bar(models[1:], improvement_values[1:], color=['skyblue', 'orange'], 
                      edgecolor='black', alpha=0.7)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[1, 0].set_title('Improvement over Baseline (%)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Improvement (%)', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(improvement_values[1:]):
            axes[1, 0].text(i, v + 0.2, f'+{v:.2f}%', ha='center', va='bottom', 
                          fontsize=10, fontweight='bold', color='green')
        
        # Plot 4: Metrics summary table
        axes[1, 1].axis('off')
        
        table_data = []
        for model in models:
            row = [
                model,
                f"{metrics[model]['rmse']:.6f}",
                f"{metrics[model]['mae']:.6f}",
                f"{metrics[model].get('improvement', 0):.2f}%"
            ]
            table_data.append(row)
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Model', 'RMSE', 'MAE', 'Improvement'],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.25, 0.25, 0.2]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        axes[1, 1].set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_performance_metrics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_3d_grm_surface(
        self,
        test_df: pd.DataFrame,
        ensemble_corrections: np.ndarray,
        volatility_history: np.ndarray,
        ticker: str = 'Asset'
    ):
        """Create 3D surface plot: Time x Volatility x Correction.
        
        This visualization shows how GRM corrections vary with time and volatility
        in a 3D space, providing an intuitive understanding of the model's behavior.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data.
        ensemble_corrections : np.ndarray
            Ensemble GRM corrections.
        volatility_history : np.ndarray
            Volatility (mass) history.
        ticker : str
            Asset ticker.
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        time = np.arange(len(ensemble_corrections))
        volatility = volatility_history
        corrections = ensemble_corrections
        
        # Create color map based on correction magnitude
        colors = cm.RdYlBu_r(plt.Normalize()(corrections))
        
        # Scatter plot
        scatter = ax.scatter(
            time, 
            volatility, 
            corrections,
            c=corrections,
            cmap='RdYlBu_r',
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Create interpolated surface (optional, for better visualization)
        # Create grid
        time_grid = np.linspace(time.min(), time.max(), 50)
        vol_grid = np.linspace(volatility.min(), volatility.max(), 50)
        TIME_GRID, VOL_GRID = np.meshgrid(time_grid, vol_grid)
        
        # Interpolate corrections
        CORR_GRID = griddata(
            (time, volatility),
            corrections,
            (TIME_GRID, VOL_GRID),
            method='cubic',
            fill_value=0
        )
        
        # Plot surface
        surf = ax.plot_surface(
            TIME_GRID,
            VOL_GRID,
            CORR_GRID,
            alpha=0.3,
            cmap='RdYlBu_r',
            linewidth=0,
            antialiased=True
        )
        
        # Zero plane for reference
        ax.plot_surface(
            TIME_GRID,
            VOL_GRID,
            np.zeros_like(CORR_GRID),
            alpha=0.1,
            color='gray'
        )
        
        # Labels and title
        ax.set_xlabel('Time Step', fontsize=12, labelpad=10)
        ax.set_ylabel('Volatility (Mass)', fontsize=12, labelpad=10)
        ax.set_zlabel('GRM Correction', fontsize=12, labelpad=10)
        ax.set_title(
            f'{ticker} - 3D Gravitational Residual Model Surface\n'
            f'Time × Volatility × Correction',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Correction Magnitude', fontsize=11)
        
        # Improve viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key statistics
        stats_text = (
            f'Mean Correction: {np.mean(corrections):.6f}\n'
            f'Std Correction: {np.std(corrections):.6f}\n'
            f'Max |Correction|: {np.max(np.abs(corrections)):.6f}\n'
            f'Corr(Vol, Correction): {np.corrcoef(volatility, np.abs(corrections))[0,1]:.3f}'
        )
        
        ax.text2D(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_3d_grm_surface.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_residual_diagnostics(
        self,
        baseline_residuals: np.ndarray,
        ensemble_residuals: np.ndarray,
        adaptive_residuals: np.ndarray,
        ticker: str = 'Asset'
    ):
        """Plot residual diagnostic plots.
        
        Parameters
        ----------
        baseline_residuals : np.ndarray
            Baseline model residuals.
        ensemble_residuals : np.ndarray
            Ensemble GRM residuals.
        adaptive_residuals : np.ndarray
            Adaptive GRM residuals.
        ticker : str
            Asset ticker.
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        models = ['Baseline', 'Ensemble', 'Adaptive']
        residuals_list = [baseline_residuals, ensemble_residuals, adaptive_residuals]
        
        for i, (model, residuals) in enumerate(zip(models, residuals_list)):
            # Histogram
            axes[i, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[i, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[i, 0].set_title(f'{model} - Residual Distribution', fontsize=12, fontweight='bold')
            axes[i, 0].set_xlabel('Residual', fontsize=10)
            axes[i, 0].set_ylabel('Density', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3, axis='y')
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{model} - Q-Q Plot', fontsize=12, fontweight='bold')
            axes[i, 1].grid(True, alpha=0.3)
            
            # ACF plot
            plot_acf(residuals, lags=40, ax=axes[i, 2], alpha=0.05)
            axes[i, 2].set_title(f'{model} - ACF', fontsize=12, fontweight='bold')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / f'{ticker}_residual_diagnostics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def create_comprehensive_report(
        self,
        test_df: pd.DataFrame,
        baseline_pred: np.ndarray,
        ensemble_pred: np.ndarray,
        ensemble_corrections: np.ndarray,
        adaptive_pred: np.ndarray,
        adaptive_corrections: np.ndarray,
        alpha_history: np.ndarray,
        volatility_history: np.ndarray,
        regime_labels: np.ndarray,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metrics: Dict[str, Dict[str, float]],
        ticker: str = 'Asset'
    ):
        """Create comprehensive visualization report.
        
        Parameters
        ----------
        All necessary data for visualization.
        ticker : str
            Asset ticker.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"  CREATING COMPREHENSIVE VISUALIZATION REPORT - {ticker}")
        logger.info(f"{'='*80}\n")
        
        # 1. Time series comparison
        logger.info("[1/7] Creating time series comparison...")
        self.plot_time_series_comparison(
            test_df, baseline_pred, ensemble_pred, adaptive_pred, ticker
        )
        
        # 2. Regime distribution
        logger.info("[2/7] Creating regime distribution plots...")
        self.plot_regime_distribution(
            regime_labels, train_df, val_df, test_df, ticker
        )
        
        # 3. Adaptive alpha evolution
        logger.info("[3/7] Creating adaptive alpha evolution plots...")
        self.plot_adaptive_alpha_evolution(
            alpha_history, volatility_history, test_df, ticker
        )
        
        # 4. Correction analysis
        logger.info("[4/7] Creating correction analysis...")
        self.plot_correction_analysis(
            ensemble_corrections, adaptive_corrections, test_df, ticker
        )
        
        # 5. Performance metrics
        logger.info("[5/7] Creating performance metrics comparison...")
        self.plot_performance_metrics(metrics, ticker)
        
        # 6. Residual diagnostics
        logger.info("[6/7] Creating residual diagnostics...")
        baseline_residuals = test_df['returns'].values - baseline_pred
        ensemble_residuals = test_df['returns'].values - ensemble_pred
        adaptive_residuals = test_df['returns'].values - adaptive_pred
        
        self.plot_residual_diagnostics(
            baseline_residuals, ensemble_residuals, adaptive_residuals, ticker
        )
        
        # 7. 3D GRM surface (FINALE!)
        logger.info("[7/7] Creating 3D GRM surface visualization...")
        self.plot_3d_grm_surface(
            test_df, ensemble_corrections, volatility_history, ticker
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"  VISUALIZATION REPORT COMPLETED!")
        logger.info(f"  All plots saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")

