"""
Görselleştirme modülü.

Bu modül, simülasyon sonuçlarını ve model performanslarını
görselleştirmek için fonksiyonlar içerir.
"""

import numpy as np
import pandas as pd
import warnings
import os
from typing import Optional, List, Dict

# Matplotlib backend'ini Agg'ye ayarla (GUI gerektirmez, sadece dosyaya kaydet)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Deprecation uyarılarını filtrele
warnings.filterwarnings('ignore', category=DeprecationWarning)


class ResultVisualizer:
    """
    Sonuç görselleştirme sınıfı.
    
    Bu sınıf, zaman serisi tahminleri, artıklar, kütle evrimi ve
    performans karşılaştırmaları için çeşitli grafikler oluşturur.
    """
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-darkgrid',
        figsize: tuple = (15, 10),
        dpi: int = 100
    ):
        """
        ResultVisualizer sınıfını başlatır.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib stili (varsayılan: 'seaborn-v0_8-darkgrid')
        figsize : tuple, optional
            Figür boyutu (varsayılan: (15, 10))
        dpi : int, optional
            Çözünürlük (varsayılan: 100)
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'actual': '#2E86AB',
            'baseline': '#A23B72',
            'grm': '#F18F01',
            'schwarzschild': '#C06C84',
            'kerr': '#F18F01',
            'shock': '#C73E1D'
        }
    
    def plot_time_series_comparison(
        self,
        time: np.ndarray,
        y_actual: np.ndarray,
        y_baseline: np.ndarray,
        y_grm: np.ndarray,
        shock_positions: Optional[List[int]] = None,
        train_end: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Zaman serisi ve tahminleri karşılaştırmalı olarak çizer.
        
        Parameters
        ----------
        time : np.ndarray
            Zaman indeksi
        y_actual : np.ndarray
            Gerçek değerler
        y_baseline : np.ndarray
            Baseline tahminler
        y_grm : np.ndarray
            GRM tahminler
        shock_positions : List[int], optional
            Şok pozisyonları (varsayılan: None)
        train_end : int, optional
            Eğitim-test sınırı (varsayılan: None)
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Zaman serilerini çiz
        ax.plot(time, y_actual, label='Gerçek Veri',
                color=self.colors['actual'], linewidth=2, alpha=0.8)
        ax.plot(time, y_baseline, label='Baseline (ARIMA)',
                color=self.colors['baseline'], linewidth=1.5,
                linestyle='--', alpha=0.7)
        ax.plot(time, y_grm, label='GRM (Schwarzschild)',
                color=self.colors['grm'], linewidth=1.5,
                linestyle='-.', alpha=0.7)
        
        # Şok pozisyonlarını işaretle
        if shock_positions is not None:
            for shock_pos in shock_positions:
                ax.axvline(x=shock_pos, color=self.colors['shock'],
                          linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(x=shock_positions[0], color=self.colors['shock'],
                      linestyle=':', alpha=0.5, linewidth=1,
                      label='Şok Noktaları')
        
        # Train-test sınırı
        if train_end is not None:
            ax.axvline(x=train_end, color='black', linestyle='-',
                      alpha=0.3, linewidth=2, label='Train/Test Sınırı')
        
        ax.set_xlabel('Zaman', fontsize=12)
        ax.set_ylabel('Değer', fontsize=12)
        ax.set_title('Zaman Serisi Tahmin Karşılaştırması', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')  # Bellek temizliği için
    
    def plot_residuals_comparison(
        self,
        time: np.ndarray,
        residuals_baseline: np.ndarray,
        residuals_grm: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Baseline ve GRM artıklarını karşılaştırır.
        
        Parameters
        ----------
        time : np.ndarray
            Zaman indeksi
        residuals_baseline : np.ndarray
            Baseline artıkları
        residuals_grm : np.ndarray
            GRM artıkları
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Baseline artıkları
        axes[0].plot(time, residuals_baseline, color=self.colors['baseline'],
                    linewidth=1, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_title('Baseline Model Artıkları', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Artık Değeri', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # GRM artıkları
        axes[1].plot(time, residuals_grm, color=self.colors['grm'],
                    linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_title('GRM Model Artıkları', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Zaman', fontsize=10)
        axes[1].set_ylabel('Artık Değeri', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')  # Bellek temizliği için
    
    def plot_mass_evolution(
        self,
        time: np.ndarray,
        mass: np.ndarray,
        shock_threshold: float,
        shock_positions: Optional[List[int]] = None,
        detected_shocks: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Kütle (volatilite) evrimini ve olay ufkunu çizer.
        
        Parameters
        ----------
        time : np.ndarray
            Zaman indeksi
        mass : np.ndarray
            Kütle serisi M(t)
        shock_threshold : float
            Olay ufku eşiği
        shock_positions : List[int], optional
            Gerçek şok pozisyonları (varsayılan: None)
        detected_shocks : List[int], optional
            Algılanan şok pozisyonları (varsayılan: None)
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Kütle evrimi
        ax.plot(time, mass, color=self.colors['grm'], linewidth=2,
               label='Kütle M(t) [Volatilite]')
        
        # Olay ufku eşiği
        ax.axhline(y=shock_threshold, color='red', linestyle='--',
                  linewidth=2, alpha=0.7, label='Olay Ufku (σ²_critical)')
        
        # Gerçek şoklar
        if shock_positions is not None:
            for shock_pos in shock_positions:
                ax.axvline(x=shock_pos, color=self.colors['shock'],
                          linestyle=':', alpha=0.4, linewidth=1.5)
            ax.axvline(x=shock_positions[0], color=self.colors['shock'],
                      linestyle=':', alpha=0.4, linewidth=1.5,
                      label='Gerçek Şoklar')
        
        # Algılanan şoklar
        if detected_shocks is not None and len(detected_shocks) > 0:
            shock_masses = [mass[int(s)] for s in detected_shocks if s < len(mass)]
            ax.scatter(detected_shocks[:len(shock_masses)], shock_masses,
                      color='red', s=100, marker='X', zorder=5,
                      label='Algılanan Şoklar', edgecolors='black')
        
        ax.set_xlabel('Zaman', fontsize=12)
        ax.set_ylabel('Kütle (Varyans)', fontsize=12)
        ax.set_title('Kütle Evrimi ve Olay Ufku', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')  # Bellek temizliği için
    
    def plot_performance_comparison(
        self,
        comparison_dict: Dict[str, any],
        save_path: Optional[str] = None
    ):
        """
        Model performans karşılaştırmasını görselleştirir.
        
        Parameters
        ----------
        comparison_dict : Dict[str, any]
            ModelEvaluator.compare_models() çıktısı
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        baseline = comparison_dict['baseline_metrics']
        grm = comparison_dict['grm_metrics']
        
        metrics = ['rmse', 'mae', 'mape', 'r2']
        metric_labels = ['RMSE', 'MAE', 'MAPE (%)', 'R²']
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            baseline_val = baseline[metric]
            grm_val = grm[metric]
            
            x = ['Baseline', 'GRM']
            y = [baseline_val, grm_val]
            colors_list = [self.colors['baseline'], self.colors['grm']]
            
            bars = axes[idx].bar(x, y, color=colors_list, alpha=0.7,
                                edgecolor='black', linewidth=1.5)
            
            # Değerleri bar üzerine yaz
            for bar, val in zip(bars, y):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.3f}',
                             ha='center', va='bottom', fontsize=10,
                             fontweight='bold')
            
            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Değer', fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Performans Karşılaştırması', fontsize=14,
                    fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')  # Bellek temizliği için
    
    def plot_spin_evolution(
        self,
        time: np.ndarray,
        spin: np.ndarray,
        mass: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Dönme (otokorelasyon) evrimini çizer (Kerr için).
        
        Parameters
        ----------
        time : np.ndarray
            Zaman indeksi
        spin : np.ndarray
            Dönme serisi a(t)
        mass : np.ndarray
            Kütle serisi M(t) (karşılaştırma için)
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Dönme a(t)
        axes[0].plot(time, spin, color=self.colors['kerr'], linewidth=2,
                    label='Dönme a(t) [Otokorelasyon]')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].fill_between(time, 0, spin, where=(spin >= 0),
                            color=self.colors['kerr'], alpha=0.3,
                            label='Pozitif Momentum')
        axes[0].fill_between(time, 0, spin, where=(spin < 0),
                            color='red', alpha=0.3,
                            label='Negatif Momentum')
        axes[0].set_ylabel('Dönme a(t)', fontsize=12)
        axes[0].set_title('Dönme Parametresi Evrimi (Kerr)', fontsize=14,
                         fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1.1, 1.1])
        
        # Kütle M(t) (karşılaştırma için)
        axes[1].plot(time, mass, color=self.colors['baseline'], linewidth=2,
                    label='Kütle M(t) [Volatilite]')
        axes[1].set_xlabel('Zaman', fontsize=12)
        axes[1].set_ylabel('Kütle M(t)', fontsize=12)
        axes[1].set_title('Kütle Parametresi (Referans)', fontsize=12,
                         fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')
    
    def plot_three_model_comparison(
        self,
        time: np.ndarray,
        y_actual: np.ndarray,
        y_baseline: np.ndarray,
        y_schwarzschild: np.ndarray,
        y_kerr: np.ndarray,
        shock_positions: Optional[List[int]] = None,
        train_end: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Üç model karşılaştırması: Baseline, Schwarzschild, Kerr.
        
        Parameters
        ----------
        time : np.ndarray
            Zaman indeksi
        y_actual : np.ndarray
            Gerçek değerler
        y_baseline : np.ndarray
            Baseline tahminler
        y_schwarzschild : np.ndarray
            Schwarzschild GRM tahminler
        y_kerr : np.ndarray
            Kerr GRM tahminler
        shock_positions : List[int], optional
            Şok pozisyonları (varsayılan: None)
        train_end : int, optional
            Eğitim-test sınırı (varsayılan: None)
        save_path : str, optional
            Kaydetme yolu (varsayılan: None)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Zaman serilerini çiz
        ax.plot(time, y_actual, label='Gerçek Veri',
                color=self.colors['actual'], linewidth=2.5, alpha=0.8)
        ax.plot(time, y_baseline, label='Baseline (ARIMA)',
                color=self.colors['baseline'], linewidth=1.5,
                linestyle='--', alpha=0.7)
        ax.plot(time, y_schwarzschild, label='GRM Schwarzschild (Kütle)',
                color=self.colors.get('schwarzschild', '#C06C84'),
                linewidth=1.5, linestyle='-.', alpha=0.7)
        ax.plot(time, y_kerr, label='GRM Kerr (Kütle+Dönme)',
                color=self.colors['kerr'], linewidth=1.5,
                linestyle=':', alpha=0.8)
        
        # Şok pozisyonları
        if shock_positions is not None:
            for shock_pos in shock_positions:
                ax.axvline(x=shock_pos, color=self.colors['shock'],
                          linestyle=':', alpha=0.4, linewidth=1.5)
            ax.axvline(x=shock_positions[0], color=self.colors['shock'],
                      linestyle=':', alpha=0.4, linewidth=1.5,
                      label='Şok Noktaları')
        
        # Train-test sınırı
        if train_end is not None:
            ax.axvline(x=train_end, color='black', linestyle='-',
                      alpha=0.3, linewidth=2, label='Train/Test Sınırı')
        
        ax.set_xlabel('Zaman', fontsize=12)
        ax.set_ylabel('Değer', fontsize=12)
        ax.set_title('Üç Model Karşılaştırması (FAZE 2)', fontsize=14,
                    fontweight='bold')
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Grafik kaydedildi: {save_path}")
        
        plt.close('all')

