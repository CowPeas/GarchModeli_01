"""
Ablasyon Çalışması Modülü - GRM Modeli için Bileşen Analizi.

Bu modül, GRM modelinin farklı bileşenlerinin (kütle, dönme, decay, vb.)
performansa katkısını sistematik olarak ölçer.

FAZE 4: ZENGİNLEŞTİRME
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from models.baseline_model import BaselineARIMA
from models.grm_model import SchwarzschildGRM
from models.kerr_grm_model import KerrGRM
from models.metrics import calculate_rmse, calculate_mae


class AblationStudy:
    """
    GRM modeli için kapsamlı ablasyon çalışması.
    
    Test edilen varyasyonlar:
    1. Sadece kütle (M) - dönme yok
    2. Sadece dönme (a) - kütle yok
    3. Decay yok (β=0)
    4. Non-linearity yok (tanh -> sign)
    5. Farklı pencere boyutları
    """
    
    def __init__(
        self,
        train_data: pd.Series,
        val_data: pd.Series,
        test_data: pd.Series
    ):
        """
        AblationStudy sınıfını başlatır.
        
        Parameters
        ----------
        train_data : pd.Series
            Eğitim verisi
        val_data : pd.Series
            Doğrulama verisi
        test_data : pd.Series
            Test verisi
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.results = {}
        self.baseline_rmse = None
        self.baseline_model = None
    
    def run_baseline(self) -> float:
        """
        Baseline model (karşılaştırma referansı).
        
        Returns
        -------
        float
            Baseline RMSE
        """
        baseline = BaselineARIMA()
        
        # Grid search ile optimal parametreleri bul (train ve val kullan)
        best_order = baseline.grid_search(
            self.train_data,
            self.val_data,
            p_range=[0, 1, 2],
            d_range=[0, 1],
            q_range=[0, 1, 2],
            verbose=False
        )
        
        # Sadece train ile fit
        baseline.fit(self.train_data, order=best_order)
        self.baseline_model = baseline
        
        # Test predictions (walk-forward)
        predictions = self.walk_forward_predict(baseline, self.test_data)
        
        rmse = calculate_rmse(self.test_data.values, predictions)
        mae = calculate_mae(self.test_data.values, predictions)
        
        self.baseline_rmse = rmse
        self.results['Baseline'] = {
            'rmse': rmse,
            'mae': mae,
            'components': []
        }
        
        print(f"\n[ABLASYON] Baseline RMSE: {rmse:.6f}")
        
        return rmse
    
    def walk_forward_predict(
        self,
        model: BaselineARIMA,
        test_data: pd.Series
    ) -> np.ndarray:
        """
        Walk-forward validation ile tahmin yap.
        
        Parameters
        ----------
        model : BaselineARIMA
            Eğitilmiş model
        test_data : pd.Series
            Test verisi
            
        Returns
        -------
        np.ndarray
            Tahmin dizisi
        """
        predictions = []
        
        for i in range(len(test_data)):
            pred = model.predict(steps=1)[0]
            predictions.append(pred)
            
            if i < len(test_data) - 1:
                try:
                    model.fitted_model = model.fitted_model.append(
                        [test_data.iloc[i]], refit=False
                    )
                except:
                    pass
        
        return np.array(predictions)
    
    def run_grm_variant(
        self,
        name: str,
        model_class,
        **kwargs
    ) -> float:
        """
        Bir GRM varyantını çalıştır.
        
        Parameters
        ----------
        name : str
            Varyant adı
        model_class : type
            GRM model sınıfı (SchwarzschildGRM veya KerrGRM)
        **kwargs
            Model parametreleri
            
        Returns
        -------
        float
            RMSE değeri
        """
        # Baseline fit (eğer yoksa)
        if self.baseline_model is None:
            baseline = BaselineARIMA()
            best_order = baseline.grid_search(
                self.train_data, self.val_data,
                p_range=[0, 1, 2], d_range=[0, 1], q_range=[0, 1, 2],
                verbose=False
            )
            baseline.fit(self.train_data, order=best_order)
            self.baseline_model = baseline
        
        baseline = self.baseline_model
        train_residuals = baseline.get_residuals()
        
        # GRM fit
        grm_model = model_class(**kwargs)
        grm_model.fit(train_residuals)
        
        # Test predictions (walk-forward GRM)
        predictions = self.walk_forward_predict_grm(
            baseline, grm_model, self.test_data
        )
        
        rmse = calculate_rmse(self.test_data.values, predictions)
        mae = calculate_mae(self.test_data.values, predictions)
        
        improvement = (self.baseline_rmse - rmse) / self.baseline_rmse * 100
        
        self.results[name] = {
            'rmse': rmse,
            'mae': mae,
            'components': list(kwargs.keys()),
            'improvement': improvement
        }
        
        print(f"[ABLASYON] {name}: RMSE={rmse:.6f}, İyileşme={improvement:+.2f}%")
        
        return rmse
    
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
    
    def test_mass_only(self) -> float:
        """
        Ablasyon 1: Sadece kütle (M), dönme yok, decay yok.
        
        Returns
        -------
        float
            RMSE
        """
        return self.run_grm_variant(
            name='Mass_Only',
            model_class=SchwarzschildGRM,
            window_size=20,
            use_decay=False
        )
    
    def test_mass_with_decay(self) -> float:
        """
        Ablasyon 2: Kütle + Decay.
        
        Returns
        -------
        float
            RMSE
        """
        return self.run_grm_variant(
            name='Mass_Decay',
            model_class=SchwarzschildGRM,
            window_size=20,
            use_decay=True,
            beta=0.05
        )
    
    def test_kerr_full(self) -> float:
        """
        Ablasyon 3: Kerr Full (M + a + decay + tanh).
        
        Returns
        -------
        float
            RMSE
        """
        return self.run_grm_variant(
            name='Kerr_Full',
            model_class=KerrGRM,
            window_size=20,
            use_decay=True,
            use_tanh=True,
            beta=0.05,
            gamma=0.5
        )
    
    def test_kerr_no_decay(self) -> float:
        """
        Ablasyon 4: Kerr No Decay (M + a + tanh, decay yok).
        
        Returns
        -------
        float
            RMSE
        """
        return self.run_grm_variant(
            name='Kerr_No_Decay',
            model_class=KerrGRM,
            window_size=20,
            use_decay=False,
            use_tanh=True,
            gamma=0.5
        )
    
    def test_kerr_linear(self) -> float:
        """
        Ablasyon 5: Kerr Linear (M + a + decay, tanh yok).
        
        Returns
        -------
        float
            RMSE
        """
        return self.run_grm_variant(
            name='Kerr_Linear',
            model_class=KerrGRM,
            window_size=20,
            use_decay=True,
            use_tanh=False,
            beta=0.05,
            gamma=0.5
        )
    
    def test_window_sizes(self, sizes: List[int] = [10, 20, 30, 50, 100]) -> Dict[str, float]:
        """
        Hassasiyet 1: Farklı pencere boyutları.
        
        Parameters
        ----------
        sizes : List[int]
            Test edilecek pencere boyutları
            
        Returns
        -------
        Dict[str, float]
            Pencere boyutu -> RMSE mapping
        """
        results = {}
        for w in sizes:
            rmse = self.run_grm_variant(
                name=f'Window_{w}',
                model_class=KerrGRM,
                window_size=w,
                use_decay=True,
                use_tanh=True
            )
            results[f'Window_{w}'] = rmse
        
        return results
    
    def generate_report(self) -> pd.DataFrame:
        """
        Ablasyon sonuçlarını raporla.
        
        Returns
        -------
        pd.DataFrame
            Sonuç tablosu
        """
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.sort_values('improvement', ascending=False)
        
        print("\n" + "=" * 80)
        print("ABLASYON ÇALIŞMASI SONUÇLARI")
        print("=" * 80)
        print(df.to_string())
        print("\n")
        
        # En iyi ve en kötü bileşenleri bul
        if len(df) > 0:
            best = df.iloc[0]
            worst = df.iloc[-1]
            
            print(f"EN İYİ VARİYASYON: {best.name}")
            print(f"  - RMSE: {best['rmse']:.6f}")
            print(f"  - İyileşme: {best['improvement']:.2f}%")
            print(f"  - Bileşenler: {best['components']}")
            print()
        
        return df
    
    def plot_results(self, save_path: str = 'results/ablation_study.png'):
        """
        Ablasyon sonuçlarını görselleştir.
        
        Parameters
        ----------
        save_path : str
            Kayıt yolu
        """
        if len(self.results) == 0:
            print("[UYARI] Görselleştirme için sonuç yok!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        names = list(self.results.keys())
        rmses = [self.results[n]['rmse'] for n in names]
        improvements = [self.results[n].get('improvement', 0) for n in names]
        
        # 1. RMSE karşılaştırması
        axes[0, 0].bar(range(len(names)), rmses, color='steelblue')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_title('RMSE Karşılaştırması')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. İyileşme yüzdeleri
        colors = ['green' if x > 0 else 'red' for x in improvements]
        axes[0, 1].barh(range(len(names)), improvements, color=colors)
        axes[0, 1].set_yticks(range(len(names)))
        axes[0, 1].set_yticklabels(names)
        axes[0, 1].set_title('Baseline\'a Göre İyileşme (%)')
        axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('İyileşme (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Pencere boyutu hassasiyeti (eğer varsa)
        window_results = {k: v for k, v in self.results.items() if k.startswith('Window_')}
        if len(window_results) > 0:
            window_sizes = [int(k.split('_')[1]) for k in window_results.keys()]
            window_rmses = [window_results[k]['rmse'] for k in window_results.keys()]
            sorted_pairs = sorted(zip(window_sizes, window_rmses))
            window_sizes, window_rmses = zip(*sorted_pairs)
            
            axes[1, 0].plot(window_sizes, window_rmses, marker='o', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel('Pencere Boyutu')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('Pencere Boyutu Hassasiyeti')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Pencere boyutu testi yapılmadı',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Pencere Boyutu Hassasiyeti')
        
        # 4. Bileşen katkıları (heatmap benzeri)
        component_contributions = {}
        for name, result in self.results.items():
            if name != 'Baseline':
                components = result.get('components', [])
                improvement = result.get('improvement', 0)
                for comp in components:
                    if comp not in component_contributions:
                        component_contributions[comp] = []
                    component_contributions[comp].append(improvement)
        
        if len(component_contributions) > 0:
            comp_names = list(component_contributions.keys())
            comp_avg_improvements = [np.mean(component_contributions[c]) for c in comp_names]
            
            axes[1, 1].barh(range(len(comp_names)), comp_avg_improvements, color='coral')
            axes[1, 1].set_yticks(range(len(comp_names)))
            axes[1, 1].set_yticklabels(comp_names)
            axes[1, 1].set_title('Bileşen Katkıları (Ortalama İyileşme)')
            axes[1, 1].set_xlabel('Ortalama İyileşme (%)')
            axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Bileşen analizi yapılamadı',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Bileşen Katkıları')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Görselleştirme kaydedildi: {save_path}")

