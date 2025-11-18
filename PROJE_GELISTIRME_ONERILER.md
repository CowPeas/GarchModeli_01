# 🎯 GRM PROJESİ: BÜTÜNCÜL GELİŞTİRME ÖNERİLERİ

**Tarih:** 2025-11-15  
**Analiz:** Mevcut Proje + Hipotez_03 + Hipotez_04 (PIML)  
**Hedef:** Daha Rasyonel ve İyi Çıktılar Üreten Sistem

---

## 📊 MEVCUT DURUMUN ANALİZİ

### ✅ Başarılan Adımlar

1. **3-Fazlı Yaklaşım İmplementasyonu**
   - Faze 1: Sentetik veri + Schwarzschild ✅
   - Faze 2: Kerr rejimi + Non-linear bükülme ✅
   - Faze 3: Gerçek veri + GARCH karşılaştırma ✅

2. **Metodolojik Sağlamlık**
   - Data leakage düzeltildi ✅
   - Walk-forward validation eklendi ✅
   - Proper time-series split ✅
   - İstatistiksel testler (Diebold-Mariano, ARCH-LM) ✅

3. **Modüler Mimari**
   - Temiz kod yapısı (PEP8/PEP257) ✅
   - Ayrı model sınıfları ✅
   - Konfigurasyon yönetimi ✅

### ⚠️ Mevcut Limitasyonlar

#### 1. **Ad-Hoc Bükülme Fonksiyonu**
```python
# Mevcut: Manuel tasarım
Γ(t) = α * M(t) * sign(ε(t))  # Schwarzschild
Γ(t) = α * M(t) * tanh(ε(t)) + γ * a(t) * ε(t)  # Kerr
```

**Sorun:**
- Fonksiyon formu keyfi olarak seçilmiş
- Veri kendi dinamiklerini öğrenemiyor
- Farklı varlıklar için genellenemeyebilir

**Etki:**
- Baseline RMSE: 0.101398
- GRM RMSE: 0.102091
- **GRM daha kötü performans gösteriyor**

#### 2. **İki Aşamalı Ayrık Model**
```python
# Mevcut: Pipeline approach
1. Baseline model eğit
2. Artıkları hesapla
3. GRM parametrelerini optimize et
4. Tahminleri birleştir
```

**Sorun:**
- Baseline, GRM'den habersiz
- Tek yönlü bilgi akışı
- Global optimum yerine lokal optimum

#### 3. **Tek Anomali Varsayımı**
- Tüm artıklar tek bir "kara delik" tarafından açıklanıyor
- Farklı şok kaynakları (pozitif/negatif, kısa/uzun dönem) modellenmemiş

#### 4. **Sınırlı Ablasyon Çalışması**
- Hangi bileşenin ne kadar katkı sağladığı net değil
- Parametre hassasiyeti sistematik olarak test edilmemiş

---

## 🎯 ÖNCELİKLİ GELİŞTİRME ÖNERİLERİ

### 🥇 **ÖNCELİK 1: Zenginleştirilmiş Bükülme Fonksiyonu (Hipotez_03)**

**Ne:** Decay factor (τ) ve gelişmiş metrik seçimi ekle

**Neden:** 
- Şokların etkisi zamanla azalmalı (fiziksel olarak tutarlı)
- Olay ufku istatistiksel olarak tanımlanmalı

**Nasıl:**

```python
# MEVCUT (models/grm_model.py):
def compute_curvature(self, residuals, mass):
    return self.alpha * mass * np.sign(residuals)

# ÖNERİLEN:
def compute_curvature_with_decay(self, residuals, mass, time_since_shock):
    """
    Decay factor eklenmiş bükülme fonksiyonu.
    
    Parameters
    ----------
    residuals : array-like
        Artık dizisi
    mass : array-like
        Kütle (volatilite) dizisi
    time_since_shock : array-like
        Her zaman noktası için son büyük şoktan geçen zaman
        
    Returns
    -------
    curvature : array-like
        Bükülme düzeltmeleri
    """
    # Decay factor: 1 / (1 + β * τ)
    decay = 1.0 / (1.0 + self.beta * time_since_shock)
    
    # Base curvature
    base_curvature = self.alpha * mass * np.tanh(residuals)
    
    # With decay
    curvature = base_curvature * decay
    
    return curvature

def detect_shocks(self, residuals, threshold_quantile=0.95):
    """
    Büyük şokları tespit et (olay ufku analojisi).
    
    Parameters
    ----------
    residuals : array-like
        Artık dizisi
    threshold_quantile : float
        Şok eşiği (örn: %95 quantile)
        
    Returns
    -------
    shock_times : list
        Şok zamanlarının indeksleri
    """
    abs_residuals = np.abs(residuals)
    threshold = np.quantile(abs_residuals, threshold_quantile)
    shock_times = np.where(abs_residuals > threshold)[0]
    return shock_times

def compute_time_since_shock(self, current_time, shock_times):
    """
    Her zaman noktası için son şoktan geçen zamanı hesapla.
    
    Parameters
    ----------
    current_time : int
        Güncel zaman indeksi
    shock_times : list
        Şok zamanlarının indeksleri
        
    Returns
    -------
    tau : float
        Son şoktan geçen zaman (adım sayısı)
    """
    if len(shock_times) == 0 or current_time < shock_times[0]:
        return float('inf')  # Hiç şok olmadı
    
    past_shocks = shock_times[shock_times < current_time]
    if len(past_shocks) == 0:
        return float('inf')
    
    last_shock = past_shocks[-1]
    tau = current_time - last_shock
    return tau
```

**Beklenen İyileşme:**
- RMSE: %2-5 iyileşme
- Fiziksel tutarlılık: ⭐⭐⭐⭐⭐
- Implementasyon süresi: 1-2 gün

**Action Plan:**
1. `models/grm_model.py` ve `models/kerr_grm_model.py` güncelle
2. `config_phase3.py` içine `decay_beta_range` ekle
3. `main_phase3.py` içinde decay parametresi optimize et
4. Sonuçları karşılaştır (eski vs yeni)

---

### 🥈 **ÖNCELİK 2: Kapsamlı Ablasyon ve Hassasiyet Çalışması (Hipotez_03)**

**Ne:** Her bileşenin katkısını sistematik olarak ölç

**Neden:**
- Hangi parametrenin kritik olduğunu anla
- Gereksiz karmaşıklıktan kaçın
- Model yorumlanabilirliğini artır

**Nasıl:**

```python
# Yeni dosya: models/ablation_study.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from models import BaselineARIMA, SchwarzschildGRM, KerrGRM
from models.metrics import calculate_rmse

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
    
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.results = {}
    
    def run_baseline(self):
        """Baseline model (karşılaştırma referansı)."""
        baseline = BaselineARIMA()
        baseline.fit(self.train_data)
        predictions = baseline.predict(len(self.test_data))
        rmse = calculate_rmse(self.test_data, predictions)
        self.results['Baseline'] = {'rmse': rmse, 'components': []}
        return rmse
    
    def run_variant(self, name: str, model_class, **kwargs):
        """Bir GRM varyantını çalıştır."""
        # Baseline fit
        baseline = BaselineARIMA()
        baseline.fit(self.train_data)
        train_residuals = baseline.get_residuals()
        
        # GRM fit
        grm_model = model_class(**kwargs)
        grm_model.fit(train_residuals)
        
        # Test predictions (walk-forward)
        # ... (implement walk-forward logic)
        
        rmse = calculate_rmse(self.test_data, predictions)
        self.results[name] = {
            'rmse': rmse,
            'components': list(kwargs.keys()),
            'improvement': (self.results['Baseline']['rmse'] - rmse) / self.results['Baseline']['rmse'] * 100
        }
        return rmse
    
    def test_mass_only(self):
        """Ablasyon 1: Sadece kütle (M), dönme yok."""
        return self.run_variant(
            name='Mass_Only',
            model_class=SchwarzschildGRM,
            window_size=20,
            use_decay=True
        )
    
    def test_spin_only(self):
        """Ablasyon 2: Sadece dönme (a), kütle sabit."""
        # Custom variant needed
        pass
    
    def test_no_decay(self):
        """Ablasyon 3: Decay yok (β=0)."""
        return self.run_variant(
            name='No_Decay',
            model_class=KerrGRM,
            window_size=20,
            use_decay=False,
            use_tanh=True
        )
    
    def test_linear_only(self):
        """Ablasyon 4: Non-linearity yok (sign yerine tanh)."""
        return self.run_variant(
            name='Linear_Only',
            model_class=KerrGRM,
            window_size=20,
            use_decay=True,
            use_tanh=False
        )
    
    def test_window_sizes(self, sizes: List[int] = [10, 20, 30, 50, 100]):
        """Hassasiyet 1: Farklı pencere boyutları."""
        for w in sizes:
            self.run_variant(
                name=f'Window_{w}',
                model_class=KerrGRM,
                window_size=w,
                use_decay=True,
                use_tanh=True
            )
    
    def test_alpha_sensitivity(self, alphas: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]):
        """Hassasiyet 2: Alpha parametresi hassasiyeti."""
        for alpha in alphas:
            # Fix alpha, optimize others
            pass
    
    def generate_report(self) -> pd.DataFrame:
        """Ablasyon sonuçlarını raporla."""
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.sort_values('improvement', ascending=False)
        
        print("\n" + "="*80)
        print("ABLASYON ÇALIŞMASI SONUÇLARI")
        print("="*80)
        print(df.to_string())
        print("\n")
        
        # En iyi ve en kötü bileşenleri bul
        best = df.iloc[0]
        worst = df.iloc[-1]
        
        print(f"EN İYİ VARİYASYON: {best.name}")
        print(f"  - RMSE: {best['rmse']:.6f}")
        print(f"  - İyileşme: {best['improvement']:.2f}%")
        print(f"  - Bileşenler: {best['components']}")
        print()
        
        return df
    
    def plot_results(self):
        """Ablasyon sonuçlarını görselleştir."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. RMSE karşılaştırması
        names = list(self.results.keys())
        rmses = [self.results[n]['rmse'] for n in names]
        axes[0, 0].bar(names, rmses)
        axes[0, 0].set_title('RMSE Karşılaştırması')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. İyileşme yüzdeleri
        improvements = [self.results[n].get('improvement', 0) for n in names]
        axes[0, 1].barh(names, improvements)
        axes[0, 1].set_title('Baseline\'a Göre İyileşme (%)')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        
        # 3. Pencere boyutu hassasiyeti
        # ...
        
        # 4. Bileşen katkıları
        # ...
        
        plt.tight_layout()
        plt.savefig('results/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()
```

**Kullanım:**

```python
# main_ablation_study.py

from models.ablation_study import AblationStudy

# Veri hazırla
train_df, val_df, test_df = split_data(df)

# Ablasyon çalışması
study = AblationStudy(train_df['y'], val_df['y'], test_df['y'])

# Baseline
study.run_baseline()

# Ablasyonlar
study.test_mass_only()
study.test_spin_only()
study.test_no_decay()
study.test_linear_only()

# Hassasiyet analizleri
study.test_window_sizes()
study.test_alpha_sensitivity()

# Rapor
results_df = study.generate_report()
study.plot_results()
```

**Beklenen Çıktı:**
```
================================================================================
ABLASYON ÇALIŞMASI SONUÇLARI
================================================================================
                    rmse  components                      improvement
Kerr_Full      0.098234  [M, a, decay, tanh]            +3.12%
Schwarzschild  0.100123  [M, decay, tanh]               +1.26%
Mass_Only      0.101456  [M]                            +0.06%
No_Decay       0.102789  [M, a, tanh]                   -1.37%
Linear_Only    0.103234  [M, a, decay]                  -1.81%
Baseline       0.101398  []                             0.00%

EN İYİ VARİYASYON: Kerr_Full
  - RMSE: 0.098234
  - İyileşme: +3.12%
  - Bileşenler: ['M', 'a', 'decay', 'tanh']
```

**Action Plan:**
1. `models/ablation_study.py` oluştur
2. `main_ablation_study.py` oluştur
3. Tüm varyasyonları çalıştır (4-6 saat hesaplama)
4. Sonuçları analiz et ve rapor oluştur

---

### 🥉 **ÖNCELİK 3: Time-Series Cross-Validation (Hipotez_03)**

**Ne:** Tek test seti yerine rolling window validation

**Neden:**
- Model sağlamlığını farklı dönemlerde test et
- Aşırı uydurma (overfitting) tespiti
- Daha güvenilir performans tahmini

**Nasıl:**

```python
# Yeni dosya: models/cross_validation.py

import numpy as np
from typing import List, Tuple, Dict

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
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, data: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Veriyi k fold'a böl.
        
        Returns
        -------
        folds : List[Tuple]
            Her fold için (train_indices, val_indices, test_indices)
        """
        n = len(data)
        folds = []
        
        current_train_end = self.initial_train_size
        
        while current_train_end + self.val_size + self.test_size <= n:
            train_indices = np.arange(0, current_train_end)
            val_indices = np.arange(current_train_end, current_train_end + self.val_size)
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
        
        Returns
        -------
        results : Dict
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
            
            train_data = data[train_idx]
            val_data = data[val_idx]
            test_data = data[test_idx]
            
            # Model train
            model = model_class(**model_kwargs)
            model.fit(train_data, val_data)
            
            # Test predict
            predictions = model.predict(len(test_data))
            
            # Metrics
            rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
            mae = np.mean(np.abs(test_data - predictions))
            
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['fold'].append(i + 1)
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Tuple[type, dict]],
        data: np.ndarray
    ) -> pd.DataFrame:
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
        comparison_df : pd.DataFrame
            Karşılaştırma tablosu
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
```

**Kullanım:**

```python
# main_cross_validation.py

from models.cross_validation import TimeSeriesCrossValidator

# CV oluştur
cv = TimeSeriesCrossValidator(
    initial_train_size=300,
    val_size=50,
    test_size=50,
    step_size=50
)

# Modelleri tanımla
models = {
    'Baseline': (BaselineARIMA, {}),
    'Schwarzschild': (SchwarzschildGRM, {'window_size': 20}),
    'Kerr': (KerrGRM, {'window_size': 20, 'use_tanh': True}),
    'GARCH': (GARCHModel, {'p': 1, 'q': 1})
}

# Karşılaştır
comparison_df, detailed_results = cv.compare_models(models, df['y'].values)

print("\n" + "="*80)
print("TIME-SERIES CROSS-VALIDATION SONUÇLARI")
print("="*80)
print(comparison_df.to_string())
```

**Beklenen Çıktı:**
```
================================================================================
TIME-SERIES CROSS-VALIDATION SONUÇLARI
================================================================================
          Model  Mean_RMSE  Std_RMSE  Min_RMSE  Max_RMSE  Mean_MAE  Std_MAE
0          Kerr    0.09823   0.01234   0.08456   0.11234   0.07234  0.00987
1  Schwarzschild  0.10012   0.01456   0.08789   0.11567   0.07456  0.01123
2      Baseline    0.10140   0.01567   0.08923   0.12345   0.07567  0.01234
3         GARCH    0.10170   0.01678   0.09012   0.12456   0.07678  0.01345
```

**Action Plan:**
1. `models/cross_validation.py` oluştur
2. `main_cross_validation.py` oluştur
3. Tüm modelleri CV ile değerlendir (6-8 saat)
4. Sonuçları raporla

---

## 🚀 UZUN VADELİ GELİŞTİRMELER (PIML - Hipotez_04)

### 🔬 **GELİŞTİRME 1: Gravitational Residual Network (GRN)**

**Ne:** Bükülme fonksiyonunu öğrenen bir sinir ağı

**Teorik Temel:**
- PINN benzeri yaklaşım
- Physics-informed inductive bias
- Öğrenilebilir parametreler

**Mimari:**

```python
# Yeni dosya: models/grn_network.py

import torch
import torch.nn as nn
import numpy as np

class GravitationalResidualNetwork(nn.Module):
    """
    Physics-inspired neural network for learning curvature function.
    
    Architecture:
        Input: [M(t), a(t), τ(t), ε(t-k:t)] → hidden layers → Output: Γ(t+1)
    
    Physics-informed constraints:
        1. Monotonicity: ∂Γ/∂M ≥ 0 (larger mass → larger curvature)
        2. Energy conservation: Σ|Γ(t)| is bounded
        3. Symmetry: Γ(M, a, τ) = -Γ(M, -a, τ) for spin
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_sizes: List[int] = [64, 32, 16],
        output_size: int = 1,
        use_monotonicity: bool = True,
        use_energy_conservation: bool = True
    ):
        super().__init__()
        
        self.use_monotonicity = use_monotonicity
        self.use_energy_conservation = use_energy_conservation
        
        # Encoder network
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Bounded output [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Learnable physics parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, mass, spin, tau, residuals_history):
        """
        Forward pass.
        
        Parameters
        ----------
        mass : torch.Tensor, shape (batch, 1)
            Kütle (volatilite)
        spin : torch.Tensor, shape (batch, 1)
            Dönme (otokorelasyon)
        tau : torch.Tensor, shape (batch, 1)
            Şoktan geçen zaman
        residuals_history : torch.Tensor, shape (batch, seq_len)
            Geçmiş artıklar dizisi
            
        Returns
        -------
        curvature : torch.Tensor, shape (batch, 1)
            Bükülme düzeltmesi
        """
        # Decay factor
        decay = 1.0 / (1.0 + self.beta * tau)
        
        # Input features
        x = torch.cat([mass, spin, tau, residuals_history[:, -1:]], dim=1)
        
        # Neural network correction
        nn_correction = self.network(x)
        
        # Physics-inspired base term
        base_term = self.alpha * mass * torch.tanh(residuals_history[:, -1:])
        spin_term = self.gamma * spin * residuals_history[:, -1:]
        
        # Combined output
        curvature = (base_term + spin_term + nn_correction) * decay
        
        return curvature
    
    def physics_loss(self, mass, curvature):
        """
        Physics-informed loss term.
        
        Enforces:
        1. Monotonicity: dΓ/dM ≥ 0
        2. Energy conservation: Total energy bounded
        """
        loss = 0.0
        
        if self.use_monotonicity:
            # Monotonicity constraint
            # Approximate derivative using finite differences
            mass_perturbed = mass + 0.01
            curvature_perturbed = self.forward(mass_perturbed, ...)
            
            derivative = (curvature_perturbed - curvature) / 0.01
            monotonicity_loss = torch.relu(-derivative).mean()  # Penalize negative derivatives
            loss += 0.1 * monotonicity_loss
        
        if self.use_energy_conservation:
            # Energy conservation: penalize large total energy
            total_energy = torch.sum(torch.abs(curvature))
            energy_loss = torch.relu(total_energy - 10.0)  # Soft threshold
            loss += 0.01 * energy_loss
        
        return loss
    
    def combined_loss(self, predictions, targets, mass, curvature):
        """
        Combined loss: Data fidelity + Physics-informed.
        
        L_total = L_data + λ * L_physics
        """
        # Data fidelity loss
        data_loss = nn.MSELoss()(predictions, targets)
        
        # Physics loss
        physics_loss = self.physics_loss(mass, curvature)
        
        # Combined
        total_loss = data_loss + 0.1 * physics_loss
        
        return total_loss, data_loss, physics_loss
```

**Training Loop:**

```python
# Yeni dosya: models/grn_trainer.py

class GRNTrainer:
    """GRN model trainer with physics-informed loss."""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            mass, spin, tau, residuals_history, targets = batch
            
            # Forward pass
            predictions = self.model(mass, spin, tau, residuals_history)
            
            # Loss
            loss, data_loss, physics_loss = self.model.combined_loss(
                predictions, targets, mass, predictions
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mass, spin, tau, residuals_history, targets = batch
                predictions = self.model(mass, spin, tau, residuals_history)
                loss, _, _ = self.model.combined_loss(
                    predictions, targets, mass, predictions
                )
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=100, early_stopping=10):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/grn_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/grn_best.pth'))
```

**Beklenen İyileşme:**
- Manuel fonksiyondan %5-10 daha iyi
- Farklı varlıklara genellenebilir
- Fiziksel kısıtlamalar sayesinde yorumlanabilir

**Action Plan:**
1. PyTorch kur: `pip install torch`
2. `models/grn_network.py` oluştur
3. `models/grn_trainer.py` oluştur
4. `main_grn_train.py` oluştur ve eğit
5. Manuel fonksiyon vs GRN karşılaştır

**Süre:** 1-2 hafta (veri hazırlama + eğitim + test)

---

### 🌌 **GELİŞTİRME 2: Uçtan Uca Birleşik Model (End-to-End)**

**Ne:** Baseline + GRM'yi tek bir modelde birleştir

**Teorik Temel:**
- PINN-style joint training
- Baseline ve GRM birbirinden öğrenir
- Global optimum

**Mimari:**

```python
# Yeni dosya: models/unified_grm.py

class UnifiedGRM(nn.Module):
    """
    End-to-end unified model: Baseline + GRM in one network.
    
    Architecture:
        Input: X(t-k:t) → [LSTM Baseline] → Ŷ(t)
                       ↓
                  [Residuals]
                       ↓
        [GRN Network] → Γ(t)
                       ↓
        Final: Ŷ(t) + Γ(t)
    
    Loss: L = L_data(Y, Ŷ+Γ) + λ₁*L_baseline(Y, Ŷ) + λ₂*L_physics(Γ)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        lstm_hidden_size: int = 64,
        grn_hidden_sizes: List[int] = [32, 16]
    ):
        super().__init__()
        
        # Baseline LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.lstm_output = nn.Linear(lstm_hidden_size, 1)
        
        # GRN for residual correction
        self.grn = GravitationalResidualNetwork(
            input_size=4,  # M, a, τ, ε
            hidden_sizes=grn_hidden_sizes,
            output_size=1
        )
    
    def forward(self, x_history):
        """
        Unified forward pass.
        
        Parameters
        ----------
        x_history : torch.Tensor, shape (batch, seq_len, 1)
            Historical time series
            
        Returns
        -------
        baseline_pred : torch.Tensor
            Baseline LSTM prediction
        grm_correction : torch.Tensor
            GRM correction
        final_pred : torch.Tensor
            Final combined prediction
        """
        # Baseline LSTM prediction
        lstm_out, _ = self.lstm(x_history)
        baseline_pred = self.lstm_output(lstm_out[:, -1, :])
        
        # Compute residuals (from training data, approximated)
        # In real implementation, this needs historical residuals
        residuals = x_history[:, :, 0] - baseline_pred.detach()
        
        # Compute GRM features
        mass = torch.var(residuals, dim=1, keepdim=True)
        spin = self.compute_autocorr(residuals)
        tau = torch.ones_like(mass) * 5.0  # Simplified
        
        # GRM correction
        grm_correction = self.grn(mass, spin, tau, residuals)
        
        # Final prediction
        final_pred = baseline_pred + grm_correction
        
        return baseline_pred, grm_correction, final_pred
    
    def compute_autocorr(self, residuals):
        """Compute lag-1 autocorrelation."""
        # Simplified implementation
        r1 = residuals[:, 1:]
        r0 = residuals[:, :-1]
        corr = torch.mean(r1 * r0, dim=1, keepdim=True) / (torch.var(residuals, dim=1, keepdim=True) + 1e-8)
        return corr
    
    def combined_loss(self, baseline_pred, grm_correction, final_pred, targets):
        """
        Three-component loss function.
        
        L_total = L_final + λ₁*L_baseline + λ₂*L_physics
        """
        # Main loss: final prediction vs targets
        loss_final = nn.MSELoss()(final_pred, targets)
        
        # Baseline loss: encourage baseline to be reasonable
        loss_baseline = nn.MSELoss()(baseline_pred, targets)
        
        # Physics loss: GRM constraints
        mass = torch.var(residuals, dim=1, keepdim=True)  # Need proper implementation
        loss_physics = self.grn.physics_loss(mass, grm_correction)
        
        # Weighted combination
        total_loss = loss_final + 0.1 * loss_baseline + 0.05 * loss_physics
        
        return total_loss, loss_final, loss_baseline, loss_physics
```

**Avantajlar:**
1. **Joint Optimization:** Baseline ve GRM birlikte optimize edilir
2. **Information Flow:** İki yönlü bilgi akışı
3. **End-to-End Learning:** Global optimum arayışı

**Beklenen İyileşme:**
- Manuel + ayrık yaklaşımdan %10-15 daha iyi
- Daha stabil tahminler
- Daha iyi genelleme

**Action Plan:**
1. `models/unified_grm.py` oluştur
2. Eğitim pipeline kur
3. Ayrık model vs Unified model karşılaştır

**Süre:** 2-3 hafta

---

### 🧬 **GELİŞTİRME 3: Symbolic Regression ile Dinamik Keşfi**

**Ne:** Veriden optimal bükülme fonksiyonunu otomatik keşfet

**Teorik Temel:**
- AI Feynman, PySR
- Genetic programming
- Interpretable formulas

**İmplementasyon:**

```python
# Yeni dosya: models/symbolic_discovery.py

from pysr import PySRRegressor
import numpy as np

class SymbolicGRM:
    """
    Symbolic regression ile bükülme fonksiyonunu keşfet.
    
    PySR kullanarak en iyi sembolik denklemi bul:
    Γ(t) = f(M(t), a(t), τ(t), ε(t))
    """
    
    def __init__(self):
        self.model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "tanh", "abs"],
            model_selection="best",
            loss="loss(x, y) = (x - y)^2",
            maxsize=20,
            populations=15
        )
    
    def prepare_features(self, residuals, window_size=20):
        """
        GRM feature'larını hazırla.
        
        Returns
        -------
        X : np.ndarray, shape (n_samples, 4)
            [M(t), a(t), τ(t), ε(t)]
        y : np.ndarray, shape (n_samples,)
            Hedef: gelecekteki artık veya düzeltme
        """
        n = len(residuals)
        X = []
        y = []
        
        for t in range(window_size, n - 1):
            window = residuals[t - window_size:t]
            
            # Features
            mass = np.var(window)
            spin = np.corrcoef(window[1:], window[:-1])[0, 1] if len(window) > 1 else 0.0
            tau = self.compute_tau(residuals[:t], threshold=2.0)
            epsilon = residuals[t]
            
            X.append([mass, spin, tau, epsilon])
            
            # Target: next residual or ideal correction
            y.append(residuals[t + 1])
        
        return np.array(X), np.array(y)
    
    def compute_tau(self, residuals, threshold=2.0):
        """Time since last shock."""
        abs_res = np.abs(residuals)
        shock_indices = np.where(abs_res > threshold)[0]
        
        if len(shock_indices) == 0:
            return len(residuals)
        
        last_shock = shock_indices[-1]
        tau = len(residuals) - last_shock
        return tau
    
    def discover_formula(self, residuals, window_size=20):
        """
        Sembolik denklemi keşfet.
        
        Returns
        -------
        best_formula : str
            En iyi sembolik denklem (e.g., "0.5*M*tanh(epsilon) + 0.1*a*epsilon")
        """
        # Feature hazırlama
        X, y = self.prepare_features(residuals, window_size)
        
        # Feature isimleri
        feature_names = ["M", "a", "tau", "epsilon"]
        
        # Symbolic regression
        print("Sembolik regresyon başlatılıyor...")
        print("(Bu işlem 10-30 dakika sürebilir)")
        
        self.model.fit(X, y, variable_names=feature_names)
        
        # En iyi formül
        best_formula = self.model.get_best()
        
        print("\n" + "="*80)
        print("KEŞFEDILEN FORMÜL")
        print("="*80)
        print(f"Γ(t) = {best_formula}")
        print(f"R² Score: {self.model.score(X, y):.4f}")
        print("="*80)
        
        # Tüm adayları göster
        print("\nTÜM ADAY FORMÜLLER (Complexity vs Accuracy):")
        print(self.model.equations_)
        
        return best_formula
    
    def predict(self, M, a, tau, epsilon):
        """Keşfedilen formülü kullanarak tahmin yap."""
        X = np.column_stack([M, a, tau, epsilon])
        return self.model.predict(X)
```

**Kullanım:**

```python
# main_symbolic_discovery.py

from models.symbolic_discovery import SymbolicGRM

# Baseline residuals
baseline = BaselineARIMA()
baseline.fit(train_df['y'])
residuals = baseline.get_residuals()

# Symbolic discovery
symbolic_grm = SymbolicGRM()
formula = symbolic_grm.discover_formula(residuals, window_size=20)

# Örnek çıktı:
# Γ(t) = 0.523*M*tanh(epsilon) + 0.187*a*epsilon*exp(-0.05*tau)
```

**Avantajlar:**
1. **Data-Driven:** Veri kendi formülünü yazıyor
2. **Interpretable:** Sembolik formül, yorumlanabilir
3. **Discovery:** Beklenmedik ilişkiler keşfedilebilir

**Beklenen Çıktı:**
```
================================================================================
KEŞFEDILEN FORMÜL
================================================================================
Γ(t) = 0.523*M*tanh(epsilon) + 0.187*a*epsilon*exp(-0.05*tau) - 0.034*M^2
R² Score: 0.8234
================================================================================

TÜMADAY FORMÜLLER (Complexity vs Accuracy):
   complexity                               equation     loss    score
0           5                    0.523*M*tanh(epsilon)  0.0234  0.7845
1           8      0.523*M*tanh(epsilon) + 0.187*a*epsilon  0.0198  0.8012
2          12  0.523*M*tanh(epsilon) + 0.187*a*epsilon*exp(-0.05*tau)  0.0165  0.8234
3          15  ... - 0.034*M^2  0.0163  0.8245
```

**Action Plan:**
1. PySR kur: `pip install pysr`
2. `models/symbolic_discovery.py` oluştur
3. `main_symbolic_discovery.py` oluştur
4. Sembolik regresyon çalıştır (30-60 dakika)
5. Keşfedilen formülü manual formül ile karşılaştır

**Süre:** 3-5 gün

---

### 🔮 **GELİŞTİRME 4: N-Body Problem - Çoklu Kara Delik**

**Ne:** Birden fazla şok kaynağını (çoklu kara delik) modelle

**Teorik Temel:**
- N-body gravitational simulation
- Regime switching models
- Clustering algorithms

**Konsept:**

```
Tek Kara Delik (Mevcut):
    [────────────●────────────]
              M(t)

Çoklu Kara Delik (Önerilen):
    [──●₁────●₂──────●₃───────]
       M₁    M₂       M₃
    
    Γ_total(t) = Σᵢ Γᵢ(t, Mᵢ, aᵢ, τᵢ)
```

**İmplementasyon:**

```python
# Yeni dosya: models/multi_body_grm.py

from sklearn.cluster import DBSCAN
import numpy as np

class MultiBodyGRM:
    """
    N-body GRM: Birden fazla gravitational anomaly.
    
    Yaklaşım:
    1. Artıkları farklı rejimlere kümeleyerek ayır (DBSCAN, HMM)
    2. Her rejim = bir "kara delik"
    3. Her kara delik için ayrı parametreler (Mi, ai, τi)
    4. Toplam etki = süperpozisyon: Γ = Σᵢ Γᵢ
    """
    
    def __init__(self, n_bodies=3, window_size=20):
        self.n_bodies = n_bodies
        self.window_size = window_size
        self.body_params = []  # Her body için (alpha, beta, gamma)
    
    def cluster_residuals(self, residuals):
        """
        Artıkları farklı rejimlere ayır (clustering).
        
        Returns
        -------
        regime_labels : np.ndarray
            Her zaman noktası için rejim etiketi (0, 1, 2, ...)
        """
        # Feature engineering: rolling statistics
        features = []
        for t in range(self.window_size, len(residuals)):
            window = residuals[t - self.window_size:t]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                self.autocorr(window)
            ])
        
        features = np.array(features)
        
        # DBSCAN clustering
        clusterer = DBSCAN(eps=0.5, min_samples=10)
        regime_labels = clusterer.fit_predict(features)
        
        print(f"Tespit edilen rejim sayısı: {len(np.unique(regime_labels))}")
        
        return regime_labels
    
    def fit_body(self, regime_residuals, body_id):
        """Bir body (rejim) için parametreleri optimize et."""
        # Her body için ayrı SchwarzschildGRM fit et
        grm = SchwarzschildGRM(window_size=self.window_size)
        grm.fit(regime_residuals)
        
        params = {
            'body_id': body_id,
            'alpha': grm.alpha,
            'beta': grm.beta,
            'n_samples': len(regime_residuals)
        }
        
        self.body_params.append(params)
        return params
    
    def fit(self, residuals):
        """Tüm body'leri fit et."""
        # Rejimleri tespit et
        regime_labels = self.cluster_residuals(residuals)
        
        # Her rejim için ayrı fit
        for regime_id in np.unique(regime_labels):
            if regime_id == -1:  # Noise points
                continue
            
            regime_mask = regime_labels == regime_id
            regime_residuals = residuals[self.window_size:][regime_mask]
            
            print(f"\nBody {regime_id} eğitiliyor ({len(regime_residuals)} sample)...")
            params = self.fit_body(regime_residuals, regime_id)
            print(f"  alpha={params['alpha']:.4f}, beta={params['beta']:.4f}")
    
    def compute_curvature(self, residuals, current_regime):
        """
        Toplam bükülmeyi hesapla (süperpozisyon).
        
        Γ_total = Σᵢ wᵢ * Γᵢ
        
        wᵢ: Her body'nin ağırlığı (güncel rejime uzaklığa göre)
        """
        total_curvature = 0.0
        
        for params in self.body_params:
            body_id = params['body_id']
            alpha = params['alpha']
            beta = params['beta']
            
            # Weight: eğer güncel rejim bu body'ye yakınsa, ağırlık yüksek
            if body_id == current_regime:
                weight = 1.0
            else:
                weight = 0.1  # Diğer body'lerin zayıf etkisi
            
            # Bu body'nin katkısı
            mass = np.var(residuals[-self.window_size:])
            gamma_i = alpha * mass * np.sign(residuals[-1])
            
            total_curvature += weight * gamma_i
        
        return total_curvature
    
    def autocorr(self, x, lag=1):
        """Autocorrelation hesapla."""
        if len(x) <= lag:
            return 0.0
        return np.corrcoef(x[lag:], x[:-lag])[0, 1]
```

**Beklenen İyileşme:**
- Tek body'den %5-10 daha iyi
- Farklı şok türlerini (pozitif/negatif, kısa/uzun) ayırt edebilir
- Rejim geçişlerini yakalayabilir

**Action Plan:**
1. `models/multi_body_grm.py` oluştur
2. Clustering ve regime detection test et
3. Tek body vs Multi-body karşılaştır

**Süre:** 1-2 hafta

---

## 📋 ÖNERİLEN UYGULAMA SIRASI

### **FAZE 4: Zenginleştirme (1-2 hafta)**
1. ✅ Decay factor ekle (Öncelik 1) - 2 gün
2. ✅ Ablasyon çalışması (Öncelik 2) - 3 gün
3. ✅ Time-series CV (Öncelik 3) - 2 gün
4. ✅ Sonuçları raporla ve analiz et - 1 gün

**Beklenen Sonuç:** RMSE 0.102 → 0.095 (%7 iyileşme)

---

### **FAZE 5: PIML Entegrasyonu - Temel (2-3 hafta)**
1. ✅ GRN (Gravitational Residual Network) - 1 hafta
2. ✅ Manuel vs GRN karşılaştırması - 2 gün
3. ✅ Symbolic regression pilot - 3 gün

**Beklenen Sonuç:** RMSE 0.095 → 0.085 (%10 iyileşme)

---

### **FAZE 6: PIML İleri Seviye (1-2 ay)**
1. ✅ Unified end-to-end model - 2 hafta
2. ✅ Multi-body GRM - 1 hafta
3. ✅ Kapsamlı karşılaştırma ve raporlama - 1 hafta

**Beklenen Sonuç:** RMSE 0.085 → 0.075 (%15 toplam iyileşme)

---

## 🎓 AKADEMİK YAYIM STRATEJİSİ

### **Yayın 1: Mevcut Sistem (Hazır)**
**Başlık:** "Gravitational Residual Model (GRM): A Physics-Inspired Framework for Time Series Residual Modeling"

**İçerik:**
- Faze 1-3 sonuçları
- Schwarzschild & Kerr rejimleri
- GARCH karşılaştırması
- Ablasyon çalışması

**Hedef:** Time Series Analysis konferansı veya dergisi

---

### **Yayın 2: PIML Entegrasyonu (6 ay sonra)**
**Başlık:** "Physics-Informed Neural Networks for Gravitational Residual Modeling: Learning Curvature Functions from Data"

**İçerik:**
- GRN mimarisi
- Physics-informed loss
- Symbolic regression sonuçları
- Manuel vs öğrenilmiş fonksiyon

**Hedef:** ICML, NeurIPS, ICLR (PIML workshop)

---

### **Yayın 3: Unified System (1 yıl sonra)**
**Başlık:** "End-to-End Gravitational Time Series Modeling: A Unified Framework Combining Baseline Forecasting and Residual Dynamics"

**İçerik:**
- Unified architecture
- Multi-body extensions
- Kapsamlı benchmark (GARCH, LSTM, Transformer)
- Gerçek dünya uygulamaları

**Hedef:** Nature Machine Intelligence, JMLR

---

## 📊 ÖZET: HIZLI KAZANIMLAR

### **BU HAFTA (1-2 gün):**
1. Decay factor ekle → %2-3 iyileşme
2. Event horizon istatistiksel tanım → Objektiflik

### **BU AY (2-3 hafta):**
1. Ablasyon çalışması → Hangi bileşen kritik?
2. Time-series CV → Sağlamlık testi
3. İlk GRN denemesi → %5-8 iyileşme

### **3 AY (PIML pilot):**
1. GRN üretim sürümü → %10-12 iyileşme
2. Symbolic discovery → Yeni formül keşfi
3. İlk makale hazırlığı

### **6-12 AY (Tam PIML sistemi):**
1. Unified model → %15-20 toplam iyileşme
2. Multi-body extensions → Rejim switching
3. Kapsamlı akademik yayın

---

## 🎯 SONUÇ

**Mevcut Projeniz:** Metodolojik olarak sağlam, iyi yapılandırılmış ✅  
**Ana Limitasyon:** Manuel bükülme fonksiyonu, tek body varsayımı ⚠️  
**En Önemli İyileştirme:** GRN (öğrenilebilir fonksiyon) 🚀  
**Akademik Potansiyel:** Çok yüksek (3 makale + PIML alanına katkı) 🎓

**İlk Adım Önerisi:** Decay factor ekleyip ablasyon çalışması yapın. Bu, hızlı kazanım sağlar ve PIML'e geçiş için zemin hazırlar.

Hangi önceliklendirmeyle başlamak istersiniz? 🔧

