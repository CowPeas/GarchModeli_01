# 🚀 GRM PROJESİ: BÜTÜNCÜL GELİŞTİRME İMPLEMENTASYON PLANI

**Tarih:** 2025-11-15  
**Versiyon:** 1.0  
**Hedef:** Hipotez_03 ve Hipotez_04'teki tüm geliştirmeleri sistematik olarak uygulamak

---

## 📋 İÇİNDEKİLER

1. [FAZE 4: Zenginleştirme (Öncelikli)](#faze-4-zenginleştirme-öncelikli)
2. [FAZE 5: PIML Temel Entegrasyonu](#faze-5-piml-temel-entegrasyonu)
3. [FAZE 6: PIML İleri Seviye](#faze-6-piml-ileri-seviye)
4. [Genel Dosya Yapısı](#genel-dosya-yapısı)
5. [Test ve Doğrulama Stratejisi](#test-ve-doğrulama-stratejisi)

---

## 🎯 FAZE 4: ZENGİNLEŞTİRME (ÖNCELİKLİ)

### **ADIM 4.1: Decay Factor ve Event Horizon İyileştirmesi**

#### **4.1.1: Mevcut Dosyaları Güncelle**

**Dosya:** `models/grm_model.py`

**Algoritma:**
```
1. MEVCUT METODLARI GÜNCELLE:
   
   1.1. compute_curvature() metodunu genişlet:
        - Parametreler: residuals, mass, time_since_shock
        - Decay factor hesapla: decay = 1 / (1 + β * τ)
        - Base curvature: α * M(t) * tanh(ε(t))
        - Final: base_curvature * decay
        - Return: curvature array
   
   1.2. detect_shocks() metodunu ekle:
        - Parametreler: residuals, threshold_quantile=0.95
        - abs_residuals = |residuals|
        - threshold = quantile(abs_residuals, threshold_quantile)
        - shock_times = where(abs_residuals > threshold)
        - self.shock_times = shock_times
        - Return: shock_times array
   
   1.3. compute_time_since_shock() metodunu ekle:
        - Parametreler: current_time, shock_times
        - IF len(shock_times) == 0 OR current_time < shock_times[0]:
            RETURN float('inf')
        - past_shocks = shock_times[shock_times < current_time]
        - IF len(past_shocks) == 0:
            RETURN float('inf')
        - last_shock = past_shocks[-1]
        - tau = current_time - last_shock
        - RETURN tau
   
   1.4. compute_curvature_with_decay() metodunu ekle:
        - Parametreler: residuals, mass, time_since_shock
        - decay = 1.0 / (1.0 + self.beta * time_since_shock)
        - base_curvature = self.alpha * mass * np.tanh(residuals)
        - curvature = base_curvature * decay
        - RETURN curvature

2. compute_event_horizon() metodunu güncelle:
   - Mevcut: quantile(mass, 0.99) veya mean + 3*std
   - YENİ: İstatistiksel tanım ekle
   - threshold = quantile(mass, quantile) VEYA mean(mass) + 3*std(mass)
   - self.shock_threshold = threshold
   - RETURN threshold

3. __init__() metodunu güncelle:
   - YENİ parametre: use_decay=True
   - YENİ parametre: shock_threshold_quantile=0.95
   - self.use_decay = use_decay
   - self.shock_threshold_quantile = shock_threshold_quantile
```

**Dosya:** `models/kerr_grm_model.py`

**Algoritma:**
```
1. compute_curvature() metodunu genişlet:
   - Decay factor desteği ekle
   - IF self.use_decay:
       decay = 1 / (1 + β * τ)
       curvature = curvature * decay
   - RETURN curvature

2. detect_shocks() ve compute_time_since_shock() metodlarını ekle
   (SchwarzschildGRM ile aynı)

3. __init__() metodunu güncelle:
   - use_decay parametresi ekle
   - shock_threshold_quantile parametresi ekle
```

**Dosya:** `config_phase3.py`

**Algoritma:**
```
1. SCHWARZSCHILD_CONFIG'a ekle:
   'use_decay': True,
   'decay_beta_range': [0.01, 0.05, 0.1, 0.2],
   'shock_threshold_quantile': 0.95,
   'shock_detection_method': 'quantile'  # 'quantile' veya 'statistical'

2. KERR_CONFIG'a ekle:
   'use_decay': True,
   'decay_beta_range': [0.01, 0.05, 0.1, 0.2],
   'shock_threshold_quantile': 0.95,
   'shock_detection_method': 'quantile'
```

**Dosya:** `main_phase3.py`

**Algoritma:**
```
1. walk_forward_predict_grm() fonksiyonunu güncelle:
   
   FOR i in range(len(test_data)):
       # 1. Baseline tahmin
       baseline_pred = baseline_model.predict(1)[0]
       
       # 2. Şok tespiti (ilk iterasyonda)
       IF i == 0:
           shock_times = grm_model.detect_shocks(all_residuals)
       
       # 3. Time since shock hesapla
       tau = grm_model.compute_time_since_shock(
           current_time=len(all_residuals),
           shock_times=shock_times
       )
       
       # 4. GRM düzeltmesi (decay ile)
       recent_residuals = all_residuals[-grm_model.window_size:]
       mass = grm_model.compute_mass(recent_residuals)[-1]
       
       IF hasattr(grm_model, 'compute_spin'):
           # Kerr
           spin = grm_model.compute_spin(recent_residuals)[-1]
           correction = grm_model.compute_curvature_with_decay(
               recent_residuals[-1:],
               mass,
               tau
           )
       ELSE:
           # Schwarzschild
           correction = grm_model.compute_curvature_with_decay(
               recent_residuals[-1:],
               mass,
               tau
           )
       
       # 5. Final tahmin
       final_pred = baseline_pred + correction
       
       # 6. Gerçek değeri gözlemle
       actual = test_data.iloc[i]
       residual = actual - baseline_pred
       all_residuals.append(residual)
       
       # 7. Baseline'ı güncelle
       baseline_model.fitted_model.append([actual], refit=False)
```

---

### **ADIM 4.2: Ablasyon Çalışması İmplementasyonu**

#### **4.2.1: Yeni Dosya Oluştur**

**Dosya:** `models/ablation_study.py`

**Algoritma:**
```
1. IMPORTS:
   import numpy as np
   import pandas as pd
   from typing import Dict, List, Tuple
   from models.baseline_model import BaselineARIMA
   from models.grm_model import SchwarzschildGRM
   from models.kerr_grm_model import KerrGRM
   from models.metrics import calculate_rmse, calculate_mae

2. CLASS AblationStudy:
   
   2.1. __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.results = {}
        self.baseline_rmse = None
   
   2.2. run_baseline():
        baseline = BaselineARIMA()
        baseline.fit(self.train_data)
        train_residuals = baseline.get_residuals()
        
        # Grid search (val kullanarak)
        best_order = baseline.grid_search(
            self.train_data, self.val_data,
            p_range=[0, 1, 2],
            d_range=[0, 1],
            q_range=[0, 1, 2]
        )
        
        # Sadece train ile fit
        baseline.fit(self.train_data, order=best_order)
        
        # Test predictions (walk-forward)
        predictions = self.walk_forward_predict(
            baseline, self.test_data
        )
        
        rmse = calculate_rmse(self.test_data, predictions)
        self.baseline_rmse = rmse
        self.results['Baseline'] = {
            'rmse': rmse,
            'mae': calculate_mae(self.test_data, predictions),
            'components': []
        }
        RETURN rmse
   
   2.3. walk_forward_predict(model, test_data):
        predictions = []
        for i in range(len(test_data)):
            pred = model.predict(steps=1)[0]
            predictions.append(pred)
            IF i < len(test_data) - 1:
                model.fitted_model.append(
                    [test_data.iloc[i]], refit=False
                )
        RETURN np.array(predictions)
   
   2.4. run_grm_variant(name, model_class, **kwargs):
        # Baseline fit
        baseline = BaselineARIMA()
        baseline.fit(self.train_data)
        train_residuals = baseline.get_residuals()
        
        # GRM fit
        grm_model = model_class(**kwargs)
        grm_model.fit(train_residuals)
        
        # Test predictions (walk-forward GRM)
        predictions = self.walk_forward_predict_grm(
            baseline, grm_model, self.test_data
        )
        
        rmse = calculate_rmse(self.test_data, predictions)
        improvement = (self.baseline_rmse - rmse) / self.baseline_rmse * 100
        
        self.results[name] = {
            'rmse': rmse,
            'mae': calculate_mae(self.test_data, predictions),
            'components': list(kwargs.keys()),
            'improvement': improvement
        }
        RETURN rmse
   
   2.5. walk_forward_predict_grm(baseline, grm, test_data):
        predictions = []
        all_residuals = list(baseline.get_residuals())
        
        FOR i in range(len(test_data)):
            # Baseline tahmin
            baseline_pred = baseline.predict(1)[0]
            
            # GRM düzeltmesi
            recent_residuals = all_residuals[-grm.window_size:]
            mass = grm.compute_mass(recent_residuals)[-1]
            
            IF hasattr(grm, 'compute_spin'):
                spin = grm.compute_spin(recent_residuals)[-1]
                correction = grm.compute_curvature(...)
            ELSE:
                correction = grm.compute_curvature(...)
            
            final_pred = baseline_pred + correction
            predictions.append(final_pred)
            
            # Gerçek değeri gözlemle
            actual = test_data.iloc[i]
            residual = actual - baseline_pred
            all_residuals.append(residual)
            
            # Baseline'ı güncelle
            baseline.fitted_model.append([actual], refit=False)
        
        RETURN np.array(predictions)
   
   2.6. test_mass_only():
        RETURN self.run_grm_variant(
            name='Mass_Only',
            model_class=SchwarzschildGRM,
            window_size=20,
            use_decay=False  # Decay yok
        )
   
   2.7. test_mass_with_decay():
        RETURN self.run_grm_variant(
            name='Mass_Decay',
            model_class=SchwarzschildGRM,
            window_size=20,
            use_decay=True,
            beta=0.05
        )
   
   2.8. test_kerr_full():
        RETURN self.run_grm_variant(
            name='Kerr_Full',
            model_class=KerrGRM,
            window_size=20,
            use_decay=True,
            use_tanh=True,
            beta=0.05,
            gamma=0.5
        )
   
   2.9. test_kerr_no_decay():
        RETURN self.run_grm_variant(
            name='Kerr_No_Decay',
            model_class=KerrGRM,
            window_size=20,
            use_decay=False,
            use_tanh=True,
            gamma=0.5
        )
   
   2.10. test_kerr_linear():
        RETURN self.run_grm_variant(
            name='Kerr_Linear',
            model_class=KerrGRM,
            window_size=20,
            use_decay=True,
            use_tanh=False,  # Linear
            beta=0.05,
            gamma=0.5
        )
   
   2.11. test_window_sizes(sizes=[10, 20, 30, 50, 100]):
        FOR w in sizes:
            self.run_grm_variant(
                name=f'Window_{w}',
                model_class=KerrGRM,
                window_size=w,
                use_decay=True,
                use_tanh=True
            )
   
   2.12. generate_report():
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.sort_values('improvement', ascending=False)
        
        PRINT "="*80
        PRINT "ABLASYON ÇALIŞMASI SONUÇLARI"
        PRINT "="*80
        PRINT df.to_string()
        
        best = df.iloc[0]
        PRINT f"EN İYİ: {best.name} (RMSE={best['rmse']:.6f}, İyileşme={best['improvement']:.2f}%)"
        
        RETURN df
   
   2.13. plot_results():
        # Matplotlib ile görselleştirme
        # 1. RMSE karşılaştırması (bar chart)
        # 2. İyileşme yüzdeleri (barh chart)
        # 3. Pencere boyutu hassasiyeti (line chart)
        # 4. Bileşen katkıları (heatmap)
        # Kaydet: results/ablation_study.png
```

**Dosya:** `main_ablation_study.py`

**Algoritma:**
```
1. IMPORTS:
   from models.ablation_study import AblationStudy
   from models.real_data_loader import RealDataLoader
   from models.alternative_data_loader import AlternativeDataLoader
   from config_phase3 import REAL_DATA_CONFIG, SPLIT_CONFIG

2. VERI YUKLEME:
   loader = RealDataLoader()
   df, metadata = loader.load_yahoo_finance(...)
   
   IF df is None:
       alt_loader = AlternativeDataLoader()
       df = alt_loader.generate_realistic_crypto_data(...)

3. VERI BOLME:
   train_df, val_df, test_df = split_data(df, **SPLIT_CONFIG)

4. ABLASYON CALISMASI:
   study = AblationStudy(
       train_df['y'],
       val_df['y'],
       test_df['y']
   )
   
   # Baseline
   study.run_baseline()
   
   # Ablasyonlar
   study.test_mass_only()
   study.test_mass_with_decay()
   study.test_kerr_full()
   study.test_kerr_no_decay()
   study.test_kerr_linear()
   
   # Hassasiyet analizleri
   study.test_window_sizes([10, 20, 30, 50, 100])
   
   # Rapor
   results_df = study.generate_report()
   study.plot_results()
   
   # Kaydet
   results_df.to_csv('results/ablation_results.csv')
```

---

### **ADIM 4.3: Time-Series Cross-Validation İmplementasyonu**

#### **4.3.1: Yeni Dosya Oluştur**

**Dosya:** `models/cross_validation.py`

**Algoritma:**
```
1. IMPORTS:
   import numpy as np
   import pandas as pd
   from typing import List, Tuple, Dict
   from models.baseline_model import BaselineARIMA
   from models.grm_model import SchwarzschildGRM
   from models.kerr_grm_model import KerrGRM
   from models.metrics import calculate_rmse, calculate_mae

2. CLASS TimeSeriesCrossValidator:
   
   2.1. __init__(initial_train_size, val_size, test_size, step_size):
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
   
   2.2. split(data):
        n = len(data)
        folds = []
        current_train_end = self.initial_train_size
        
        WHILE current_train_end + self.val_size + self.test_size <= n:
            train_indices = arange(0, current_train_end)
            val_indices = arange(
                current_train_end,
                current_train_end + self.val_size
            )
            test_indices = arange(
                current_train_end + self.val_size,
                current_train_end + self.val_size + self.test_size
            )
            
            folds.append((train_indices, val_indices, test_indices))
            current_train_end += self.step_size
        
        RETURN folds
   
   2.3. evaluate_model(model_class, data, **model_kwargs):
        folds = self.split(data)
        results = {
            'rmse': [],
            'mae': [],
            'fold': []
        }
        
        FOR i, (train_idx, val_idx, test_idx) in enumerate(folds):
            PRINT f"  Fold {i+1}/{len(folds)}..."
            
            train_data = data[train_idx]
            val_data = data[val_idx]
            test_data = data[test_idx]
            
            # Model train
            IF model_class == BaselineARIMA:
                model = model_class()
                best_order = model.grid_search(train_data, val_data)
                model.fit(train_data, order=best_order)
            ELSE:
                # GRM: önce baseline, sonra GRM
                baseline = BaselineARIMA()
                baseline.fit(train_data)
                train_residuals = baseline.get_residuals()
                
                model = model_class(**model_kwargs)
                model.fit(train_residuals)
                model.baseline = baseline  # Baseline'ı sakla
            
            # Test predict (walk-forward)
            predictions = self.walk_forward_predict(model, test_data)
            
            # Metrics
            rmse = calculate_rmse(test_data, predictions)
            mae = calculate_mae(test_data, predictions)
            
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['fold'].append(i + 1)
        
        RETURN results
   
   2.4. walk_forward_predict(model, test_data):
        IF hasattr(model, 'baseline'):
            # GRM model
            RETURN self.walk_forward_predict_grm(
                model.baseline, model, test_data
            )
        ELSE:
            # Baseline model
            predictions = []
            FOR i in range(len(test_data)):
                pred = model.predict(1)[0]
                predictions.append(pred)
                IF i < len(test_data) - 1:
                    model.fitted_model.append(
                        [test_data.iloc[i]], refit=False
                    )
            RETURN np.array(predictions)
   
   2.5. walk_forward_predict_grm(baseline, grm, test_data):
        # (AblationStudy'deki ile aynı)
        predictions = []
        all_residuals = list(baseline.get_residuals())
        
        FOR i in range(len(test_data)):
            baseline_pred = baseline.predict(1)[0]
            # ... (GRM düzeltmesi)
            # ... (güncelleme)
        
        RETURN np.array(predictions)
   
   2.6. compare_models(models_dict, data):
        all_results = {}
        
        FOR name, (model_class, kwargs) in models_dict.items():
            PRINT f"\n{name} değerlendiriliyor..."
            results = self.evaluate_model(model_class, data, **kwargs)
            all_results[name] = results
        
        # Özet istatistikler
        summary = []
        FOR name, results in all_results.items():
            summary.append({
                'Model': name,
                'Mean_RMSE': mean(results['rmse']),
                'Std_RMSE': std(results['rmse']),
                'Min_RMSE': min(results['rmse']),
                'Max_RMSE': max(results['rmse']),
                'Mean_MAE': mean(results['mae']),
                'Std_MAE': std(results['mae'])
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('Mean_RMSE')
        
        RETURN df, all_results
```

**Dosya:** `main_cross_validation.py`

**Algoritma:**
```
1. IMPORTS:
   from models.cross_validation import TimeSeriesCrossValidator
   from models.baseline_model import BaselineARIMA
   from models.grm_model import SchwarzschildGRM
   from models.kerr_grm_model import KerrGRM
   from models.garch_model import GARCHModel

2. VERI YUKLEME:
   # (main_ablation_study.py ile aynı)

3. CV OLUSTUR:
   cv = TimeSeriesCrossValidator(
       initial_train_size=300,
       val_size=50,
       test_size=50,
       step_size=50
   )

4. MODELLERI TANIMLA:
   models = {
       'Baseline': (BaselineARIMA, {}),
       'Schwarzschild': (SchwarzschildGRM, {
           'window_size': 20,
           'use_decay': True
       }),
       'Kerr': (KerrGRM, {
           'window_size': 20,
           'use_decay': True,
           'use_tanh': True
       }),
       'GARCH': (GARCHModel, {'p': 1, 'q': 1})
   }

5. KARSILASTIR:
   comparison_df, detailed_results = cv.compare_models(
       models, df['y'].values
   )

6. RAPOR:
   PRINT comparison_df.to_string()
   comparison_df.to_csv('results/cv_results.csv')
```

---

## 🔬 FAZE 5: PIML TEMEL ENTEGRASYONU

### **ADIM 5.1: Gravitational Residual Network (GRN) İmplementasyonu**

#### **5.1.1: Yeni Dosyalar Oluştur**

**Dosya:** `models/grn_network.py`

**Algoritma:**
```
1. IMPORTS:
   import torch
   import torch.nn as nn
   import numpy as np
   from typing import List, Optional

2. CLASS GravitationalResidualNetwork(nn.Module):
   
   2.1. __init__(input_size, hidden_sizes, output_size, use_monotonicity, use_energy_conservation):
        super().__init__()
        
        self.use_monotonicity = use_monotonicity
        self.use_energy_conservation = use_energy_conservation
        
        # Encoder network
        layers = []
        prev_size = input_size
        FOR hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Learnable physics parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.5))
   
   2.2. forward(mass, spin, tau, residuals_history):
        # Decay factor
        decay = 1.0 / (1.0 + self.beta * tau)
        
        # Input features
        x = cat([mass, spin, tau, residuals_history[:, -1:]], dim=1)
        
        # Neural network correction
        nn_correction = self.network(x)
        
        # Physics-inspired base term
        base_term = self.alpha * mass * tanh(residuals_history[:, -1:])
        spin_term = self.gamma * spin * residuals_history[:, -1:]
        
        # Combined output
        curvature = (base_term + spin_term + nn_correction) * decay
        
        RETURN curvature
   
   2.3. physics_loss(mass, curvature):
        loss = 0.0
        
        IF self.use_monotonicity:
            # Monotonicity constraint
            mass_perturbed = mass + 0.01
            curvature_perturbed = self.forward(...)
            derivative = (curvature_perturbed - curvature) / 0.01
            monotonicity_loss = relu(-derivative).mean()
            loss += 0.1 * monotonicity_loss
        
        IF self.use_energy_conservation:
            total_energy = sum(abs(curvature))
            energy_loss = relu(total_energy - 10.0)
            loss += 0.01 * energy_loss
        
        RETURN loss
   
   2.4. combined_loss(predictions, targets, mass, curvature):
        data_loss = MSELoss()(predictions, targets)
        physics_loss = self.physics_loss(mass, curvature)
        total_loss = data_loss + 0.1 * physics_loss
        RETURN total_loss, data_loss, physics_loss
```

**Dosya:** `models/grn_trainer.py`

**Algoritma:**
```
1. IMPORTS:
   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader, Dataset
   from models.grn_network import GravitationalResidualNetwork

2. CLASS GRMDataSet(Dataset):
   
   2.1. __init__(mass, spin, tau, residuals_history, targets):
        self.mass = torch.FloatTensor(mass)
        self.spin = torch.FloatTensor(spin)
        self.tau = torch.FloatTensor(tau)
        self.residuals_history = torch.FloatTensor(residuals_history)
        self.targets = torch.FloatTensor(targets)
   
   2.2. __len__():
        RETURN len(self.mass)
   
   2.3. __getitem__(idx):
        RETURN (
            self.mass[idx],
            self.spin[idx],
            self.tau[idx],
            self.residuals_history[idx],
            self.targets[idx]
        )

3. CLASS GRNTrainer:
   
   3.1. __init__(model, learning_rate):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )
   
   3.2. train_epoch(train_loader):
        self.model.train()
        total_loss = 0.0
        
        FOR batch in train_loader:
            mass, spin, tau, residuals_history, targets = batch
            
            # Forward pass
            predictions = self.model(
                mass, spin, tau, residuals_history
            )
            
            # Loss
            loss, data_loss, physics_loss = self.model.combined_loss(
                predictions, targets, mass, predictions
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        RETURN total_loss / len(train_loader)
   
   3.3. evaluate(val_loader):
        self.model.eval()
        total_loss = 0.0
        
        WITH torch.no_grad():
            FOR batch in val_loader:
                # ... (train_epoch ile aynı, ama backward yok)
        
        RETURN total_loss / len(val_loader)
   
   3.4. fit(train_loader, val_loader, epochs, early_stopping):
        best_val_loss = float('inf')
        patience_counter = 0
        
        FOR epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            PRINT f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}"
            
            IF val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/grn_best.pth')
            ELSE:
                patience_counter += 1
                IF patience_counter >= early_stopping:
                    PRINT f"Early stopping at epoch {epoch+1}"
                    BREAK
        
        # Load best model
        self.model.load_state_dict(torch.load('models/grn_best.pth'))
```

**Dosya:** `models/grn_data_preparator.py`

**Algoritma:**
```
1. CLASS GRNDataPreparator:
   
   1.1. prepare_features(residuals, window_size):
        n = len(residuals)
        mass_list = []
        spin_list = []
        tau_list = []
        residuals_history_list = []
        targets_list = []
        
        FOR t in range(window_size, n - 1):
            window = residuals[t - window_size:t]
            
            # Features
            mass = var(window)
            spin = corrcoef(window[1:], window[:-1])[0, 1]
            tau = compute_tau(residuals[:t])
            residuals_history = window
            
            mass_list.append(mass)
            spin_list.append(spin)
            tau_list.append(tau)
            residuals_history_list.append(residuals_history)
            targets_list.append(residuals[t + 1])
        
        RETURN (
            np.array(mass_list),
            np.array(spin_list),
            np.array(tau_list),
            np.array(residuals_history_list),
            np.array(targets_list)
        )
   
   1.2. compute_tau(residuals, threshold):
        abs_res = abs(residuals)
        shock_indices = where(abs_res > threshold)[0]
        
        IF len(shock_indices) == 0:
            RETURN len(residuals)
        
        last_shock = shock_indices[-1]
        tau = len(residuals) - last_shock
        RETURN tau
```

**Dosya:** `main_grn_train.py`

**Algoritma:**
```
1. IMPORTS:
   from models.grn_network import GravitationalResidualNetwork
   from models.grn_trainer import GRNTrainer, GRMDataSet
   from models.grn_data_preparator import GRNDataPreparator
   from torch.utils.data import DataLoader

2. VERI HAZIRLAMA:
   # Baseline residuals al
   baseline = BaselineARIMA()
   baseline.fit(train_df['y'])
   train_residuals = baseline.get_residuals()
   val_residuals = baseline.get_residuals_val()  # Val için de
   
   # GRN features hazırla
   preparator = GRNDataPreparator()
   train_features = preparator.prepare_features(
       train_residuals, window_size=20
   )
   val_features = preparator.prepare_features(
       val_residuals, window_size=20
   )
   
   # Dataset oluştur
   train_dataset = GRMDataSet(*train_features)
   val_dataset = GRMDataSet(*val_features)
   
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

3. MODEL OLUSTUR:
   model = GravitationalResidualNetwork(
       input_size=4,
       hidden_sizes=[64, 32, 16],
       output_size=1,
       use_monotonicity=True,
       use_energy_conservation=True
   )

4. TRAINER OLUSTUR:
   trainer = GRNTrainer(model, learning_rate=0.001)

5. EGITIM:
   trainer.fit(
       train_loader,
       val_loader,
       epochs=100,
       early_stopping=10
   )

6. TEST:
   # Test verisi üzerinde tahmin yap
   # Manuel fonksiyon vs GRN karşılaştır
```

---

### **ADIM 5.2: Symbolic Regression İmplementasyonu**

**Dosya:** `models/symbolic_discovery.py`

**Algoritma:**
```
1. IMPORTS:
   from pysr import PySRRegressor
   import numpy as np

2. CLASS SymbolicGRM:
   
   2.1. __init__():
        self.model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "tanh", "abs"],
            model_selection="best",
            loss="loss(x, y) = (x - y)^2",
            maxsize=20,
            populations=15
        )
   
   2.2. prepare_features(residuals, window_size):
        # (GRNDataPreparator ile aynı mantık)
        X = []
        y = []
        
        FOR t in range(window_size, n - 1):
            window = residuals[t - window_size:t]
            mass = var(window)
            spin = corrcoef(window[1:], window[:-1])[0, 1]
            tau = compute_tau(residuals[:t])
            epsilon = residuals[t]
            
            X.append([mass, spin, tau, epsilon])
            y.append(residuals[t + 1])
        
        RETURN np.array(X), np.array(y)
   
   2.3. discover_formula(residuals, window_size):
        X, y = self.prepare_features(residuals, window_size)
        feature_names = ["M", "a", "tau", "epsilon"]
        
        PRINT "Sembolik regresyon başlatılıyor..."
        self.model.fit(X, y, variable_names=feature_names)
        
        best_formula = self.model.get_best()
        
        PRINT f"Γ(t) = {best_formula}"
        PRINT f"R² Score: {self.model.score(X, y):.4f}"
        
        RETURN best_formula
   
   2.4. predict(M, a, tau, epsilon):
        X = column_stack([M, a, tau, epsilon])
        RETURN self.model.predict(X)
```

**Dosya:** `main_symbolic_discovery.py`

**Algoritma:**
```
1. IMPORTS:
   from models.symbolic_discovery import SymbolicGRM
   from models.baseline_model import BaselineARIMA

2. VERI HAZIRLAMA:
   baseline = BaselineARIMA()
   baseline.fit(train_df['y'])
   residuals = baseline.get_residuals()

3. SYMBOLIC DISCOVERY:
   symbolic_grm = SymbolicGRM()
   formula = symbolic_grm.discover_formula(residuals, window_size=20)

4. KARSILASTIRMA:
   # Manuel formül vs keşfedilen formül
   # Test verisi üzerinde performans karşılaştır
```

---

## 🌌 FAZE 6: PIML İLERİ SEVİYE

### **ADIM 6.1: Unified End-to-End Model**

**Dosya:** `models/unified_grm.py`

**Algoritma:**
```
1. CLASS UnifiedGRM(nn.Module):
   
   1.1. __init__(input_size, lstm_hidden_size, grn_hidden_sizes):
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
            input_size=4,
            hidden_sizes=grn_hidden_sizes,
            output_size=1
        )
   
   1.2. forward(x_history):
        # Baseline LSTM prediction
        lstm_out, _ = self.lstm(x_history)
        baseline_pred = self.lstm_output(lstm_out[:, -1, :])
        
        # Compute residuals
        residuals = x_history[:, :, 0] - baseline_pred.detach()
        
        # Compute GRM features
        mass = var(residuals, dim=1, keepdim=True)
        spin = self.compute_autocorr(residuals)
        tau = ones_like(mass) * 5.0  # Simplified
        
        # GRM correction
        grm_correction = self.grn(mass, spin, tau, residuals)
        
        # Final prediction
        final_pred = baseline_pred + grm_correction
        
        RETURN baseline_pred, grm_correction, final_pred
   
   1.3. combined_loss(baseline_pred, grm_correction, final_pred, targets):
        loss_final = MSELoss()(final_pred, targets)
        loss_baseline = MSELoss()(baseline_pred, targets)
        loss_physics = self.grn.physics_loss(mass, grm_correction)
        
        total_loss = loss_final + 0.1 * loss_baseline + 0.05 * loss_physics
        RETURN total_loss, loss_final, loss_baseline, loss_physics
```

---

### **ADIM 6.2: Multi-Body GRM**

**Dosya:** `models/multi_body_grm.py`

**Algoritma:**
```
1. IMPORTS:
   from sklearn.cluster import DBSCAN
   from models.grm_model import SchwarzschildGRM

2. CLASS MultiBodyGRM:
   
   2.1. __init__(n_bodies, window_size):
        self.n_bodies = n_bodies
        self.window_size = window_size
        self.body_params = []
   
   2.2. cluster_residuals(residuals):
        features = []
        FOR t in range(self.window_size, len(residuals)):
            window = residuals[t - self.window_size:t]
            features.append([
                mean(window),
                std(window),
                max(window),
                min(window),
                autocorr(window)
            ])
        
        features = np.array(features)
        
        # DBSCAN clustering
        clusterer = DBSCAN(eps=0.5, min_samples=10)
        regime_labels = clusterer.fit_predict(features)
        
        RETURN regime_labels
   
   2.3. fit(residuals):
        regime_labels = self.cluster_residuals(residuals)
        
        FOR regime_id in unique(regime_labels):
            IF regime_id == -1:
                CONTINUE
            
            regime_mask = regime_labels == regime_id
            regime_residuals = residuals[self.window_size:][regime_mask]
            
            # Her rejim için ayrı GRM fit
            grm = SchwarzschildGRM(window_size=self.window_size)
            grm.fit(regime_residuals)
            
            self.body_params.append({
                'body_id': regime_id,
                'alpha': grm.alpha,
                'beta': grm.beta,
                'n_samples': len(regime_residuals)
            })
   
   2.4. compute_curvature(residuals, current_regime):
        total_curvature = 0.0
        
        FOR params in self.body_params:
            body_id = params['body_id']
            alpha = params['alpha']
            
            # Weight
            IF body_id == current_regime:
                weight = 1.0
            ELSE:
                weight = 0.1
            
            # Bu body'nin katkısı
            mass = var(residuals[-self.window_size:])
            gamma_i = alpha * mass * sign(residuals[-1])
            
            total_curvature += weight * gamma_i
        
        RETURN total_curvature
```

---

## 📁 GENEL DOSYA YAPISI

```
Proje/
├── models/
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── grm_model.py                    ← GÜNCELLENECEK (decay factor)
│   ├── kerr_grm_model.py               ← GÜNCELLENECEK (decay factor)
│   ├── ablation_study.py               ← YENİ
│   ├── cross_validation.py              ← YENİ
│   ├── grn_network.py                  ← YENİ (PIML)
│   ├── grn_trainer.py                  ← YENİ (PIML)
│   ├── grn_data_preparator.py           ← YENİ (PIML)
│   ├── symbolic_discovery.py            ← YENİ (PIML)
│   ├── unified_grm.py                   ← YENİ (PIML)
│   └── multi_body_grm.py                ← YENİ (PIML)
│
├── main_phase3.py                      ← GÜNCELLENECEK (decay factor)
├── main_ablation_study.py               ← YENİ
├── main_cross_validation.py             ← YENİ
├── main_grn_train.py                    ← YENİ (PIML)
├── main_symbolic_discovery.py           ← YENİ (PIML)
│
├── config_phase3.py                     ← GÜNCELLENECEK (decay params)
├── config_phase4.py                     ← YENİ (FAZE 4 config)
├── config_phase5.py                     ← YENİ (FAZE 5 config)
│
├── results/
│   ├── ablation_results.csv
│   ├── cv_results.csv
│   ├── grn_training_log.txt
│   └── symbolic_formula.txt
│
└── requirements.txt                     ← GÜNCELLENECEK (torch, pysr)
```

---

## 🧪 TEST VE DOĞRULAMA STRATEJİSİ

### **Test Sırası:**

```
1. FAZE 4.1: Decay Factor Test
   - Mevcut main_phase3.py çalıştır
   - Decay factor ile/olmadan karşılaştır
   - RMSE iyileşmesini ölç
   - Beklenen: %2-3 iyileşme

2. FAZE 4.2: Ablasyon Test
   - main_ablation_study.py çalıştır
   - Tüm varyasyonları test et
   - En iyi kombinasyonu bul
   - Beklenen: Hangi bileşen kritik?

3. FAZE 4.3: Cross-Validation Test
   - main_cross_validation.py çalıştır
   - Tüm modelleri CV ile değerlendir
   - Sağlamlık testi
   - Beklenen: Model genellenebilirliği

4. FAZE 5.1: GRN Test
   - main_grn_train.py çalıştır
   - Manuel fonksiyon vs GRN karşılaştır
   - Beklenen: %5-10 iyileşme

5. FAZE 5.2: Symbolic Regression Test
   - main_symbolic_discovery.py çalıştır
   - Keşfedilen formülü test et
   - Beklenen: Yeni formül keşfi

6. FAZE 6: İleri Seviye Test
   - Unified model test
   - Multi-body test
   - Kapsamlı karşılaştırma
```

---

## 📊 BEKLENEN SONUÇLAR

### **FAZE 4 Sonrası:**
- RMSE: 0.101406 → 0.095 (%6 iyileşme)
- Decay factor etkisi: +%2-3
- Ablasyon: Kritik bileşenler belirlendi
- CV: Model sağlamlığı doğrulandı

### **FAZE 5 Sonrası:**
- RMSE: 0.095 → 0.085 (%10 iyileşme)
- GRN: Manuel fonksiyondan %5-8 daha iyi
- Symbolic: Yeni formül keşfedildi

### **FAZE 6 Sonrası:**
- RMSE: 0.085 → 0.075 (%15 toplam iyileşme)
- Unified: End-to-end öğrenme
- Multi-body: Rejim switching

---

## 🎯 UYGULAMA SIRASI

### **1. HAFTA (FAZE 4):**
1. ✅ Decay factor ekle (2 gün)
2. ✅ Ablasyon çalışması (3 gün)
3. ✅ Time-series CV (2 gün)

### **2-3. HAFTA (FAZE 5):**
1. ✅ GRN implementasyonu (1 hafta)
2. ✅ Symbolic regression (3 gün)
3. ✅ Karşılaştırma ve raporlama (2 gün)

### **4-8. HAFTA (FAZE 6):**
1. ✅ Unified model (2 hafta)
2. ✅ Multi-body GRM (1 hafta)
3. ✅ Kapsamlı test ve raporlama (1 hafta)

---

## ✅ DOĞRULAMA CHECKLIST

- [ ] Decay factor çalışıyor mu?
- [ ] Ablasyon sonuçları mantıklı mı?
- [ ] CV sonuçları tutarlı mı?
- [ ] GRN eğitimi başarılı mı?
- [ ] Symbolic regression formül buldu mu?
- [ ] Unified model çalışıyor mu?
- [ ] Multi-body rejimleri tespit ediyor mu?
- [ ] Tüm testler geçiyor mu?
- [ ] Performans iyileşmesi beklenen seviyede mi?

---

**SONUÇ:** Bu plan, tüm geliştirmeleri sistematik ve algoritmik olarak uygulamanızı sağlar. Her adım net tanımlanmış ve test edilebilir durumda. Hangi fazdan başlamak istersiniz? 🚀

