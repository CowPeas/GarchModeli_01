# 🎓 GRM PROJESİ - İLERİ SEVİYE GELİŞTİRME ROADMAP

## 📐 MATEMATİKSEL VE KAVRAMSAL ÇERÇEVE

Bu doküman, GRM projesinin yüksek seviye matematiksel ve kavramsal geliştirme planını içerir.

---

## 🎯 **FAZ 1: İSTATİSTİKSEL GÜÇ VE REJİM COVERAGE OPTİMİZASYONU**

### **Hedef:** Test setinde en az 3-5 farklı rejim yakalamak

### **Matematiksel Temeller**

**Ergodic coverage teoremi:**
```
π_k = lim_{T→∞} (1/T) Σ 𝟙{R_t = k}

Test validity: ∀k, π_k > π_min (örn. π_min = 0.05)
```

**Minimum test size:**
```
T_min = -log(1-γ) / λ₂ · K

γ = 0.95 (coverage confidence)
λ₂ = second eigenvalue of transition matrix
K = number of regimes
```

### **İmplementasyon Adımları**

#### **1.1 Power Analysis Tool**

```python
# Yeni dosya: models/power_analysis.py

class StatisticalPowerAnalyzer:
    """
    İstatistiksel güç analizi ve optimal sample size hesaplayıcı.
    """
    
    @staticmethod
    def compute_required_sample_size(
        delta: float,           # Hedef effect size
        sigma: float,           # Standard deviation
        alpha: float = 0.05,    # Type I error
        power: float = 0.80     # Desired power (1 - Type II error)
    ) -> int:
        """
        Diebold-Mariano test için gerekli sample size.
        
        Formula:
        n = ((z_α/2 + z_β) · σ / δ)²
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = ((z_alpha + z_beta) * sigma / delta) ** 2
        return int(np.ceil(n))
    
    @staticmethod
    def estimate_power(
        n: int,
        delta: float,
        sigma: float,
        alpha: float = 0.05
    ) -> float:
        """
        Verilen sample size için test gücünü hesapla.
        
        Power = Φ(δ/(σ/√n) - z_α/2)
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        power = norm.cdf(delta / (sigma / np.sqrt(n)) - z_alpha)
        return power
```

#### **1.2 Markov Chain Regime Analyzer**

```python
# models/regime_markov_analysis.py

class RegimeMarkovAnalyzer:
    """
    Rejim geçişlerini Markov zinciri olarak modeller.
    """
    
    @staticmethod
    def estimate_transition_matrix(regime_labels: np.ndarray) -> np.ndarray:
        """
        Transition matrix P[i,j] = P(R_t+1 = j | R_t = i)
        """
        K = len(np.unique(regime_labels))
        P = np.zeros((K, K))
        
        for t in range(len(regime_labels) - 1):
            i, j = regime_labels[t], regime_labels[t+1]
            P[i, j] += 1
        
        # Normalize
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / np.where(row_sums > 0, row_sums, 1)
        
        return P
    
    @staticmethod
    def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
        """
        Stationary distribution: π^T P = π^T
        """
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        # İlk eigenvector (eigenvalue = 1)
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()  # Normalize
        
        return pi
    
    @staticmethod
    def compute_mixing_time(P: np.ndarray, epsilon: float = 0.01) -> float:
        """
        Mixing time: Stationary distribution'a yakınsamak için gereken zaman.
        
        τ_mix = -1 / log|λ₂|
        """
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        
        lambda_2 = eigenvalues_sorted[1]  # Second largest eigenvalue
        
        if lambda_2 >= 1:
            return np.inf
        
        mixing_time = -np.log(epsilon) / np.log(lambda_2)
        return mixing_time
    
    @staticmethod
    def recommend_test_size(
        P: np.ndarray,
        coverage_confidence: float = 0.95
    ) -> int:
        """
        Tüm rejimleri yeterince örneklemek için minimum test size.
        
        T_min = -log(1-γ) / log(λ₂) · K
        """
        K = P.shape[0]
        mixing_time = RegimeMarkovAnalyzer.compute_mixing_time(P)
        
        T_min = -np.log(1 - coverage_confidence) * mixing_time * K
        
        return int(np.ceil(T_min))
```

#### **1.3 Config Güncelleme**

```python
# config_phase3.py - ENHANCED

# Data config - EXTENDED TIME SERIES
REAL_DATA_CONFIG = {
    'ticker': 'BTC-USD',
    'start_date': '2018-01-01',  # 5 yıl data (2023 → 2018)
    'end_date': '2025-11-09',
    'period': '5y',  # 2y → 5y
    'use_returns': True,
    'detect_volatility': True
}

# Split config - OPTIMIZED FOR REGIME COVERAGE
SPLIT_CONFIG = {
    'train_ratio': 0.50,  # 0.70 → 0.50 (daha fazla test için)
    'val_ratio': 0.15,
    'test_ratio': 0.35   # 0.15 → 0.35 (CRITICAL!)
}

# Markov analysis config - NEW
MARKOV_ANALYSIS_CONFIG = {
    'enable': True,
    'coverage_confidence': 0.95,
    'min_regime_samples': 20,  # Her rejimde en az 20 gözlem
    'auto_adjust_split': True  # T_min'e göre otomatik ayarlama
}
```

---

## 🧮 **FAZ 2: DBSCAN PARAMETRE OPTİMİZASYONU**

### **Matematiksel Optimizasyon Problemi**

```
(ε*, minPts*) = arg max_{ε,m} Silhouette(C_{ε,m})

s.t. 
  K_min ≤ |C_{ε,m}| ≤ K_max
  outlier_ratio < θ_max
```

### **İmplementasyon**

#### **2.1 K-Distance Graph Analizi**

```python
# models/dbscan_optimizer.py

class DBSCANOptimizer:
    """
    DBSCAN parametrelerini otomatik optimize eder.
    """
    
    @staticmethod
    def compute_k_distances(X: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Her nokta için k-nearest neighbor mesafesini hesapla.
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # k-distance (ilk mesafe 0 olduğu için k+1'inci)
        k_distances = distances[:, k]
        
        return np.sort(k_distances)[::-1]  # Descending order
    
    @staticmethod
    def find_elbow_point(k_distances: np.ndarray) -> float:
        """
        K-distance grafiğindeki "elbow point"i bul (optimal ε).
        
        Method: Maximum curvature (2nd derivative)
        """
        # 2. türev (discrete approximation)
        second_derivative = np.diff(k_distances, 2)
        
        # Maksimum curvature noktası
        elbow_idx = np.argmax(np.abs(second_derivative)) + 1
        
        epsilon = k_distances[elbow_idx]
        
        return epsilon
    
    @staticmethod
    def optimize_eps_minpts_grid(
        X: np.ndarray,
        eps_range: np.ndarray = None,
        minpts_range: np.ndarray = None,
        K_desired: int = None
    ) -> Tuple[float, int, Dict]:
        """
        Grid search ile optimal (ε, minPts) bulunması.
        
        Objective: Maximum silhouette score
        Constraints: K_min ≤ K ≤ K_max, outlier_ratio < 0.2
        """
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        if eps_range is None:
            k_dists = DBSCANOptimizer.compute_k_distances(X, k=5)
            eps_baseline = DBSCANOptimizer.find_elbow_point(k_dists)
            eps_range = np.linspace(eps_baseline * 0.5, eps_baseline * 1.5, 10)
        
        if minpts_range is None:
            D = X.shape[1]
            minpts_range = np.arange(max(D+1, 3), max(D+1, 10))
        
        best_score = -1
        best_params = None
        results = []
        
        for eps in eps_range:
            for minpts in minpts_range:
                dbscan = DBSCAN(eps=eps, min_samples=int(minpts))
                labels = dbscan.fit_predict(X)
                
                # Constraints
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                outlier_ratio = np.sum(labels == -1) / len(labels)
                
                if n_clusters < 2:
                    continue
                
                if outlier_ratio > 0.3:
                    continue
                
                if K_desired is not None:
                    if abs(n_clusters - K_desired) > 3:
                        continue
                
                # Silhouette score
                try:
                    score = silhouette_score(X, labels)
                except:
                    score = -1
                
                results.append({
                    'eps': eps,
                    'minpts': minpts,
                    'n_clusters': n_clusters,
                    'outlier_ratio': outlier_ratio,
                    'silhouette': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, int(minpts))
        
        return best_params[0], best_params[1], {
            'best_score': best_score,
            'all_results': pd.DataFrame(results)
        }
    
    @staticmethod
    def adaptive_eps_from_hopkins(X: np.ndarray) -> float:
        """
        Hopkins statistic ile clustering tendency'yi ölç.
        
        H ≈ 1 → data clusterable
        H ≈ 0.5 → random
        """
        from scipy.spatial.distance import cdist
        
        n = len(X)
        m = min(int(0.1 * n), 100)  # Sample size
        
        # Random sample from data
        indices = np.random.choice(n, m, replace=False)
        X_sample = X[indices]
        
        # Random uniform sample
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_random = np.random.uniform(X_min, X_max, (m, X.shape[1]))
        
        # Distance to nearest neighbor
        u_dists = cdist(X_random, X).min(axis=1)
        w_dists = cdist(X_sample, X).min(axis=1)
        
        H = u_dists.sum() / (u_dists.sum() + w_dists.sum())
        
        return H
```

#### **2.2 Feature Engineering Enhancement**

```python
# models/grm_feature_engineering.py

class GRMFeatureEngineer:
    """
    GRM için optimal feature extraction.
    """
    
    @staticmethod
    def extract_regime_features(
        residuals: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Multi-dimensional feature extraction için optimize edilmiş.
        
        Features:
        1. Mass (volatility)
        2. Spin (autocorrelation)
        3. Time since shock
        4. Higher moments (skewness, kurtosis)
        5. Local trend
        6. Entropy
        """
        n = len(residuals)
        features = []
        
        for t in range(window, n):
            window_data = residuals[t-window:t]
            
            # 1. Mass (variance)
            mass = np.var(window_data)
            
            # 2. Spin (ACF lag-1)
            spin = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
            if np.isnan(spin):
                spin = 0
            
            # 3. Time since shock (quantile-based)
            shock_threshold = np.percentile(np.abs(residuals[:t]), 95)
            shock_times = np.where(np.abs(residuals[:t]) > shock_threshold)[0]
            tau = t - shock_times[-1] if len(shock_times) > 0 else window
            
            # 4. Kurtosis (tail behavior)
            from scipy.stats import kurtosis, skew
            kurt = kurtosis(window_data)
            skewness = skew(window_data)
            
            # 5. Local trend (linear regression slope)
            x = np.arange(window)
            slope = np.polyfit(x, window_data, 1)[0]
            
            # 6. Entropy (discretized)
            hist, _ = np.histogram(window_data, bins=10, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            
            features.append([
                mass, spin, tau, kurt, skewness, slope, entropy
            ])
        
        return np.array(features)
    
    @staticmethod
    def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Z-score standardization with outlier clipping.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        X_std = (X - mean) / (std + 1e-8)
        
        # Clip outliers (±5σ)
        X_std = np.clip(X_std, -5, 5)
        
        scaler_params = {'mean': mean, 'std': std}
        
        return X_std, scaler_params
```

---

## 🌍 **FAZ 3: MULTI-ASSET FRAMEWORK**

### **Hierarchical Bayesian Model**

```
Level 1 (Global): θ_global ~ N(μ₀, Σ₀)
Level 2 (Asset):  θ_asset ~ N(θ_global, Σ_asset)
Level 3 (Time):   y_t ~ f(x_t; θ_asset)
```

### **İmplementasyon**

```python
# models/multi_asset_grm.py

class MultiAssetGRM:
    """
    Birden fazla varlık üzerinde GRM modeli.
    
    Hierarchical structure:
    - Global parameters (shared across assets)
    - Asset-specific parameters
    - Transfer learning between assets
    """
    
    def __init__(self, assets: List[str]):
        self.assets = assets
        self.asset_models = {}
        self.global_params = None
    
    def fit_hierarchical(
        self,
        data_dict: Dict[str, pd.Series],
        share_ratio: float = 0.5
    ):
        """
        Hierarchical Bayesian estimation.
        
        share_ratio: Global parameters'ın ağırlığı
        """
        # Stage 1: Her asset için ayrı fit
        asset_params = {}
        for asset, data in data_dict.items():
            model = MultiBodyGRM()
            model.fit(data)
            asset_params[asset] = model.get_parameters()
        
        # Stage 2: Global parameters (empirical Bayes)
        self.global_params = self._aggregate_parameters(asset_params)
        
        # Stage 3: Shrinkage toward global
        for asset in self.assets:
            local = asset_params[asset]
            global_p = self.global_params
            
            # James-Stein estimator
            shrunk = (1 - share_ratio) * local + share_ratio * global_p
            
            self.asset_models[asset].set_parameters(shrunk)
    
    def _aggregate_parameters(
        self,
        asset_params: Dict
    ) -> Dict:
        """
        Weighted average of asset parameters.
        """
        # Simple average (can be weighted by sample size)
        global_alpha = np.mean([p['alpha'] for p in asset_params.values()])
        global_beta = np.mean([p['beta'] for p in asset_params.values()])
        
        return {'alpha': global_alpha, 'beta': global_beta}
    
    def predict_ensemble(
        self,
        asset: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Ensemble prediction using similar assets.
        """
        # Bu asset'in modeli
        main_pred = self.asset_models[asset].predict(X)
        
        # Correlation'a göre diğer asset modellerinin weighted prediction'ı
        correlations = self._compute_asset_correlations()
        
        ensemble_pred = main_pred
        total_weight = 1.0
        
        for other_asset in self.assets:
            if other_asset != asset:
                corr = correlations[asset][other_asset]
                weight = max(0, corr)  # Sadece pozitif correlation
                
                other_pred = self.asset_models[other_asset].predict(X)
                ensemble_pred += weight * other_pred
                total_weight += weight
        
        return ensemble_pred / total_weight
```

#### **3.1 Asset Selection Framework**

```python
# models/asset_selection.py

class AssetSelector:
    """
    Optimal asset portfolio for robust testing.
    """
    
    @staticmethod
    def select_diverse_assets(
        available_assets: List[str],
        n_select: int = 5,
        correlation_threshold: float = 0.7
    ) -> List[str]:
        """
        Maximum diversity portfolio.
        
        Objective: min Σ|ρᵢⱼ| s.t. |portfolio| = n_select
        """
        import yfinance as yf
        
        # Download data
        data = yf.download(available_assets, period='1y')['Adj Close']
        returns = data.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns.corr().abs()
        
        # Greedy selection
        selected = []
        remaining = available_assets.copy()
        
        # Start with most volatile asset
        volatilities = returns.std()
        selected.append(volatilities.idxmax())
        remaining.remove(selected[0])
        
        while len(selected) < n_select and len(remaining) > 0:
            best_asset = None
            min_avg_corr = 1.0
            
            for candidate in remaining:
                avg_corr = corr_matrix.loc[selected, candidate].mean()
                
                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    best_asset = candidate
            
            if min_avg_corr < correlation_threshold:
                selected.append(best_asset)
                remaining.remove(best_asset)
            else:
                break
        
        return selected
    
    @staticmethod
    def recommended_portfolio() -> Dict[str, Dict]:
        """
        Önceden tanımlanmış optimal portföy.
        """
        return {
            'BTC-USD': {
                'type': 'cryptocurrency',
                'volatility': 'very_high',
                'regime_dynamics': 'fast',
                'weight': 0.25
            },
            'ETH-USD': {
                'type': 'cryptocurrency',
                'volatility': 'high',
                'regime_dynamics': 'fast',
                'weight': 0.20
            },
            '^GSPC': {
                'type': 'equity_index',
                'volatility': 'medium',
                'regime_dynamics': 'slow',
                'weight': 0.25
            },
            '^VIX': {
                'type': 'volatility_index',
                'volatility': 'very_high',
                'regime_dynamics': 'counter_cyclical',
                'weight': 0.15
            },
            'GC=F': {
                'type': 'commodity',
                'volatility': 'low',
                'regime_dynamics': 'safe_haven',
                'weight': 0.15
            }
        }
```

---

## 📈 **FAZ 4: ADAPTIF WINDOWING VE NON-STATIONARITY**

### **Matematiksel Çerçeve**

**Exponential forgetting:**
```
θ_t = λ θ_{t-1} + (1-λ) ∇_θ L(θ; x_t, y_t)

λ = exp(-1/w) → half-life = w · log(2)
```

**Change point detection (CUSUM):**
```
S_t = max(0, S_{t-1} + (y_t - μ₀) - k)

Alarm: S_t > h
```

### **İmplementasyon**

```python
# models/adaptive_windowing.py

class AdaptiveWindowGRM:
    """
    Non-stationary time series için adaptive windowing.
    """
    
    def __init__(self, base_window: int = 252):
        self.base_window = base_window
        self.lambda_forgetting = np.exp(-1/base_window)
        self.change_points = []
    
    def detect_change_points(
        self,
        residuals: np.ndarray,
        k: float = 0.5,
        h: float = 5.0
    ) -> List[int]:
        """
        CUSUM test ile structural break detection.
        
        Parameters:
        k: allowance (typically σ/2)
        h: threshold (typically 4-5σ)
        """
        n = len(residuals)
        S = np.zeros(n)
        change_points = []
        
        mu_0 = np.mean(residuals[:min(50, n)])
        
        for t in range(1, n):
            S[t] = max(0, S[t-1] + (residuals[t] - mu_0) - k)
            
            if S[t] > h:
                change_points.append(t)
                S[t] = 0  # Reset
                mu_0 = np.mean(residuals[max(0, t-50):t])  # Update baseline
        
        self.change_points = change_points
        return change_points
    
    def fit_with_adaptive_window(
        self,
        data: pd.Series,
        detect_breaks: bool = True
    ):
        """
        Structural break'lere göre adaptive window kullan.
        """
        if detect_breaks:
            residuals = self._get_residuals(data)
            change_points = self.detect_change_points(residuals)
            
            if len(change_points) > 0:
                print(f"[DETECTED] {len(change_points)} structural breaks")
                
                # Her segment için ayrı model
                segments = self._split_by_change_points(data, change_points)
                
                for i, segment in enumerate(segments):
                    print(f"  Segment {i+1}: {len(segment)} observations")
                    # Fit segment-specific model
```

---

## 🔬 **FAZ 5: ROBUST ESTIMATION VE OUTLIER HANDLING**

### **M-Estimators Implementation**

```python
# models/robust_estimation.py

class RobustGRM:
    """
    Outlier'lara robust GRM estimation.
    """
    
    @staticmethod
    def huber_loss(u: np.ndarray, delta: float = 1.35) -> np.ndarray:
        """
        Huber loss function.
        
        ρ(u) = u²/2         if |u| ≤ δ
               δ|u| - δ²/2  if |u| > δ
        """
        return np.where(
            np.abs(u) <= delta,
            0.5 * u**2,
            delta * np.abs(u) - 0.5 * delta**2
        )
    
    @staticmethod
    def iteratively_reweighted_least_squares(
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        IRLS algorithm for robust regression.
        """
        n, p = X.shape
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        for iteration in range(max_iter):
            theta_old = theta.copy()
            
            # Residuals
            residuals = y - X @ theta
            
            # MAD scale estimate
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = 1.4826 * mad
            
            # Weights (Huber)
            u = residuals / (scale + 1e-10)
            weights = np.where(
                np.abs(u) <= 1.35,
                1.0,
                1.35 / np.abs(u)
            )
            
            # Weighted least squares
            W = np.diag(weights)
            theta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            
            # Convergence check
            if np.linalg.norm(theta - theta_old) < tol:
                break
        
        return theta
```

---

## 📊 **EXECUTION SUMMARY**

### **Timeline ve Öncelikler**

| Faz | Süre | Öncelik | Beklenen Etki |
|-----|------|---------|---------------|
| **Faz 1: Test periyodu opt.** | 1 hafta | 🔴 CRITICAL | +rejim coverage |
| **Faz 2: DBSCAN opt.** | 3 gün | 🔴 CRITICAL | +clustering quality |
| **Faz 3: Multi-asset** | 1 hafta | 🟡 HIGH | +generalization |
| **Faz 4: Adaptive window** | 3 gün | 🟢 MEDIUM | +robustness |
| **Faz 5: Robust est.** | 2 gün | 🟢 MEDIUM | +outlier handling |

### **Beklenen İyileştirmeler**

**Quantitative targets:**

```
Current:
- Test size: 110 obs → Target: 250-400 obs
- Regime coverage: 1 regime → Target: 3-5 regimes
- DM p-value: 0.507 → Target: < 0.05
- Statistical power: ~0.05 → Target: > 0.80
- RMSE improvement: 0.21% → Target: > 2-5%

Multi-asset:
- Single asset → 5 diverse assets
- Correlation structure: analyzed
- Generalization bound: reduced
```

---

## 🎯 **BAŞARILI PROJE KRİTERLERİ**

### **Matematiksel Başarı Kriterleri**

1. **İstatistiksel Anlamlılık:**
   ```
   DM test: p < 0.05
   Bootstrap CI: 0 ∉ [CI_lower, CI_upper]
   Statistical power: > 0.80
   ```

2. **Rejim Coverage:**
   ```
   Test setinde K ≥ 3 rejim
   Her rejim: n_k ≥ 20 observation
   Ergodic coverage: π_k > 0.05 ∀k
   ```

3. **Performans İyileştirmesi:**
   ```
   RMSE improvement: > 2%
   R² > 0 (baseline'dan better)
   MDA > 55% (random'dan better)
   ```

4. **Residual Quality:**
   ```
   ARCH-LM: p > 0.05 (homoskedastic)
   Ljung-Box: p > 0.05 (white noise)
   Normality: Jarque-Bera test
   ```

5. **Generalization:**
   ```
   Multi-asset consistency
   Cross-validation: CV_score variance < 10%
   Out-of-sample: R²_oos > 0
   ```

---

## 📚 **TEORİK REFERANSLAR**

### **İstatistiksel Testler**
- Diebold & Mariano (1995): "Comparing Predictive Accuracy"
- Harvey et al. (1997): "Testing the equality of prediction mean squared errors"
- Giacomini & White (2006): "Tests of Conditional Predictive Ability"

### **Regime-Switching Models**
- Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Ang & Bekaert (2002): "Regime Switches in Interest Rates"
- Guidolin & Timmermann (2007): "Asset Allocation under Multivariate Regime Switching"

### **Clustering & DBSCAN**
- Ester et al. (1996): "A Density-Based Algorithm for Discovering Clusters"
- Schubert et al. (2017): "DBSCAN Revisited: Why and How You Should (Still) Use DBSCAN"

### **Time Series**
- Box & Jenkins (1970): "Time Series Analysis: Forecasting and Control"
- Tsay (2005): "Analysis of Financial Time Series"
- Hamilton (1994): "Time Series Analysis"

### **Machine Learning Theory**
- Vapnik (1998): "Statistical Learning Theory"
- Hastie et al. (2009): "The Elements of Statistical Learning"
- Murphy (2012): "Machine Learning: A Probabilistic Perspective"

---

**Son Güncelleme:** 2025-11-24  
**Versiyon:** 1.0  
**Katkıda Bulunanlar:** GRM Research Team

