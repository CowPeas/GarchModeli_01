# 🚀 GRM PROJESİ - GELİŞMİŞ ÖZELLİKLER REHBERİ

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Yeni Modüller](#yeni-modüller)
3. [Kullanım Örnekleri](#kullanım-örnekleri)
4. [Konfigürasyon](#konfigürasyon)
5. [Sonuç Yorumlama](#sonuç-yorumlama)

---

## 🎯 Genel Bakış

Bu geliştirme paketi, GRM projesine aşağıdaki **kritik** iyileştirmeleri ekler:

### ✅ Eklenen Özellikler

| Özellik | Modül | Amaç |
|---------|-------|------|
| **İstatistiksel Testler** | `statistical_tests.py` | Diebold-Mariano, ARCH-LM, Ljung-Box testleri |
| **Bootstrap CI** | `advanced_metrics.py` | Performans farklarının güven aralıkları |
| **GARCH Benchmark** | `garch_model.py` | Volatilite modellemesi karşılaştırması |
| **Gelişmiş Metrikler** | `advanced_metrics.py` | MDA, R², Sharpe Ratio, Hit Ratio, vb. |
| **Rejim Analizi** | `regime_analysis.py` | Multi-Body GRM rejimlerinin karakterizasyonu |
| **Kapsamlı Karşılaştırma** | `comprehensive_comparison.py` | Tüm modelleri tek raporda karşılaştırma |
| **Gelişmiş CV** | `cross_validation.py` (enhanced) | Expanding, blocked, rolling window CV |

---

## 📦 Yeni Modüller

### 1. `models/statistical_tests.py`

**Amaç:** Model performans farklarının istatistiksel anlamlılığını test etme.

**Sınıflar ve Metodlar:**

```python
from models.statistical_tests import StatisticalTests

# Diebold-Mariano Test
dm_stat, dm_pval = StatisticalTests.diebold_mariano_test(
    errors1,  # Model 1 hataları
    errors2,  # Model 2 hataları
    alternative='two-sided'
)

# ARCH-LM Test (heteroskedasticity)
arch_lm, arch_pval = StatisticalTests.arch_lm_test(
    residuals,
    lags=5
)

# Ljung-Box Test (autocorrelation)
lb_stats, lb_pvals = StatisticalTests.ljung_box_test(
    residuals,
    lags=10
)
```

**Yorumlama:**
- **Diebold-Mariano:** p < 0.05 → Model 2, Model 1'den anlamlı şekilde farklı
- **ARCH-LM:** p < 0.05 → ARCH etkileri var (heteroskedasticity)
- **Ljung-Box:** p < 0.05 → Otokorelasyon var (beyaz gürültü değil)

---

### 2. `models/advanced_metrics.py`

**Amaç:** Çok boyutlu performans değerlendirmesi ve bootstrap analizi.

**Sınıflar:**

#### `AdvancedMetrics`

```python
from models.advanced_metrics import AdvancedMetrics

# Tüm metrikleri hesapla
metrics = AdvancedMetrics.calculate_all_metrics(
    y_true,
    y_pred,
    return_series=False  # True ise finansal metrikler de hesaplanır
)

# Sonuç: {'rmse': ..., 'mae': ..., 'mape': ..., 'r2': ..., 'mda': ..., 'hit_ratio': ...}
```

**Metrikler:**
- **RMSE, MAE, MAPE:** Temel hata metrikleri
- **R²:** Açıklanan varyans oranı
- **MDA (Mean Directional Accuracy):** Yön tahmin doğruluğu
- **Hit Ratio:** İsabet oranı
- **Sharpe Ratio, Max Drawdown, Win Rate:** Finansal metrikler (getiri serileri için)

#### `BootstrapCI`

```python
from models.advanced_metrics import BootstrapCI

boot = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)

# İki model arasındaki fark CI
ci_result = boot.performance_difference_ci(
    y_true,
    y_pred_model1,
    y_pred_model2,
    metric='rmse'
)

print(ci_result['interpretation'])
# Örnek: "Model 2, Model 1'den İSTATİSTİKSEL OLARAK ANLAMLI şekilde daha iyi (95% CI: [-0.0123, -0.0045])"
```

**Yorumlama:**
- CI sıfırı içermiyorsa → Anlamlı fark var
- `is_significant=True` → Model performansları istatistiksel olarak farklı

---

### 3. `models/garch_model.py`

**Amaç:** GARCH ailesi modellerini eğitmek ve benchmark olarak kullanmak.

**Kullanım:**

```python
from models.garch_model import GARCHModel

# Model oluştur
garch = GARCHModel(
    model_type='GARCH',  # 'GARCH', 'EGARCH', 'GJR-GARCH'
    p=1,  # ARCH order
    q=1,  # GARCH order
    mean_model='Constant'  # 'Constant', 'Zero', 'AR'
)

# Eğit
garch.fit(train_returns, verbose=True)

# Volatilite tahmini
volatility_forecast = garch.predict(horizon=10)

# Ortalama (getiri) tahmini
mean_forecast = garch.forecast_mean(horizon=10)
```

**Karşılaştırma:**

```python
from models.garch_model import compare_garch_models

comparison_df = compare_garch_models(
    train_data,
    val_data,
    model_types=['GARCH', 'EGARCH', 'GJR-GARCH']
)
```

---

### 4. `models/regime_analysis.py`

**Amaç:** Multi-Body GRM tarafından tespit edilen rejimleri analiz etme.

**Kullanım:**

```python
from models.regime_analysis import RegimeAnalyzer

analyzer = RegimeAnalyzer()
analyzer.fit(test_data, regime_labels)

# Rejim özeti
summary = analyzer.get_regime_summary()
print(summary)

# Rejim geçişleri
transitions = analyzer.get_regime_transitions()

# Dataset karakterizasyonu
char = analyzer.characterize_dataset()
print(f"Toplam Rejim: {char['n_regimes']}")
print(f"Outlier Oranı: {char['outlier_ratio']*100:.1f}%")

# Rapor oluştur
analyzer.generate_report(output_file='regime_report.txt')
```

**Rejim Özellikleri:**
- **Volatility Regime:** Low / Medium / High
- **Trend Type:** Stationary / Upward / Downward
- **Persistence:** High / Low (autokorelasyona göre)
- **Skewness, Kurtosis:** Dağılım özellikleri

---

### 5. `models/comprehensive_comparison.py`

**Amaç:** Tüm modelleri tek bir raporda kapsamlı karşılaştırma.

**Kullanım:**

```python
from models.comprehensive_comparison import ComprehensiveComparison

comp = ComprehensiveComparison(baseline_name='ARIMA')

# Model sonuçlarını ekle
comp.add_model_results('ARIMA', y_true, arima_preds)
comp.add_model_results('GARCH', y_true, garch_preds)
comp.add_model_results('Schwarzschild_GRM', y_true, schwarz_preds)
comp.add_model_results('Multi_Body_GRM', y_true, multi_body_preds)

# Kapsamlı rapor oluştur
report = comp.generate_comprehensive_report(
    output_file='comprehensive_report.txt'
)
print(report)
```

**Rapor İçeriği:**
1. **Temel Performans Metrikleri:** RMSE, MAE, MAPE, R², MDA, Hit Ratio
2. **Baseline'a Göre İyileştirmeler:** % iyileştirme
3. **İstatistiksel Anlamlılık Testleri:** DM, ARCH-LM, Ljung-Box
4. **Bootstrap Güven Aralıkları:** 95% CI ile performans farkları
5. **Genel Değerlendirme:** En iyi model, en fazla iyileştirme, anlamlı iyileştirme sayısı

**Hızlı Kullanım:**

```python
from models.comprehensive_comparison import quick_compare

report = quick_compare(
    y_true,
    {
        'ARIMA': arima_preds,
        'GRM': grm_preds,
        'Multi-Body': mb_preds
    },
    baseline_name='ARIMA',
    output_file='quick_report.txt'
)
```

---

## 🔧 Konfigürasyon

### `config_phase3.py` Güncellemeleri

```python
# İstatistiksel Testler
STATISTICAL_TEST_CONFIG = {
    'significance_level': 0.05,
    'bootstrap_n_iterations': 1000,
    'bootstrap_confidence_level': 0.95,
    'diebold_mariano_alternative': 'two-sided',
    'ljung_box_lags': 10,
    'arch_lm_lags': 5
}

# Cross-Validation
CV_CONFIG = {
    'method': 'expanding',  # 'expanding', 'walk-forward', 'blocked'
    'n_splits': 5,
    'test_size': 100,
    'gap': 0
}

# Gelişmiş Metrikler
METRICS_CONFIG = {
    'calculate_financial_metrics': False,
    'calculate_volatility_metrics': True,
    'calculate_directional_accuracy': True
}

# Rejim Analizi
REGIME_CONFIG = {
    'enable_regime_analysis': True,
    'dbscan_eps': 0.5,
    'dbscan_min_samples': 5,
    'auto_tune_dbscan': True
}

# GARCH
GARCH_CONFIG = {
    'model_type': 'GARCH',
    'p': 1,
    'q': 1,
    'mean_model': 'Constant'
}
```

---

## 🎓 Kullanım Örnekleri

### Örnek 1: Multi-Body GRM ile Kapsamlı Test

```bash
python main.py --multi-body
```

**Çıktılar:**
- `results/multi_body_grm_results.txt` - Temel sonuçlar
- `results/comprehensive_comparison_report.txt` - Kapsamlı karşılaştırma
- `results/regime_analysis_report.txt` - Rejim analizi

**Rapor İçeriği:**
- Performans metrikleri (RMSE, MAE, R², MDA, vb.)
- Diebold-Mariano test sonucu (p-değeri)
- Bootstrap CI (95% güven aralığı)
- ARCH-LM ve Ljung-Box test sonuçları
- Rejim özellikleri ve geçişler

### Örnek 2: Tüm Modelleri Karşılaştırma

```bash
python main_comprehensive_comparison.py
```

**Karşılaştırılan Modeller:**
1. ARIMA (Baseline)
2. GARCH
3. Schwarzschild GRM
4. Kerr GRM

**Çıktı:**
- `results/comprehensive_all_models_report.txt`

### Örnek 3: Manuel Rejim Analizi

```python
from models.regime_analysis import RegimeAnalyzer

# Rejim analizörü oluştur
analyzer = RegimeAnalyzer()
analyzer.fit(test_data, regime_labels)

# Özet tablo
summary_df = analyzer.get_regime_summary()
print(summary_df)

# Rejim geçişleri
transitions = analyzer.get_regime_transitions()
for trans, count in transitions.items():
    print(f"{trans}: {count} kez")

# Rapor
analyzer.generate_report('my_regime_report.txt')
```

### Örnek 4: Bootstrap ile İyileştirme Testi

```python
from models.advanced_metrics import BootstrapCI

boot = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)

# Manuel GRM vs Multi-Body GRM
ci_result = boot.performance_difference_ci(
    y_true,
    manual_grm_predictions,
    multi_body_grm_predictions,
    metric='rmse'
)

print(f"Ortalama Fark: {ci_result['mean_difference']:.6f}")
print(f"95% CI: [{ci_result['ci_lower']:.6f}, {ci_result['ci_upper']:.6f}]")
print(f"Anlamlı: {ci_result['is_significant']}")
print(f"\nYorum: {ci_result['interpretation']}")
```

---

## 📊 Sonuç Yorumlama Kılavuzu

### 1. RMSE İyileştirmesi

| İyileştirme | Değerlendirme |
|-------------|--------------|
| < 1% | Çok küçük, pratik önemi sınırlı |
| 1-5% | Küçük ama gözle görülür |
| 5-10% | Orta düzeyde iyileştirme |
| 10-20% | Önemli iyileştirme |
| > 20% | Çok büyük iyileştirme |

**ÖNEMLİ:** İyileştirme yüzdesi tek başına yeterli değil, **istatistiksel anlamlılık** şart!

### 2. İstatistiksel Anlamlılık

#### Diebold-Mariano Test

```
p-değeri < 0.05 → Modeller arasında ANLAMLI fark var
p-değeri ≥ 0.05 → Modeller arasında anlamlı fark YOK
```

**Örnek Yorumlar:**

✅ **p = 0.012 (< 0.05):**
> "Multi-Body GRM, Manuel GRM'den **istatistiksel olarak anlamlı** şekilde farklı performans göstermektedir (DM p=0.012). RMSE'deki %3.2 iyileştirme güvenilir."

⚠️ **p = 0.18 (≥ 0.05):**
> "Multi-Body GRM'in RMSE'de %0.21 iyileştirme göstermesine rağmen, bu fark **istatistiksel olarak anlamlı değil** (DM p=0.18). İyileştirme şans eseri olabilir."

#### Bootstrap CI

```
CI sıfırı içermiyorsa → Anlamlı fark var
CI sıfırı içeriyorsa → Anlamlı fark yok
```

**Örnek:**

```
RMSE Farkı: -0.0045
95% CI: [-0.0089, -0.0001]
```

✅ CI negatif ve sıfırı içermiyor → Multi-Body **kesinlikle** daha iyi

```
RMSE Farkı: -0.0012
95% CI: [-0.0035, +0.0011]
```

⚠️ CI sıfırı içeriyor → Fark **anlamsız**

### 3. ARCH-LM ve Ljung-Box Testleri

#### ARCH-LM (Heteroskedasticity)

```
p < 0.05 → ARCH etkileri VAR (heteroskedastic)
p ≥ 0.05 → ARCH etkileri YOK (homoskedastic)
```

✅ **p ≥ 0.05 (İDEAL):** Residual'lar homoskedastic, model volatiliteyi iyi yakalıyor.

⚠️ **p < 0.05:** ARCH etkileri var, model volatiliteyi tam yakalayamıyor. GARCH eklenmeli.

#### Ljung-Box (Autocorrelation)

```
p < 0.05 → Otokorelasyon VAR (beyaz gürültü değil)
p ≥ 0.05 → Otokorelasyon YOK (beyaz gürültü)
```

✅ **p ≥ 0.05 (İDEAL):** Residual'lar beyaz gürültü, model tüm yapıyı yakalamış.

⚠️ **p < 0.05:** Residual'larda hala yapı var, model yetersiz.

### 4. Rejim Analizi

#### Rejim Sayısı

| Durum | Yorum |
|-------|-------|
| n_regimes = 1 | Multi-Body gereksiz, veri tek rejimli |
| n_regimes = 2-5 | İDEAL, Multi-Body faydalı |
| n_regimes > 10 | DBSCAN aşırı hassas, parametreleri gevşet |

#### Outlier Oranı

| Durum | Yorum |
|-------|-------|
| < 10% | İyi, veri temiz |
| 10-30% | Kabul edilebilir |
| > 30% | Yüksek, veri ön işleme gerekebilir |

#### Rejim Geçişleri

```
n_transitions < 5 → Multi-Body'nin faydası sınırlı
n_transitions > 20 → Rejim geçişleri çok, Multi-Body etkili olabilir
```

### 5. Gelişmiş Metrikler

#### Mean Directional Accuracy (MDA)

```
MDA > 0.55 → İyi yön tahmini
MDA ≈ 0.50 → Rastgele tahmin kadar
MDA < 0.50 → Kötü, rastgeleden daha kötü
```

#### R² (Coefficient of Determination)

```
R² > 0.7 → Çok iyi fit
R² = 0.3-0.7 → Orta fit
R² < 0.3 → Zayıf fit
R² < 0 → Model ortalamadan daha kötü!
```

---

## 🏆 Başarı Kriterleri

Bir modelin **bilimsel olarak başarılı** sayılması için:

1. ✅ **RMSE İyileştirmesi:** En az %1-2% (bağlama göre)
2. ✅ **İstatistiksel Anlamlılık:** DM p-değeri < 0.05
3. ✅ **Bootstrap CI:** Sıfırı içermemeli
4. ✅ **ARCH-LM:** p ≥ 0.05 (residual'lar homoskedastic)
5. ✅ **Ljung-Box:** p ≥ 0.05 (residual'lar beyaz gürültü)
6. ✅ **Pratik Değer:** Model gerçek dünyada uygulanabilir olmalı

**UYARI:** Sadece RMSE iyileştirmesi yeterli DEĞİLDİR!

---

## 📝 Raporlama Şablonu

Sonuçları sunarken aşağıdaki şablonu kullanın:

```markdown
## Model Karşılaştırması: [Model A] vs [Model B]

### Performans Metrikleri
- Model A RMSE: X.XXXX
- Model B RMSE: Y.YYYY
- İyileştirme: ±Z.Z%

### İstatistiksel Anlamlılık
- **Diebold-Mariano Test:** p = 0.XXX
  - Yorum: [Anlamlı / Anlamsız]
  
- **Bootstrap 95% CI:** [lower, upper]
  - Yorum: [CI sıfırı içeriyor mu?]

### Residual Analizi
- **ARCH-LM Test:** p = 0.XXX
  - Yorum: [Heteroskedasticity var/yok]
  
- **Ljung-Box Test:** p = 0.XXX
  - Yorum: [Otokorelasyon var/yok]

### Sonuç
[Model B], [Model A]'dan [istatistiksel olarak anlamlı / anlamsız] 
şekilde [daha iyi / farklı değil]. İyileştirme [güvenilir / şans eseri olabilir].

**Öneri:** [Model B'yi kullan / Daha fazla veri gerekli / vb.]
```

---

## 🐛 Sorun Giderme

### Problem: Bootstrap CI çok geniş

**Neden:** Sample size küçük veya varyans yüksek  
**Çözüm:** 
- Test periyodunu uzat
- Bootstrap iterasyon sayısını artır (1000 → 2000)

### Problem: Tüm testler p > 0.05 (anlamlı fark yok)

**Neden:** Gerçekten fark yok veya sample size yetersiz  
**Çözüm:**
- Test setini uzat
- Farklı veri setlerinde test et
- Model farklılıklarını artır (daha agresif hiperparametre arama)

### Problem: ARCH-LM ve Ljung-Box hep anlamlı (p < 0.05)

**Neden:** Model residual'ları tam olarak yakalayamıyor  
**Çözüm:**
- GARCH ekleme düşünülmeli
- Model karmaşıklığını artır
- Veri ön işleme (outlier temizleme)

### Problem: Rejim sayısı çok az veya çok fazla

**Neden:** DBSCAN parametreleri (eps, min_samples) uygun değil  
**Çözüm:**
```python
from models.regime_analysis import recommend_dbscan_params

optimal_eps, optimal_min_samples = recommend_dbscan_params(
    data, feature_matrix
)
```

---

## 📚 Referanslar

1. **Diebold-Mariano Test:** Diebold, F.X., & Mariano, R.S. (1995). "Comparing predictive accuracy."
2. **ARCH-LM Test:** Engle, R.F. (1982). "Autoregressive conditional heteroscedasticity."
3. **Ljung-Box Test:** Ljung, G.M., & Box, G.E.P. (1978). "On a measure of lack of fit."
4. **Bootstrap Methods:** Efron, B., & Tibshirani, R.J. (1994). "An introduction to the bootstrap."
5. **GARCH Models:** Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity."

---

## 📞 Destek

Sorularınız için:
- 📧 Email: [proje ekibi]
- 📖 Dokümantasyon: `docs/` klasörü
- 🐛 Issue: GitHub Issues

---

**Son Güncelleme:** 2025-11-24  
**Versiyon:** 3.1.0 (Enhanced)

