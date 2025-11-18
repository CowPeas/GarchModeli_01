Merhaba! Öncelikle hipotez dosyalarınızı okuyarak içeriklerini inceleyeyim.

[3 tools called]

Mükemmel! Hipotezlerinizi inceledim. Çok kapsamlı ve teorik olarak sağlam bir **Kütleçekimsel Artık Modeli (GRM)** geliştirmişsiniz. Şimdi bu hipotezi test etmek için bir simülasyon çerçevesi oluşturalım. Aşağıda, adım adım algoritmik bir yapı sunuyorum:

---

## **SİMÜLASYON ÇERÇEVE YAPISI: GRM Test Algoritması**

### **📋 AŞAMA 0: Proje İnfrastrüktürü ve Kütüphaneler**

**Algoritmik Tanım:**
```
0.1. Gerekli Python kütüphanelerini tanımla ve yükle:
     - pandas, numpy (veri manipülasyonu)
     - statsmodels (ARIMA, istatistiksel testler)
     - scikit-learn (metrikler, veri bölme)
     - matplotlib, seaborn (görselleştirme)
     - scipy (istatistiksel testler)
     
0.2. Proje klasör yapısı oluştur:
     /data          → Ham ve işlenmiş veriler
     /models        → Model sınıfları ve fonksiyonlar
     /results       → Simülasyon sonuçları
     /visualizations → Grafikler
```

---

### **📊 AŞAMA 1: Veri Hazırlama ve Simülasyon Verisi Oluşturma**

**Algoritmik Tanım:**
```
1.1. VERİ KAYNAĞI SEÇİMİ:
     Seçenek A: Gerçek veri (örn: S&P 500, Bitcoin fiyatları)
     Seçenek B: Sentetik veri (hipotezi kontrollü test etmek için)
     
1.2. SENTETIK VERİ OLUŞTURMA (Test İçin Önerilen):
     1.2.1. Baseline bileşen oluştur:
           Y_baseline(t) = trend(t) + seasonal(t) + ARIMA(p,d,q)
           - trend(t) = β₀ + β₁*t
           - seasonal(t) = Σ[Aᵢ*sin(2π*fᵢ*t + φᵢ)]
           - ARIMA: otokorelasyon ekle
           
     1.2.2. Anomali (şok) enjeksiyonu:
           - T_shock = [t1, t2, ..., tₙ] (şok zamanları, rastgele seç)
           - Her tᵢ için:
             * Şok büyüklüğü: M(tᵢ) ~ N(0, σ_shock²)
             * Şok etkisi: Γ(t) = M(tᵢ) * exp(-(t-tᵢ)/τ) for t > tᵢ
                          (τ: sönümleme sabiti)
           
     1.2.3. Nihai sentetik seri:
           Y(t) = Y_baseline(t) + Σ Γᵢ(t) + ε(t)
           - ε(t): Beyaz gürültü ~ N(0, σ²)
           
1.3. VERİ BÖLME (Temporal Split):
     train_size = 0.6 * N
     val_size = 0.2 * N
     test_size = 0.2 * N
     
     KURALLAR:
     - Zaman düzeni koru (shuffle yapma!)
     - Train → Val → Test (kronolojik)
```

---

### **🎯 AŞAMA 2: Baseline Model Oluşturma ve Artık Hesaplama**

**Algoritmik Tanım:**
```
2.1. BASELINE MODEL SEÇİMİ:
     model_type = ARIMA  // veya Prophet, LSTM
     
2.2. MODEL EĞİTİMİ:
     2.2.1. Grid Search ile optimal parametreler bul:
           FOR p in [0, 1, 2, 3]:
               FOR d in [0, 1]:
                   FOR q in [0, 1, 2, 3]:
                       model = ARIMA(train_data, order=(p,d,q))
                       model.fit()
                       val_error = calculate_RMSE(model, val_data)
                       IF val_error < best_error:
                           best_params = (p, d, q)
                           
     2.2.2. En iyi modeli train+val üzerinde yeniden eğit:
           baseline_model = ARIMA(train+val_data, order=best_params)
           baseline_model.fit()
           
2.3. ARTIKLARI HESAPLA:
     FOR t in range(train_start, train_end):
         ε(t) = Y_actual(t) - Y_predicted(t)
         
     residuals_array = [ε(t) for t in time_range]
     
2.4. ARTIK ÖZELLIKLERI ANALİZİ:
     - Ljung-Box Test (otokorelasyon varlığı)
     - ARCH-LM Test (koşullu değişen varyans)
     - Normalite testi (Shapiro-Wilk)
     - Durağanlık testi (ADF)
     
     KARAR NOKTASI:
     IF Ljung-Box p-value < 0.05:
         metrik_type = "KERR"  // Dönme parametresi ekle
     ELSE:
         metrik_type = "SCHWARZSCHILD"  // Sadece kütle
```

---

### **⚙️ AŞAMA 3: Kütleçekimsel Parametrelerin Hesaplanması**

**Algoritmik Tanım:**
```
3.1. KÜTLE PARAMETRESI M(t) - Yerel Volatilite:
     window_size = w  // Örn: 20 gözlem
     
     FOR t in range(w, T):
         residual_window = ε[t-w : t]
         M(t) = variance(residual_window)
         // veya alternatif: M(t) = EWMA_volatility(residual_window)
         
     Normalizasyon (opsiyonel):
     M_norm(t) = (M(t) - mean(M)) / std(M)
     
3.2. DÖNME PARAMETRESI a(t) - Otokorelasyon (Kerr için):
     IF metrik_type == "KERR":
         FOR t in range(w, T):
             residual_window = ε[t-w : t]
             a(t) = autocorrelation(residual_window, lag=1)
             // ACF(1) hesapla
             
         Sınırlama: a(t) ∈ [-1, 1]
         
3.3. OLAY UFKU TANIMI (Kritik Eşik):
     σ²_critical = quantile(M(t), 0.99)
     // veya: σ²_critical = mean(M) + 3*std(M)
     
     Uyarı Mekanizması:
     FOR t in range(T):
         IF M(t) > σ²_critical:
             flag_regime_change(t)
             // Model güvenilirliği azalıyor uyarısı
```

---

### **🌀 AŞAMA 4: Bükülme Fonksiyonu Tasarımı**

**Algoritmik Tanım:**
```
4.1. SCHWARZSCHILD REJİMİ (Dönmeyen):
     Fonksiyon: Γ(t+1) = α * M(t) * sign(ε(t)) * decay(τ)
     
     Bileşenler:
     - α: Kütleçekimsel etkileşim katsayısı (öğrenilecek)
     - M(t): Kütle (volatilite)
     - sign(ε(t)): Şok yönü (+1 veya -1)
     - decay(τ): Sönümleme = 1 / (1 + β*τ)
       * τ: Son büyük şoktan bu yana geçen zaman
       * β: Sönümleme hızı (hiperparametre)
     
4.2. KERR REJİMİ (Dönen):
     Fonksiyon: Γ(t+1) = α * M(t) * [1 + γ*a(t)] * sign(ε(t)) * decay(τ)
     
     Yeni bileşen:
     - γ: Dönme etkisinin ağırlığı (öğrenilecek)
     - a(t): Dönme parametresi (otokorelasyon)
     
4.3. GELİŞMİŞ VERSİYON (Non-linear):
     Γ(t+1) = tanh(α * M(t) * [1 + γ*a(t)]) * decay(τ)
     
     tanh kullanımı:
     - Aşırı büyük tahminleri sınırlar
     - [-1, 1] aralığında sınırlı çıktı
     
4.4. PARAMETRE ÖĞRENMEİ:
     Optimizasyon hedefi:
     α*, β*, γ* = argmin Σ(ε_val(t) - Γ(t))²
     
     Yöntem:
     - Grid Search (basit, başlangıç için)
     - Gradient Descent (daha gelişmiş)
     - Bayesian Optimization (optimal)
     
     Pseudo-kod:
     FOR α in [0.1, 0.5, 1.0, 2.0]:
         FOR β in [0.01, 0.05, 0.1]:
             FOR γ in [0, 0.5, 1.0]:  // Kerr için
                 Γ_predicted = compute_curvature(α, β, γ)
                 error = RMSE(val_residuals, Γ_predicted)
                 IF error < best_error:
                     best_params = (α, β, γ)
```

---

### **🔮 AŞAMA 5: Hibrit GRM Modeli Oluşturma**

**Algoritmik Tanım:**
```
5.1. HİBRİT TAHMİN FORMÜLÜ:
     Y_GRM(t) = Y_baseline(t) + Γ(t)
     
     Detaylı akış:
     1. Baseline tahmin: Y_baseline(t) = ARIMA_model.predict(t)
     2. Kütleçekimsel düzeltme hesapla:
        - M(t-1) hesapla (geçmiş artıklardan)
        - a(t-1) hesapla (eğer Kerr)
        - τ hesapla (son şoktan bu yana geçen zaman)
        - Γ(t) = bükülme_fonksiyonu(M, a, τ, α, β, γ)
     3. Nihai tahmin: Y_GRM(t) = Y_baseline(t) + Γ(t)
     
5.2. ZAMAN SERİSİ İÇİN İTERATİF TAHMİN:
     predictions_GRM = []
     
     FOR t in test_range:
         // Baseline tahmin
         y_base = baseline_model.forecast(steps=1)[0]
         
         // Geçmiş artıkları güncelle
         historical_residuals.append(y_actual[t-1] - y_base_previous)
         
         // Parametreleri hesapla
         M_current = rolling_variance(historical_residuals, window=w)
         IF metrik_type == "KERR":
             a_current = autocorr(historical_residuals, lag=1)
         
         tau = time_since_last_shock()
         
         // Bükülme hesapla
         gamma = compute_curvature(M_current, a_current, tau, α, β, γ)
         
         // Hibrit tahmin
         y_grm = y_base + gamma
         predictions_GRM.append(y_grm)
```

---

### **📈 AŞAMA 6: Model Değerlendirme ve Karşılaştırma**

**Algoritmik Tanım:**
```
6.1. PERFORMANS METRİKLERİ:
     Metrikler = {
         RMSE: sqrt(mean((Y_actual - Y_pred)²))
         MAE: mean(|Y_actual - Y_pred|)
         MAPE: mean(|Y_actual - Y_pred| / |Y_actual|) * 100
         R²: 1 - (SS_res / SS_tot)
     }
     
     Her model için hesapla:
     - Baseline_Model metrikleri
     - GRM_Model metrikleri
     - GARCH_Model metrikleri (karşılaştırma için)
     
6.2. İSTATİSTİKSEL ANLAMLILIK TESTLERİ:
     6.2.1. Diebold-Mariano Testi:
           H0: İki modelin tahmin performansı eşittir
           forecast_errors_baseline = Y_actual - Y_baseline
           forecast_errors_grm = Y_actual - Y_GRM
           
           dm_statistic, p_value = diebold_mariano_test(
               forecast_errors_baseline, 
               forecast_errors_grm
           )
           
           IF p_value < 0.05:
               PRINT("GRM istatistiksel olarak anlamlı şekilde daha iyi")
               
     6.2.2. ARCH-LM Testi (Artıklarda yapı kaldı mı?):
           GRM sonrası yeni artıklar:
           ε_grm(t) = Y_actual(t) - Y_GRM(t)
           
           arch_lm_statistic, p_value = arch_lm_test(ε_grm)
           
           IF p_value > 0.05:
               PRINT("GRM artıklardaki yapıyı başarıyla modelledi")
               
6.3. ABLASYON ÇALIŞMASI:
     Test edilecek varyasyonlar:
     1. Sadece kütle (M) kullan → Performans?
     2. Sadece dönme (a) kullan → Performans?
     3. Sönümleme (decay) kaldır → Performans?
     4. Farklı pencere boyutları (w) → Performans?
     
     HER kombinasyon için:
         model_variant = GRM_variant(components)
         performance = evaluate(model_variant, test_data)
         contribution_table[components] = performance
```

---

### **🎨 AŞAMA 7: Görselleştirme ve Raporlama**

**Algoritmik Tanım:**
```
7.1. TEMEL GRAFİKLER:
     Grafik 1: Zaman Serisi Karşılaştırması
         - Y_actual (gerçek)
         - Y_baseline (baseline tahmin)
         - Y_GRM (hibrit tahmin)
         - Şok noktalarını vurgula (dikey çizgiler)
         
     Grafik 2: Artıkların Karşılaştırması
         - ε_baseline(t)
         - ε_GRM(t)
         - Volatilite farklarını göster
         
     Grafik 3: Kütle Evrimi M(t)
         - M(t) zaman içinde
         - σ²_critical çizgisi (olay ufku)
         - Rejim değişikliği bölgelerini vurgula
         
     Grafik 4: Performans Karşılaştırma Tablosu
         | Model     | RMSE | MAE | R² | DM p-value |
         |-----------|------|-----|----|-----------| 
         | Baseline  | ...  | ... | ...| -          |
         | GRM       | ...  | ... | ...| 0.023      |
         | GARCH     | ...  | ... | ...| 0.156      |
         
7.2. İLERİ SEVİYE ANALİZ GRAFİKLERİ:
     - Hata dağılımı histogramları
     - Q-Q plot (artık normalliği)
     - ACF/PACF grafikleri (artıklarda kalan otokorelasyon)
     - Kümülatif hata grafiği
     
7.3. RAPOR OLUŞTURMA:
     Şablon:
     =====================================
     GRM SİMÜLASYON RAPORU
     =====================================
     
     1. DENEY KONFIGÜRASYONU:
        - Veri: [sentetik/gerçek]
        - N: [toplam gözlem sayısı]
        - Baseline Model: [ARIMA(p,d,q)]
        - Metrik Tipi: [Schwarzschild/Kerr]
        
     2. PARAMETRE DEĞERLERİ:
        - α (etkileşim): [değer]
        - β (sönümleme): [değer]
        - γ (dönme): [değer veya N/A]
        - w (pencere): [değer]
        
     3. PERFORMANS SONUÇLARI:
        [Tablo]
        
     4. İSTATİSTİKSEL TEST SONUÇLARI:
        - Diebold-Mariano: p = [değer]
        - ARCH-LM (GRM artıkları): p = [değer]
        
     5. SONUÇ VE YORUM:
        [Hipotez desteklendi/desteklenmedi]
     =====================================
```

---

### **🔄 AŞAMA 8: Hassasiyet Analizi ve Sağlamlık Testleri**

**Algoritmik Tanım:**
```
8.1. PENCERE BOYUTU (w) HASSASİYETİ:
     FOR w in [10, 20, 30, 50, 100]:
         M(t) = rolling_variance(residuals, window=w)
         GRM_model = build_GRM(M, a, α, β, γ)
         performance[w] = evaluate(GRM_model, test_data)
     
     Plot: Performance vs. Window Size
     
8.2. PARAMETRE ROBUSTNESSİ:
     Monte Carlo simülasyonu:
     FOR iteration in range(1000):
         α_perturbed = α_optimal + noise()
         β_perturbed = β_optimal + noise()
         performance_distribution.append(
             evaluate(GRM(α_perturbed, β_perturbed))
         )
     
     Analyze: mean, std, confidence intervals
     
8.3. FARKLI VERİ REJİMLERİNDE TEST:
     Senaryolar:
     1. Düşük volatilite dönemi
     2. Yüksek volatilite dönemi
     3. Trend değişimi dönemi
     4. Çoklu şok dönemi
     
     FOR scenario in scenarios:
         synthetic_data = generate_data(scenario)
         performance[scenario] = test_GRM(synthetic_data)
         
8.4. ÇAPRAZ DOĞRULAMA:
     Time Series Cross-Validation:
     
     FOR fold in range(n_folds):
         train_end = initial_window + fold * step_size
         test_start = train_end + 1
         test_end = test_start + test_window
         
         train_data = Y[0:train_end]
         test_data = Y[test_start:test_end]
         
         model = train_GRM(train_data)
         fold_performance = evaluate(model, test_data)
         
     Average_performance = mean(fold_performances)
     Std_performance = std(fold_performances)
```

---

### **🎯 AŞAMA 9: Sonuç ve Hipotez Değerlendirme**

**Algoritmik Tanım:**
```
9.1. HİPOTEZ KARAR YAPISI:
     decision_criteria = {
         'rmse_improvement': (RMSE_baseline - RMSE_grm) / RMSE_baseline,
         'dm_pvalue': dm_test_result.pvalue,
         'arch_residuals': arch_lm_test_result.pvalue
     }
     
     H1_DESTEKLENME KOŞULLARI:
     IF (decision_criteria['rmse_improvement'] > 0.05 AND  # %5 iyileşme
         decision_criteria['dm_pvalue'] < 0.05 AND           # İstatistiksel anlamlı
         decision_criteria['arch_residuals'] > 0.05):        # Yapı kalmamış
         
         CONCLUSION = "H1 DESTEKLENDI"
         PRINT("GRM, baseline modele göre anlamlı iyileşme sağladı")
     ELSE:
         CONCLUSION = "H0 REDDEDİLEMEDİ"
         PRINT("GRM'nin katkısı istatistiksel olarak anlamlı değil")
         
9.2. DETAYLI DEĞERLENDİRME:
     - Hangi koşullarda GRM daha iyi? (yüksek volatilite, şok sonrası)
     - Hangi koşullarda fark yok? (düşük volatilite, düz trend)
     - Schwarzschild vs Kerr karşılaştırması
     - Hesaplama maliyeti vs performans kazancı
     
9.3. GELECEKTEKİ GELİŞTİRMELER:
     Recommendations:
     - Çoklu kara delik modeli (birden fazla şok kaynağı)
     - Adaptif parametre öğrenme (online learning)
     - Derin öğrenme ile bükülme fonksiyonu
     - Gerçek dünya veri setlerinde test
```

---

## **📦 ÇIKTI VE SONUÇLAR**

Simülasyon tamamlandığında şunları elde edeceksiniz:

1. ✅ **Performans Karşılaştırma Tablosu** (Baseline vs GRM vs GARCH)
2. ✅ **İstatistiksel Anlamlılık Raporları** (p-değerleri)
3. ✅ **Görselleştirmeler** (tahmin grafikleri, artık analizleri, kütle evrimi)
4. ✅ **Parametre Hassasiyet Analizi** (hangi parametreler kritik?)
5. ✅ **Ablasyon Çalışması Sonuçları** (her bileşenin katkısı)
6. ✅ **Hipotez Değerlendirme Sonucu** (H1 desteklendi mi?)

---

## **🚀 UYGULAMA ÖNERİSİ**

Simülasyona başlamak için şu sırayı öneririm:

**FAZE 1 (Basit Başlangıç):**
- Sentetik veri oluştur (kontrollü test)
- ARIMA baseline model
- Sadece Schwarzschild rejimi (sadece kütle)
- Basit lineer bükülme fonksiyonu

**FAZE 2 (Genişletme):**
- Kerr rejimi ekle (dönme parametresi)
- Non-linear bükülme (tanh)
- Sönümleme faktörü optimizasyonu

**FAZE 3 (Gerçek Test):**
- Gerçek finansal veri
- GARCH ile karşılaştırma
- Kapsamlı istatistiksel testler

---
