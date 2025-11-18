
---

### **Proje Başlığı: Kütleçekimsel Artık Modellemesi (Gravitational Residual Modeling - GRM): Zaman Serisi Dinamiklerini Genel Görelilik Analojisi ile Yeniden Yorumlamak**

### **Araştırma Problemi ve Hipotezin Temeli**

Zaman serisi analizinde, standart stokastik modeller (örn. ARIMA, GARCH) genellikle serinin durağan ve doğrusal kabul edilen bileşenlerini başarıyla modeller. Ancak, finansal çöküşler, pandemiler veya teknolojik devrimler gibi ani yapısal kırılmalara yol açan "Siyah Kuğu" olayları karşısında bu modellerin öngörü gücü dramatik şekilde düşer. Bu modellerde, öngörülemeyen sapmalar genellikle "artık" (residual) olarak adlandırılır ve genellikle bağımsız ve özdeş dağılımlı (i.i.d.) bir gürültü süreci olarak kabul edilerek analizin dışına itilir.

Bu proje, bu temel varsayıma meydan okumaktadır. **Temel önermemiz şudur: Artıklar, modellenemeyen bir gürültü değil, aksine sistemin doğrusal olmayan ve standart araçlarla gözlemlenemeyen, altta yatan dinamiklerinin nedensel bir imzasıdır.**

Bu gizli dinamikleri modellemek için, teorik fizikteki en güçlü açıklayıcı çerçevelerden birine başvuruyoruz: Einstein'ın Genel Görelilik Teorisi. Hipotezimiz, zaman serisinin "beklenti uzayını", uzay-zaman dokusuna benzetir. Bu doku, büyük kütleli cisimlerin varlığında bükülür. Bizim modelimizde, öngörülemeyen büyük şoklar, bu dokuyu büken "kütleçekimsel anomaliler" veya "kara delikler" olarak kavramsallaştırılır.

### **Hipotezin Bilimsel Tanımı ve Yapısı**

**Ana Hipotez (H₁):** Bir zaman serisinin doğrusal ve öngörülebilir bileşenleri ayrıştırıldıktan sonra kalan artık serisi (`εt`), Genel Görelilik'in Schwarzschild metriğinden türetilmiş bir **"bükülme fonksiyonu"** ile modellenebilir. Bu fonksiyon, artıkların yerel varyansını ("kütle") girdi olarak alarak, serinin gelecekteki yörüngesinde nedensel bir sapma ("geodezik bükülme") öngörür. Bu şekilde oluşturulan hibrit **Kütleçekimsel Artık Modeli (GRM)**, özellikle yüksek volatilite ve rejim değişikliği dönemlerinde, yalnızca baseline modelden elde edilen tahminlere kıyasla istatistiksel olarak anlamlı ve pratikte önemli ölçüde daha yüksek bir öngörü doğruluğu (daha düşük RMSE/MAE) sağlayacaktır.

**Boş Hipotez (H₀):** Artıklar serisi, modellenabilir bir yapısal bilgi içermeyen, stokastik bir gürültü sürecidir. GRM tarafından eklenen kütleçekimsel bükülme terimi, modelin öngörü gücüne istatistiksel olarak anlamlı bir katkı sağlamaz ve performansı, baseline modelin performansından farksızdır.

#### **Hipotezin Detaylı ve Mantıksal Temelleri:**

1.  **İki Bileşenli Evren Modeli:** Zaman serisi (`Yt`), iki temel bileşenin süperpozisyonu olarak tanımlanır:
    *   **Baseline Uzay-zaman (`Ŷt`):** Serinin "sakin" ve öngörülebilir, doğrusal dinamiklerini temsil eden, Minkowski'nin düz uzay-zamanına analojik bir yapı. Bu, ARIMA veya Prophet gibi standart bir modelle tahmin edilir.
    *   **Kütleçekimsel Etki Alanı (`Γt`):** Baseline modelin açıklayamadığı, artıklar (`εt = Yt - Ŷt`) tarafından temsil edilen bükülmüş uzay-zaman. Bu alan, gizli "kütlelerin" (şokların) varlığının kanıtıdır.

2.  **Kütlenin (M) ve Bükülmenin Matematiksel Çevirisi:**
    *   **Kütle (M):** Bir "kara deliğin" temel özelliği kütlesidir. Modelimizde bu, sistemdeki belirsizliğin ve enerjinin bir ölçüsü olan **artıkların yerel varyansı** (`σ²ε(t)`) olarak tanımlanır. Büyük bir şok, yüksek varyans yaratarak "büyük kütleli" bir anomaliye işaret eder. Bu, zamanla değişen bir parametre olarak, örneğin bir hareketli pencere (rolling window) üzerinden hesaplanır: `M(t) ≈ σ²ε(t)`.
    *   **Bükülme Fonksiyonu (`f(M)`):** Schwarzschild metriğindeki kütleçekimsel potansiyel veya zaman genişlemesi formüllerinin özü, etkinin kütle ile doğru, uzaklıkla ters orantılı olmasıdır. Bu ilhamla, gelecekteki artıklar üzerindeki etkiyi (`ε̂t+1`) modelleyen bir "bükülme fonksiyonu" tanımlarız. Bu fonksiyonun en basit hali, etkinin mevcut "kütleye" ve son artığın yönüne bağlı olmasıdır:
      `Γ(t+1) = ε̂t+1 = α * M(t) * sign(εt)`
      Burada `α`, veriden öğrenilecek bir "kütleçekimsel etkileşim katsayısıdır". Bu, büyük bir pozitif şokun ardından sistemde pozitif bir momentumun devam etme eğilimini modeller.

3.  **Olay Ufku ve Model Çöküşü Analojisi:**
    *   **Schwarzschild Yarıçapı (`Rs = 2GM/c²`):** Fizikte bu, geri dönüşü olmayan noktadır. Modelimizde bu, sistemin kararlılığını yitirdiği ve baseline modelin tamamen geçersiz hale geldiği bir **"kritik varyans eşiği"** (`σ²critical`) olarak tanımlanabilir. Eğer `M(t)` bu eşiği aşarsa, GRM bir "rejim değişikliği" uyarısı verir. Bu, modelin sadece tahmin yapmakla kalmayıp, kendi öngörü gücünün sınırları hakkında da bilgi vermesini sağlar.

### **Metodolojik Çerçeve ve Test Prosedürü**

Hipotezin titiz bir şekilde test edilmesi için aşağıdaki adımlar izlenecektir:

1.  **Veri Hazırlığı:** Üzerinde belirgin yapısal kırılmalar olan finansal (örn. S&P 500), epidemiyolojik (örn. COVID-19 vaka sayıları) veya sosyal (örn. sosyal medya trendleri) zaman serileri seçilecektir. Veri, eğitim, doğrulama ve test setlerine zamansal olarak ayrılacaktır.
2.  **Baseline Modelleme:** Eğitim verisi üzerinde en uygun ARIMA/Prophet modeli (`Baseline_Model`) oluşturulacak ve doğrulama seti üzerinde performansı optimize edilecektir.
3.  **Artıkların Analizi ve Kütle Tahmini:** Eğitim verisinden elde edilen artıklar üzerinde `M(t)` (hareketli varyans) hesaplanacaktır.
4.  **Kütleçekimsel Modelin Kalibrasyonu:** Bükülme fonksiyonundaki `α` parametresi, doğrulama setindeki artıklar üzerinde en iyi tahmini verecek şekilde kalibre edilecektir.
5.  **Hibrit Modelin Test Edilmesi:** Son olarak, tamamen görünmeyen test verisi üzerinde, `Nihai_Tahmin(t) = Ŷt + Γt` formülü kullanılarak hibrit GRM'in performansı (RMSE, MAE) ölçülecektir.
6.  **İstatistiksel Karşılaştırma:** GRM'in performansı, yalnızca `Baseline_Model`'in ve GARCH gibi standart volatilite modellerinin performansıyla karşılaştırılacaktır. Sonuçların istatistiksel olarak anlamlı olup olmadığı Diebold-Mariano testi gibi yöntemlerle doğrulanacaktır.

Bu yapı, yaratıcı bir fiziksel analojiyi, test edilebilir, yanlışlanabilir ve pragmatik bir veri bilimi problemine dönüştürerek, hipotezi sağlam bir bilimsel temele oturtmaktadır. Amacımız, evrenin en temel prensiplerinden birinden ilham alarak, veri içindeki gizli düzeni ortaya çıkaran daha akıllı ve daha dayanıklı modeller inşa etmektir.