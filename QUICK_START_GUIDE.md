# 🚀 **HIZLI BAŞLANGIÇ KILAVUZU - ENHANCED GRM**

## ⚡ **5 DAKİKADA BAŞLA**

### **1️⃣ Coverage Kontrolü (2 dk)**

```bash
python scripts/validate_regime_coverage.py
```

**Çıktı yorumla:**
- ✅ "Status: ✅ OK" → Adım 3'e geç
- ❌ "Status: ❌ PROBLEM" → Adım 2'ye geç

---

### **2️⃣ Enhanced Setup (3 dk)**

```bash
python main_multi_body_grm_enhanced.py
```

**Ne yapar:**
- Auto-tuned DBSCAN
- Stratified split
- Coverage validation

---

### **3️⃣ Full Testing (5 dk)**

```bash
python main.py --multi-body
```

**Sonuçlar:**
- `./results/multi_body_grm_results.txt`
- `./results/regime_coverage_report.txt`

---

## 📊 **SONUÇLARI DEĞERLENDİR**

### **✅ Başarı Kriterleri**

```
Test Regimes ≥ 3          → ✅
Coverage ≥ 50%            → ✅
DM p-value < 0.05         → ✅
RMSE improvement > 1%     → ✅
```

### **❌ Sorun Varsa**

1. **Test'te 1 rejim:**
   ```bash
   python scripts/compare_split_strategies.py
   # En iyi stratejiyi seç
   ```

2. **Coverage < 50%:**
   - config_enhanced.py'de test_ratio artır
   - Stratified split kullan

3. **DM p-value > 0.05:**
   - Farklı varlık dene (ETH-USD, ^GSPC)
   - Test periyodunu uzat

---

## 🎯 **EN İYİ KULLANIM**

```bash
# Tam pipeline
python main_advanced_test.py                    # Feature tests
python scripts/validate_regime_coverage.py       # Coverage check
python main_multi_body_grm_enhanced.py          # Enhanced setup
python main.py --multi-body                      # Full test
```

---

## 💡 **İPUÇLARI**

- ✅ Her zaman validation'dan başla
- ✅ Stratified split'i varsayılan kullan
- ✅ Raporları oku ve analiz et
- ✅ Hopkins > 0.7 ise clustering uygun
- ✅ Test regimes ≥ 3 hedefle

---

**5 dakikada başla, production-ready sonuçlar al!**

