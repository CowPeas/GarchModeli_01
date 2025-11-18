# 📋 GRM Ana Main Dosyası Kullanım Kılavuzu

## 🎯 Genel Bakış

`main.py` dosyası, GRM projesinin tüm fazlarını çalıştırmak için merkezi bir kontrol noktası sağlar. Detaylı loglama, progress tracking ve hata yönetimi içerir.

## 🚀 Kullanım

### Temel Kullanım

```bash
# Sadece FAZE 1 çalıştır
python main.py --phase 1

# Sadece FAZE 2 çalıştır
python main.py --phase 2

# Sadece FAZE 3 çalıştır
python main.py --phase 3

# Tüm fazları sırayla çalıştır
python main.py --all
```

### Gelişmiş Özellikler

```bash
# Ablasyon çalışması
python main.py --ablation

# Cross-validation
python main.py --cross-validation

# GRN eğitimi
python main.py --grn

# Symbolic regression discovery
python main.py --symbolic

# Unified GRM testi
python main.py --unified

# Multi-Body GRM testi
python main.py --multi-body
```

### Loglama Seçenekleri

```bash
# Sessiz mod (sadece dosyaya log yaz)
python main.py --phase 3 --quiet

# Özel log dosyası
python main.py --phase 3 --log-file logs/custom.log
```

## 📊 Log Dosyaları

Tüm loglar otomatik olarak `logs/` dizinine kaydedilir:
- Format: `logs/grm_YYYYMMDD_HHMMSS.log`
- Encoding: UTF-8
- Hem konsola hem dosyaya yazılır (--quiet kullanılmadığı sürece)

## 🔍 Özellikler

### 1. Detaylı Loglama
- Her adım için detaylı log mesajları
- Hata yönetimi ve exception tracking
- Progress tracking

### 2. Merkezi Kontrol
- Tüm fazları tek yerden çalıştırma
- Sonuç özeti ve raporlama
- Hata durumlarında graceful handling

### 3. Esnek Yapı
- Komut satırı argümanları ile kontrol
- Modüler faz yapısı
- Kolay genişletilebilir

## 📝 Örnek Çıktı

```
================================================================================
GRM (GRAVITATIONAL RESIDUAL MODEL) PROJESİ
Ana Kontrol Merkezi
Python Versiyonu: 3.10.0
Çalışma Dizini: C:\Users\...\Proje
================================================================================

================================================================================
FAZE 3 BAŞLATILIYOR
================================================================================
INFO - Başlangıç Zamanı: 2025-11-15 14:30:00
INFO - [ADIM 1] VERİ YÜKLEME
INFO - --------------------------------------------------------------------------------
INFO - [OK] MANUEL CSV BULUNDU: data/BTC-USD.csv
INFO - [OK] CSV'DEN YÜKLEME BAŞARILI! (730 gözlem)
...
```

## ⚙️ Yapılandırma

Log seviyesi ve format `GRMLogger` sınıfında yapılandırılabilir:
- `log_level`: logging.INFO, logging.DEBUG, vb.
- `verbose`: Konsola yazdırma (True/False)

## 🐛 Hata Yönetimi

- Tüm hatalar loglanır
- Faz başarısız olsa bile diğer fazlar çalışmaya devam eder
- Sonuç özetinde hata durumları gösterilir

## 📚 İlgili Dosyalar

- `main_phase1.py`: FAZE 1 implementasyonu
- `main_phase2.py`: FAZE 2 implementasyonu
- `main_phase3.py`: FAZE 3 implementasyonu
- `main_ablation_study.py`: Ablasyon çalışması
- `main_cross_validation.py`: Cross-validation
- `main_grn_train.py`: GRN eğitimi
- `main_symbolic_discovery.py`: Symbolic discovery
- `main_unified_grm.py`: Unified GRM
- `main_multi_body_grm.py`: Multi-Body GRM

## 💡 İpuçları

1. **İlk çalıştırma**: `python main.py --phase 1` ile başlayın
2. **Tüm fazlar**: `python main.py --all` ile tüm fazları sırayla çalıştırın
3. **Log takibi**: `logs/` dizinindeki log dosyalarını kontrol edin
4. **Hata durumunda**: Log dosyasına bakarak detaylı hata bilgisi alın

## 🔧 Geliştirme

Yeni bir faz eklemek için:
1. Faz fonksiyonunu ilgili `main_*.py` dosyasına ekleyin
2. `GRMMain` sınıfına yeni bir `run_*` metodu ekleyin
3. `main()` fonksiyonuna yeni argüman ekleyin

