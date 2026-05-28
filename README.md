# 📊 TEFAS Fon Analiz Merkezi

Altın & Emtia Fonlarına odaklanan, makine öğrenmesi destekli fon
performans analiz ve tahmin uygulaması.

---

## 🚀 Kurulum (3 adım)

### 1. Python Gereksinimleri Kur
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlat
```bash
streamlit run app.py
```

Tarayıcınızda otomatik açılır: `http://localhost:8501`

---

## 🌐 Otomatik Veri Çekme

Uygulama doğrudan TEFAS API'sine bağlanır ve Altın & Emtia
fonlarının tarihsel verilerini çeker. İnternet bağlantısı gereklidir.

## 📁 Manuel CSV Yükleme

TEFAS'tan CSV indirmek için:
1. https://www.tefas.gov.tr → Fon Bilgileri → Tarihsel Veriler
2. Fon türü olarak "Altın Fonu" veya "Kıymetli Madenler" seçin
3. Tarih aralığını belirleyin → CSV İndir
4. Uygulamada "CSV Yükle" seçeneğiyle dosyayı yükleyin

---

## 🤖 ML Modelleri

| Model | Açıklama | Güçlü Olduğu Durum |
|-------|----------|--------------------|
| **Prophet** | Trend + mevsimsellik ayrıştırması | Döngüsel hareketler |
| **Random Forest** | Özellik tabanlı pattern tanıma | Kısa vadeli dalgalanmalar |
| **Ridge Regression** | Doğrusal baseline | Düz trendler |
| **Ensemble** | Tüm modellerin ortalaması | Genel güvenilirlik |

---

## 📈 Özellikler

- **Normalize Getiri Grafiği** — Fonları aynı başlangıç noktasından karşılaştır
- **Aylık Getiri Isı Haritası** — Hangi aylar tarihsel olarak daha iyi?
- **Risk-Getiri Scatter** — Sharpe oranı ile en verimli fonu bul
- **ML Tahmin** — Kısa (1-4 hafta) ve orta (1-3 ay) vadeli fiyat öngörüsü
- **Korelasyon Matrisi** — Portföy çeşitlendirmesi için fon ilişkileri
- **VaR & Max Drawdown** — Kötü senaryo risk analizi

---

## ⚠️ Yasal Uyarı

Bu uygulama **yatırım tavsiyesi vermez**. Geçmiş performans gelecek
getiriyi garanti etmez. Yatırım kararlarınızı bir finansal danışmana
danışarak alınız.

---

## 📁 Dosya Yapısı

```
tefas_analyzer/
├── app.py           # Ana Streamlit uygulaması
├── data_loader.py   # TEFAS API & CSV yükleme
├── ml_models.py     # Prophet, RF, LinReg modelleri
├── analyzer.py      # Risk & performans metrikleri
├── requirements.txt # Python bağımlılıkları
└── README.md        # Bu dosya
```
