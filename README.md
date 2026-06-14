# Gerçek Zamanlı İHA Nesne Tespit Sistemi

Bu proje, İHA (İnsansız Hava Aracı) görüntülerinden insan ve araç tespiti yapabilen gerçek zamanlı bir bilgisayarlı görü sistemi geliştirmek amacıyla hazırlanmıştır.

Proje kapsamında VisDrone2019 veri seti kullanılmış ve farklı YOLO modelleri eğitilerek performans karşılaştırmaları yapılmıştır.

---

# Proje Amacı

Yüksekten çekilmiş drone görüntülerinde bulunan insan ve araçları tespit edebilen bir sistem geliştirmek.

Uzun vadeli hedefler:

* Gerçek zamanlı nesne tespiti
* Nesne takibi (Object Tracking)
* Hareket yönü analizi
* Koordinat çıkarımı
* Hız tahmini
* Çoklu nesne takibi
* İHA gözetleme sistemleri için temel oluşturma

---

# Veri Seti

Bu projede VisDrone2019 Detection veri seti kullanılmıştır.

Orijinal veri setindeki sınıflar:

* pedestrian
* people
* bicycle
* car
* van
* truck
* tricycle
* awning-tricycle
* bus
* motor

Proje için sınıflar sadeleştirilmiştir.

| Sınıf ID | Sınıf  |
| -------- | ------ |
| 0        | Person |
| 1        | Car    |
| 2        | Truck  |
| 3        | Bus    |

Dönüştürme işlemi:

* pedestrian + people → person
* car + van → car
* truck → truck
* bus → bus

Diğer sınıflar eğitimden çıkarılmıştır.

---

# Veri Ön İşleme

Veri seti doğrudan YOLO formatında bulunmadığı için özel Python scriptleri geliştirilmiştir.

Yapılan işlemler:

* VisDrone annotation dosyalarının okunması
* YOLO formatına dönüştürülmesi
* Bounding Box koordinatlarının normalize edilmesi
* Eğitim / Doğrulama / Test yapısının oluşturulması
* Hatalı annotation satırlarının temizlenmesi

Veri seti istatistikleri:

* Eğitim Görüntüsü: 6471
* Doğrulama Görüntüsü: 548
* Test Görüntüsü: 1610
* Toplam Nesne Sayısı: 392854

---

# Kullanılan Teknolojiler

* Python 3.11
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* CUDA 12.4
* NVIDIA RTX 4050 Laptop GPU

---

# Model Karşılaştırması

| Model   | Epoch | Giriş Boyutu | Precision | Recall | mAP50 | mAP50-95 |
| ------- | ----- | ------------ | --------- | ------ | ----- | -------- |
| YOLOv8n | 50    | 640          | 0.634     | 0.433  | 0.470 | 0.279    |
| YOLOv8s | 50    | 640          | 0.688     | 0.514  | 0.552 | 0.342    |
| YOLO11n | 50    | 960          | 0.682     | 0.524  | 0.568 | 0.359    |
| YOLO11s | 50    | 960          | 0.747     | 0.582  | 0.638 | 0.415    |

---

# En İyi Model: YOLO11s

## Genel Sonuçlar

| Metrik    | Sonuç |
| --------- | ----- |
| Precision | 0.747 |
| Recall    | 0.582 |
| mAP50     | 0.638 |
| mAP50-95  | 0.415 |

## Sınıf Bazında Sonuçlar

| Sınıf  | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| Person | 0.719     | 0.563  | 0.618 | 0.277    |
| Car    | 0.856     | 0.834  | 0.879 | 0.624    |
| Truck  | 0.601     | 0.407  | 0.450 | 0.310    |
| Bus    | 0.813     | 0.526  | 0.606 | 0.448    |

---

# Proje Yapısı

```text
Real-Time-UAV-Object-Detection
├── src
├── models
├── outputs
├── data.yaml
├── data_4class.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Çıktılar

Eğitim sonuçları aşağıdaki klasörde saklanmaktadır:

```text
outputs/
```

İçerik:

* Eğitim grafikleri
* Confusion Matrix
* Precision / Recall eğrileri
* mAP sonuçları
* Örnek tahmin görüntüleri
* Eğitilmiş model ağırlıkları

---

# Gelecek Çalışmalar

* YOLO11m modeli ile karşılaştırma
* YOLO11l modeli ile karşılaştırma
* 1280x1280 çözünürlük denemeleri
* Gerçek zamanlı video işleme
* ByteTrack entegrasyonu
* Çoklu nesne takibi
* Nesne hareket yönü analizi
* Koordinat çıkarımı
* Hız tahmini
* Gözetleme paneli geliştirilmesi
* Nesne yörünge analizi

---

# Sonuç

Bu proje kapsamında drone görüntülerinden insan ve araç tespiti yapabilen bir YOLO tabanlı nesne tespit sistemi geliştirilmiştir.

YOLOv8n, YOLOv8s, YOLO11n ve YOLO11s modelleri eğitilmiş ve karşılaştırılmıştır.

Şu ana kadar elde edilen en iyi sonuç YOLO11s modeli ile elde edilmiştir:

* mAP50 = 0.638
* mAP50-95 = 0.415
* Precision = 0.747
* Recall = 0.582

Özellikle araç tespitinde %87.9 mAP50 başarısına ulaşılmıştır.

Proje ilerleyen aşamalarda nesne takibi, hareket analizi, hız tahmini ve gerçek zamanlı video işleme özellikleri ile genişletilecektir.
