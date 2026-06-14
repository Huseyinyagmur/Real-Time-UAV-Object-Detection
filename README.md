# Gerçek Zamanlı İHA Nesne Tespit Sistemi

Bu proje, İHA (İnsansız Hava Aracı) görüntülerinden insan ve araç tespiti yapabilen gerçek zamanlı bir bilgisayarlı görü sistemi geliştirmek amacıyla hazırlanmıştır.

Proje kapsamında VisDrone2019 veri seti kullanılmış ve YOLO tabanlı nesne tespit modelleri eğitilerek performans karşılaştırmaları yapılmıştır.

---

# Proje Amacı

Yüksekten çekilmiş drone görüntülerinde bulunan insan ve araçları tespit edebilen bir sistem geliştirmek.

Uzun vadeli hedefler:

- Gerçek zamanlı nesne tespiti
- Nesne takibi (Object Tracking)
- Hareket yönü analizi
- Koordinat çıkarımı
- Hız tahmini
- Çoklu nesne takibi
- İHA gözetleme sistemleri için temel oluşturma

---

# Veri Seti

Bu projede VisDrone2019 Detection veri seti kullanılmıştır.

Orijinal veri setindeki sınıflar:

- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

Proje için sınıflar sadeleştirilmiştir.

| Sınıf ID | Sınıf  |
| -------- | ------ |
| 0        | Person |
| 1        | Car    |
| 2        | Truck  |
| 3        | Bus    |

Dönüştürme işlemi:

- pedestrian + people → person
- car + van → car
- truck → truck
- bus → bus

Diğer sınıflar eğitimden çıkarılmıştır.

---

# Veri Ön İşleme

Veri seti doğrudan YOLO formatında bulunmadığı için özel Python scriptleri geliştirilmiştir.

Yapılan işlemler:

- VisDrone annotation dosyalarının okunması
- YOLO formatına dönüştürülmesi
- Bounding Box koordinatlarının normalize edilmesi
- Eğitim/Doğrulama/Test yapısının oluşturulması
- Hatalı annotation satırlarının temizlenmesi

Veri seti istatistikleri:

- Eğitim Görüntüsü: 6471
- Doğrulama Görüntüsü: 548
- Test Görüntüsü: 1610
- Toplam Nesne Sayısı: 392854

---

# Kullanılan Teknolojiler

- Python 3.11
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- CUDA 12.4
- NVIDIA RTX 4050 Laptop GPU

---

# Model Karşılaştırması

| Model   | Epoch | Giriş Boyutu | Precision | Recall | mAP50 | mAP50-95 |
| ------- | ----- | ------------ | --------- | ------ | ----- | -------- |
| YOLOv8n | 50    | 640          | 0.634     | 0.433  | 0.470 | 0.279    |
| YOLOv8s | 50    | 640          | 0.688     | 0.514  | 0.552 | 0.342    |
| YOLO11n | 36    | 960          | 0.682     | 0.524  | 0.568 | 0.359    |

---

# YOLO11n Sonuçları (En İyi Model)

| Metrik    | Sonuç |
| --------- | ----- |
| Precision | 0.682 |
| Recall    | 0.524 |
| mAP50     | 0.568 |
| mAP50-95  | 0.359 |

Sınıf bazında sonuçlar:

| Sınıf  | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| Person | 0.631     | 0.487  | 0.527 | 0.224    |
| Car    | 0.790     | 0.807  | 0.843 | 0.580    |
| Truck  | 0.553     | 0.341  | 0.366 | 0.252    |
| Bus    | 0.754     | 0.463  | 0.535 | 0.381    |

---

# Proje Yapısı

```text
Real-Time-UAV-Object-Detection
├── src
├── outputs
├── data_4class.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Gelecek Çalışmalar

- YOLO11s model eğitimi ve karşılaştırması
- Gerçek zamanlı video işleme
- ByteTrack entegrasyonu
- Çoklu nesne takibi
- Nesne hareket yönü analizi
- Koordinat çıkarımı
- Hız tahmini
- Gözetleme paneli geliştirilmesi
- Nesne yörünge analizi

---

# Sonuç

Bu proje kapsamında drone görüntülerinden insan ve araç tespiti yapabilen bir YOLO tabanlı nesne tespit sistemi geliştirilmiştir.

YOLOv8n, YOLOv8s ve YOLO11n modelleri eğitilmiş ve karşılaştırılmıştır. Şu ana kadar elde edilen en iyi sonuç YOLO11n modeli ile mAP50 = 0.568 ve mAP50-95 = 0.359 olarak elde edilmiştir.

Proje ilerleyen aşamalarda nesne takibi, hareket analizi ve gerçek zamanlı video işleme özellikleri ile genişletilecektir.

---
