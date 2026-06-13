# Gerçek Zamanlı İHA Nesne Tespit Sistemi

Bu proje, İHA (İnsansız Hava Aracı) görüntülerinden insan ve araç tespiti yapabilen gerçek zamanlı bir bilgisayarlı görü sistemi geliştirmek amacıyla hazırlanmıştır.

Proje kapsamında VisDrone2019 veri seti kullanılmış ve YOLO tabanlı nesne tespit modelleri eğitilmiştir.

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

# Model Sonuçları

## YOLOv8n (50 Epoch)

| Metrik    | Sonuç |
| --------- | ----- |
| Precision | 0.634 |
| Recall    | 0.433 |
| mAP50     | 0.470 |
| mAP50-95  | 0.279 |

Sınıf bazında mAP50 sonuçları:

| Sınıf  | mAP50 |
| ------ | ----- |
| Person | 0.397 |
| Car    | 0.764 |
| Truck  | 0.295 |
| Bus    | 0.423 |

---

## YOLOv8s (50 Epoch)

Eğitim devam ediyor...

---

## YOLO11n

Planlanıyor.

---

## YOLO11s

Planlanıyor.

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

- YOLO11 model karşılaştırmaları
- Gerçek zamanlı video işleme
- ByteTrack entegrasyonu
- Çoklu nesne takibi
- Nesne hareket yönü analizi
- Koordinat çıkarımı
- Hız tahmini
- Gözetleme paneli geliştirilmesi

---

# Sonuç

Bu proje kapsamında drone görüntülerinden insan ve araç tespiti yapabilen bir YOLO tabanlı nesne tespit sistemi geliştirilmiştir. Sistem GPU destekli olarak eğitilmiş ve VisDrone veri seti üzerinde başarılı sonuçlar elde edilmiştir.

Proje ilerleyen aşamalarda nesne takibi ve hareket analizi özellikleri ile genişletilecektir.

---
