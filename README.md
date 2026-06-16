# Gerçek Zamanlı İHA Nesne Tespit Sistemi

Bu proje, drone/İHA görüntülerinden **person** ve **vehicle** tespiti yapan
YOLO tabanlı bir bilgisayarlı görü sistemidir. VisDrone2019 veri seti üzerinde
farklı YOLO modelleri ve sınıf yapılandırmaları denenmiş, güncel final model
olarak **YOLO11s 2-Class** seçilmiştir.

## Proje Özellikleri

- VisDrone annotation verilerini YOLO formatına dönüştürme
- Final model ile `Person` ve `Vehicle` tespiti
- Önceki deney olarak 4-class `Person`, `Car`, `Truck`, `Bus` sürümünün korunması
- Video üzerinde frame bazlı YOLO11s nesne tespiti
- ByteTrack ile çoklu nesne takibi ve kalıcı `track_id` atama
- Bounding box, sınıf adı ve güven skorunun görüntüye çizilmesi
- Nesne merkez koordinatının çıkarılması ve merkez noktasının işaretlenmesi
- Aktif nesne sayımı ve benzersiz track ID sayımı
- Hareket yönü ve yörünge çizimi
- Piksel tabanlı göreli hız tahmini
- Anlık inference FPS değerinin görüntülenmesi
- İşlenmiş videonun MP4 formatında kaydedilmesi
- Tespit ve takip sonuçlarının CSV olarak kaydedilmesi
- Yerel video dosyası ve doğrudan HTTP(S) MP4 URL desteği
- Model karşılaştırma grafikleri ve doğrulama görselleri

## Kullanılan Teknolojiler

- Python 3.11
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- CUDA 12.4
- NVIDIA RTX 4050 Laptop GPU

## Veri Seti

Projede VisDrone2019 Detection veri seti kullanılmıştır. Güncel final veri seti
2 sınıflıdır:

| VisDrone sınıfı | YOLO ID | Final sınıf |
| --- | ---: | --- |
| pedestrian, people | 0 | person |
| car, van, truck, bus, motor | 1 | vehicle |

`bicycle`, `tricycle` ve `awning-tricycle` sınıfları final 2-class veri
setinden çıkarılmıştır.

### Veri Seti İstatistikleri

| Bölüm | Görüntü sayısı |
| --- | ---: |
| Eğitim | 6.471 |
| Doğrulama | 548 |
| Test | 1.610 |

2-class veri setinde toplam **433.232 annotation** bulunmaktadır:

| Sınıf | Annotation sayısı |
| --- | ---: |
| person | 147.747 |
| vehicle | 285.485 |

Önceki 4-class deney setinde `Person`, `Car`, `Truck` ve `Bus` sınıfları ayrı
tutulmuştur. Bu sürüm karşılaştırma ve deney amaçlı korunmaktadır.

## Veri Ön İşleme

VisDrone annotation dosyaları özel Python scriptleriyle YOLO formatına
dönüştürülmüştür:

- Annotation satırlarının okunması ve sınıf eşlemesi
- Bounding box koordinatlarının normalize edilmesi
- Hatalı annotation satırlarının temizlenmesi
- Eğitim, doğrulama ve test klasörlerinin oluşturulması

Final 2-class dönüştürme scripti:

```text
src/convert_visdrone_2class.py
data_2class.yaml
dataset/yolo_2class/
```

Önceki 4-class deney scripti:

```text
src/convert_visdrone_4class.py
data_4class.yaml
dataset/yolo_4class/
```

## Model Karşılaştırma Sonuçları

| Model | Epoch | Input Size | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 50 | 640 | 0.634 | 0.433 | 0.470 | 0.279 |
| YOLOv8s | 50 | 640 | 0.688 | 0.514 | 0.552 | 0.342 |
| YOLO11n | 50 | 960 | 0.682 | 0.524 | 0.568 | 0.359 |
| YOLO11s (4-class) | 50 | 960 | 0.747 | 0.582 | 0.638 | 0.415 |
| **YOLO11s (2-class)** | **50** | **960** | **0.787** | **0.638** | **0.710** | **0.407** |

- En yüksek mAP50 sonucu **YOLO11s 2-Class** modelinde elde edilmiştir.
- Person/Vehicle yaklaşımı, Car/Truck/Bus ayrımına göre daha kararlı sonuç vermiştir.
- Vehicle sınıfında **mAP50 = 0.818** elde edilmiştir.

Eğitim ve doğrulama görselleri aşağıdaki klasörlerde tutulur:

```text
outputs/v8n_4class_50/
outputs/v8s_4class_50/
outputs/yolo11n_4class_960/
outputs/yolo11s_4class_960/
```

Bu klasörlerde eğitim metrik grafikleri (`results.png`), confusion matrix,
normalize confusion matrix, sınıf dağılımı (`labels.jpg`) ve doğrulama tahmin
örnekleri bulunur. Görseller model davranışını ve karşılaştırma sonuçlarını
GitHub üzerinden incelemek amacıyla repoya dahil edilmiştir.

## Final Model

Güncel final model:

```text
models/yolo11s_2class_960_best.pt
```

Model yapılandırması:

```text
YOLO11s 2-Class
50 Epoch
Input Size: 960
Classes: person, vehicle
```

## En İyi Model: YOLO11s 2-Class

Karşılaştırma sonucunda en yüksek mAP50 değerini **YOLO11s 2-Class** modeli
sağlamış ve projenin final modeli olarak seçilmiştir.

### Genel Sonuçlar

| Metrik | Sonuç |
| --- | ---: |
| Precision | 0.787 |
| Recall | 0.638 |
| mAP50 | 0.710 |
| mAP50-95 | 0.407 |

### Sınıf Bazında Sonuçlar

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Person | 0.731 | 0.537 | 0.602 | 0.267 |
| Vehicle | 0.842 | 0.740 | 0.818 | 0.547 |

Önceki YOLO11s 4-class modeli başarılı bir deney olarak korunmaktadır; ancak
final sistemde person/vehicle ayrımı kullanılmaktadır.

## Kurulum

Proje kök dizininde sanal ortamı oluşturun ve bağımlılıkları yükleyin:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Video Inference Kullanımı

Video inference pipeline:

```text
src/inference_video.py
```

Script videoyu frame frame işler. Her tespit için bounding box, sınıf adı,
confidence değeri ve merkez noktası çizilir. Sol üst köşede anlık FPS bilgisi
gösterilir.

`--source` parametresi yerel video yolu veya doğrudan video dosyası döndüren
bir HTTP(S) URL kabul eder. URL ile verilen video geçici olarak indirilir ve
işlem tamamlandıktan sonra silinir.

YouTube sayfa bağlantıları desteklenmez. URL'nin doğrudan `.mp4` gibi bir
video dosyası döndürmesi gerekir.

Örnek yerel video:

```powershell
python src/inference_video.py --source data/sample_videos/test.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960
```

Örnek doğrudan MP4 URL:

```powershell
python src/inference_video.py --source "https://example.com/video.mp4" --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960
```

Inference çıktıları:

```text
outputs/videos/<video_adi>_detected.mp4
outputs/logs/detections.csv
```

Detection CSV kolonları:

```text
frame,class,confidence,x1,y1,x2,y2,center_x,center_y
```

## ByteTrack ile Nesne Takibi

Takip pipeline'ı:

```text
src/track_video.py
```

Final YOLO11s 2-Class tespitleri ByteTrack ile eşleştirilerek her nesneye bir
`track_id` atanır. Her ID için geçmiş merkez noktaları tutulur, yörünge
çizilir ve hareket yönü `left`, `right`, `up`, `down` veya `stable` olarak
hesaplanır.

Yerel video:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2
```

Varsayılan demo görünümü sade ve sunuma uygundur. Panelde yalnızca aktif toplam,
aktif vehicle, aktif person ve FPS gösterilir. Kutu etiketlerinde de yalnızca
ID, sınıf ve confidence yer alır:

```text
ID 12 | Vehicle 0.87
```

Detaylı debug/demo görünümü için yön, hız ve unique track sayısı ayrıca
açılabilir:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2 --show-direction --show-speed --show-unique
```

Doğrudan MP4 URL:

```powershell
python src/track_video.py --source "https://example.com/video.mp4" --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2
```

Tracking çıktıları:

```text
outputs/videos/<video_adi>_tracked.mp4
outputs/logs/tracking.csv
```

Tracking CSV kolonları final 2-class mantığıyla şu bilgileri içerir:

```text
frame,track_id,class,confidence,x1,y1,x2,y2,center_x,center_y,direction,speed_px_per_sec,active_total,active_vehicle,active_person,unique_total,unique_vehicle,unique_person
```

## Geliştirilmiş Nesne Sayımı

Takip sistemi aktif ve benzersiz olmak üzere iki farklı sayım üretir.

**Active count**, mevcut frame'de YOLO tarafından tespit edilen ve sınıfına ait
confidence eşiğini geçen detection sayısıdır. Track ID veya minimum track ömrü
şartı kullanmaz. Final model için ana sayaçlar:

- `active_person`
- `active_vehicle`
- `active_total = active_person + active_vehicle`

**Unique count**, video boyunca oluşan benzersiz `track_id` değerlerini
kümülatif olarak tutar. Bu değer fiziksel nesnelerin kusursuz yeniden
kimliklendirilmesi değil, benzersiz track ID sayısıdır. Final model için:

- `unique_person`
- `unique_vehicle`
- `unique_total = unique_person + unique_vehicle`

Video üzerinde örnek panel:

```text
Active Total: 32
Active Vehicle: 27
Active Person: 5
FPS: 24.8
```

`Unique Tracks` değeri CSV içinde kalır, ancak varsayılan video panelinde
gösterilmez. Panelde görmek için `--show-unique` kullanılabilir.

4-class deney sürümünde `active_car`, `active_truck`, `active_bus` gibi alt
araç kırılımları kullanılmıştır; final 2-class sistemde bu kırılımlar
`active_vehicle` altında birleştirilmiştir.

## Speed Estimation

Her `track_id` için merkez koordinatı geçmişi kullanılarak piksel tabanlı hız
tahmini yapılır. Bounding box titreşimini azaltmak için merkez koordinatlarına
iki noktalık moving average uygulanır.

Hız hesabı tek bir frame farkına göre yapılmaz. Track en az 10 merkez gözlemine
ulaştığında son 10 yumuşatılmış noktanın ilk ve son merkezi arasındaki Öklid
yer değiştirmesi hesaplanır. Gerçek frame numarası farkı kaynak video FPS
değerine bölünerek pencerenin süresi bulunur:

```text
pixel_distance = distance(first_center, last_center)
time_difference = frame_difference / fps
speed_px_per_sec = pixel_distance / time_difference
```

İlk ve son merkez arasındaki yer değiştirme varsayılan olarak 2 pikselden
küçükse nesne `stable` kabul edilir ve hız `0.0 px/s` olarak yazılır.

Video üzerindeki hız bilgisi varsayılan olarak gizlidir. Etikette görmek için
`--show-speed` kullanılmalıdır. Yön bilgisi de aynı şekilde `--show-direction`
argümanı verilirse gösterilir:

```text
ID 12 | Vehicle 0.87 | right | 42.3 px/s
```

Bu hız değeri gerçek km/h değildir. Piksel tabanlı göreli ve yaklaşık bir
hızdır; kamera hareketi, perspektif, irtifa ve görüntü ölçeğinden etkilenir.
Gerçek dünya hızı için kamera kalibrasyonu ve sahne ölçeği gerekir.

## Proje Yapısı

```text
UAV_Object_Detection/
├── data/
├── dataset/
├── models/
│   └── yolo11s_2class_960_best.pt
├── outputs/
│   ├── logs/
│   ├── videos/
│   ├── v8n_4class_50/
│   ├── v8s_4class_50/
│   ├── yolo11n_4class_960/
│   └── yolo11s_4class_960/
├── src/
│   ├── convert_visdrone_2class.py
│   ├── convert_visdrone_4class.py
│   ├── inference_video.py
│   └── track_video.py
├── data_2class.yaml
├── data_4class.yaml
├── requirements.txt
└── README.md
```

## Gelecek Çalışmalar

- ByteTrack parametrelerinin farklı sahneler için optimize edilmesi
- Yeniden kimliklendirme ile uzun süreli nesne ID takibi
- Kamera hareketi telafisi
- Piksel hızını gerçek dünya hızına çevirmek için kamera kalibrasyonu
- Nesne yörüngelerinin kaydedilmesi
- Canlı kamera ve RTSP akış desteği
- Tespit ve takip sonuçlarını gösteren web dashboard

## Sonuç

Şu ana kadarki en başarılı model **YOLO11s 2-Class** modelidir:

- mAP50 = 0.710
- mAP50-95 = 0.407

Bu model proje içerisinde nesne tespiti, ByteTrack tabanlı takip, aktif nesne
sayımı, yön analizi ve piksel tabanlı hız tahmini için kullanılmaktadır.
