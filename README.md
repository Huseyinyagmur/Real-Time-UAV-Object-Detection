# Gerçek Zamanlı İHA Nesne Tespit Sistemi

Bu proje, drone/İHA görüntülerinden insan ve araç tespiti yapan YOLO tabanlı
bir bilgisayarlı görü sistemidir. VisDrone2019 veri seti üzerinde farklı YOLO
modelleri eğitilmiş, karşılaştırılmış ve final model olarak **YOLO11s**
seçilmiştir.

## Proje Özellikleri

- VisDrone annotation verilerini dört sınıflı YOLO formatına dönüştürme
- Video üzerinde frame bazlı YOLO11s nesne tespiti
- ByteTrack ile çoklu nesne takibi ve kalıcı `track_id` atama
- `Person`, `Car`, `Truck` ve `Bus` sınıflarının tespiti
- Bounding box, sınıf adı ve güven skorunun görüntüye çizilmesi
- Her nesnenin merkez koordinatının hesaplanması ve işaretlenmesi
- Anlık inference FPS değerinin görüntülenmesi
- İşlenmiş videonun MP4 formatında kaydedilmesi
- Tespit sonuçlarının frame bazında CSV dosyasına yazılması
- Yerel video dosyası ve doğrudan HTTP(S) video URL desteği
- Track geçmişi, hareket yönü ve yörünge çizimi
- Benzersiz `track_id` değerlerine göre toplam ve sınıf bazlı nesne sayımı
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

Projede VisDrone2019 Detection veri seti kullanılmıştır. Orijinal sınıflar
proje için aşağıdaki şekilde sadeleştirilmiştir:

| VisDrone sınıfı | YOLO ID | Proje sınıfı |
| --- | ---: | --- |
| pedestrian, people | 0 | Person |
| car, van | 1 | Car |
| truck | 2 | Truck |
| bus | 3 | Bus |

`bicycle`, `tricycle`, `awning-tricycle` ve `motor` sınıfları eğitimden
çıkarılmıştır.

### Veri Seti İstatistikleri

| Bölüm | Görüntü sayısı |
| --- | ---: |
| Eğitim | 6.471 |
| Doğrulama | 548 |
| Test | 1.610 |

Dört sınıflı veri setinde toplam **392.854 nesne** bulunmaktadır.

## Veri Ön İşleme

VisDrone annotation dosyaları özel Python scriptleriyle YOLO formatına
dönüştürülmüştür:

- Annotation satırlarının okunması ve sınıf eşlemesi
- Bounding box koordinatlarının normalize edilmesi
- Hatalı annotation satırlarının temizlenmesi
- Eğitim, doğrulama ve test klasörlerinin oluşturulması

Dört sınıflı dönüştürme scripti:

```text
src/convert_visdrone_4class.py
```

## Model Karşılaştırma Sonuçları

| Model | Epoch | Giriş boyutu | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 50 | 640 | 0.634 | 0.433 | 0.470 | 0.279 |
| YOLOv8s | 50 | 640 | 0.688 | 0.514 | 0.552 | 0.342 |
| YOLO11n | 50 | 960 | 0.682 | 0.524 | 0.568 | 0.359 |
| **YOLO11s** | **50** | **960** | **0.747** | **0.582** | **0.638** | **0.415** |

Her modelin eğitim ve doğrulama görselleri aşağıdaki klasörlerde tutulur:

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

## En İyi Model: YOLO11s

Karşılaştırma sonucunda en yüksek genel başarıyı **YOLO11s** modeli sağlamış
ve projenin final modeli olarak seçilmiştir.

Model dosyası:

```text
models/yolo11s_4class_960_best.pt
```

### Genel Sonuçlar

| Metrik | Sonuç |
| --- | ---: |
| Precision | 0.747 |
| Recall | 0.582 |
| mAP50 | 0.638 |
| mAP50-95 | 0.415 |

### Sınıf Bazında Sonuçlar

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Person | 0.719 | 0.563 | 0.618 | 0.277 |
| Car | 0.856 | 0.834 | 0.879 | 0.624 |
| Truck | 0.601 | 0.407 | 0.450 | 0.310 |
| Bus | 0.813 | 0.526 | 0.606 | 0.448 |

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

### Komut Satırı Parametreleri

| Parametre | Açıklama | Varsayılan |
| --- | --- | --- |
| `--source` | Yerel video yolu veya doğrudan HTTP(S) video URL'si | Zorunlu |
| `--model` | YOLO model ağırlıklarının yolu | Final YOLO11s modeli |
| `--conf` | Tespit güven eşiği | `0.25` |
| `--imgsz` | Inference giriş boyutu | `960` |

## Örnek Komutlar

Yerel video:

```powershell
python src/inference_video.py --source data/sample_videos/test.mp4 --conf 0.35 --imgsz 960
```

Doğrudan video URL'si:

```powershell
python src/inference_video.py --source "https://example.com/video.mp4" --conf 0.35 --imgsz 960
```

Özel model dosyası:

```powershell
python src/inference_video.py --source video.mp4 --model models/yolo11s_4class_960_best.pt
```

## Çıktı Dosyaları

İşlenmiş video kaynak dosyanın adına göre kaydedilir:

```text
outputs/videos/<video_adi>_detected.mp4
```

Tespit kayıtları şu dosyaya yazılır:

```text
outputs/logs/detections.csv
```

CSV kolonları:

```text
frame,class,confidence,x1,y1,x2,y2,center_x,center_y
```

`outputs/videos/` ve `outputs/logs/` çalışma zamanında otomatik oluşturulur ve
üretilen büyük dosyalar Git tarafından takip edilmez.

## ByteTrack ile Nesne Takibi

Takip pipeline'ı `src/track_video.py` scriptinde bulunur. YOLO11s tespitleri
ByteTrack ile eşleştirilerek her nesneye bir `track_id` atanır. Her ID için
geçmiş merkez noktaları tutulur, yörünge çizilir ve hareket yönü `left`,
`right`, `up`, `down` veya `stable` olarak hesaplanır.

Yerel video:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --conf 0.35 --imgsz 960 --count-conf 0.50 --min-track-frames 5
```

Doğrudan MP4 URL:

```powershell
python src/track_video.py --source "https://example.com/video.mp4" --conf 0.35 --imgsz 960
```

Hareket hassasiyeti ve tutulan geçmiş uzunluğu isteğe bağlı olarak
değiştirilebilir:

```powershell
python src/track_video.py --source video.mp4 --history-length 30 --direction-threshold 8
```

Takip çıktıları:

```text
outputs/videos/<video_adi>_tracked.mp4
outputs/logs/tracking.csv
```

Takip CSV kolonları:

```text
frame,track_id,class,confidence,x1,y1,x2,y2,center_x,center_y,direction,active_total,active_vehicle,active_person,active_car,active_truck,active_bus,unique_total,unique_vehicle,unique_person,unique_car,unique_truck,unique_bus
```

## Geliştirilmiş Nesne Sayımı

Takip sistemi aktif ve benzersiz olmak üzere iki farklı sayım üretir.

**Active count**, mevcut frame'de görünen, minimum track ömrünü tamamlayan ve
confidence filtresini geçen nesneleri gösterir. Ekrandaki ana sayaçlar active
count değerleridir:

- `active_person`
- `active_car`
- `active_truck`
- `active_bus`
- `active_vehicle = active_car + active_truck + active_bus`
- `active_total`

**Unique count**, video boyunca sayım filtrelerini geçen benzersiz `track_id`
değerlerini kümülatif olarak tutar. Sınıf bazlı `unique_*` değerlerinin tamamı
CSV dosyasına yazılır; ekranda ikincil bilgi olarak yalnızca `Unique Total`
gösterilir.

ByteTrack bir nesneyi kaybedip daha sonra aynı nesneye yeni bir ID atarsa,
unique count artabilir. Bu değer fiziksel nesnelerin kusursuz yeniden
kimliklendirilmesi değil, benzersiz ve filtrelenmiş track ID sayısıdır.

Yanlış pozitif ve kısa süreli track kaynaklı aşırı sayımı azaltmak için iki
filtre uygulanır:

- Bir track varsayılan olarak en az 5 frame görülmeden sayılmaz.
- Person, car ve bus için varsayılan minimum sayım güveni `0.50` değeridir.
- Truck sınıfı için daha sıkı, sabit `0.60` minimum güven eşiği kullanılır.

Filtreler komut satırından değiştirilebilir:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --conf 0.35 --imgsz 960 --count-conf 0.50 --min-track-frames 5
```

`--conf`, YOLO ve ByteTrack pipeline'ına girecek detection eşiğini;
`--count-conf` ise oluşan track'lerin kümülatif sayıma dahil edilme eşiğini
kontrol eder. Truck sayım eşiği her zaman `0.60` değeridir.

Video üzerinde aşağıdaki bilgiler gerçek zamanlı gösterilir:

```text
Active Total: 32
Active Vehicle: 27
Active Person: 5
Active Car: 21
Active Truck: 4
Active Bus: 2
Unique Total: 46
FPS: 24.8
```

Active ve unique sınıf sayımları her takip satırının sonuna eklenen CSV
kolonlarında saklanır. Minimum track frame filtresi, genel count confidence
filtresi ve truck için `0.60` confidence eşiği iki sayım türünde de korunur.

## Proje Yapısı

```text
UAV_Object_Detection/
├── data/
├── dataset/
├── models/
│   └── yolo11s_4class_960_best.pt
├── outputs/
│   ├── logs/
│   ├── videos/
│   ├── v8n_4class_50/
│   ├── v8s_4class_50/
│   ├── yolo11n_4class_960/
│   └── yolo11s_4class_960/
├── src/
│   ├── convert_visdrone_4class.py
│   ├── inference_video.py
│   └── track_video.py
├── data_4class.yaml
├── requirements.txt
└── README.md
```

## Gelecek Çalışmalar

- ByteTrack parametrelerinin farklı sahneler için optimize edilmesi
- Yeniden kimliklendirme ile uzun süreli nesne ID takibi
- Kamera hareketi telafisi
- Piksel ve gerçek dünya tabanlı hız tahmini
- Nesne yörüngelerinin kaydedilmesi
- Canlı kamera ve RTSP akış desteği
- Tespit ve takip sonuçlarını gösteren web dashboard

## Sonuç

YOLOv8n, YOLOv8s, YOLO11n ve YOLO11s modellerinin karşılaştırılması sonucunda
YOLO11s, `0.638 mAP50` ve `0.415 mAP50-95` ile en başarılı model olmuştur.
Tamamlanan video inference pipeline sayesinde final model yerel veya doğrudan
URL ile sağlanan videolarda çalıştırılabilmekte; görsel ve yapısal tespit
çıktıları video ve CSV formatında kaydedilebilmektedir.
