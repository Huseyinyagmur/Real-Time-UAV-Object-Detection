# UAV Object Detection & Analytics System

**Real-Time UAV Object Detection and Traffic/Security Analytics System**

## Project Overview

Bu proje, drone/İHA videolarında **person** ve **vehicle** tespiti yapan,
tespit sonuçlarını ByteTrack ile takip eden ve trafik/güvenlik analitiği
üreten modüler bir bilgisayarlı görü sistemidir. Sistem; video inference,
nesne takibi, çizgi geçiş sayımı, ROI tabanlı analizler, hız ihlali, ters yön
tespiti, kalabalık izleme, heatmap üretimi ve final dashboard raporlamasını tek
bir portföy projesi içinde birleştirir.

Proje VisDrone2019 veri seti üzerinde eğitilen YOLO modelleriyle geliştirilmiş,
güncel final model olarak **YOLO11s 2-Class** seçilmiştir. Önceki 4-class
`Person`, `Car`, `Truck`, `Bus` deneyleri karşılaştırma amacıyla korunmaktadır.

## Final Model

```text
Model: YOLO11s 2-Class
Classes: person, vehicle
Weights: models/yolo11s_2class_960_best.pt
mAP50: 0.710
mAP50-95: 0.407
```

## Modüler Özellikler

**Core**

- Detection
- Tracking
- Video Inference
- CSV Logging

**Traffic Analytics**

- Line Crossing Counter
- Traffic Heatmap
- Traffic Flow Analysis
- Speed Violation Alert
- Wrong Way Detection

**Security Analytics**

- ROI Zone Counting
- ROI Intrusion Alert
- Pedestrian Zone Intrusion
- Crowd Detection

**Reporting**

- Project Dashboard
- JSON/CSV/PNG reports

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

- YOLO11s 4-Class modeli mAP50-95 metriğinde çok az daha yüksek sonuç üretmiştir.
- Ancak final sistemin amacı `person` ve `vehicle` tespiti ile kararlı takip yapmaktır.
- YOLO11s 2-Class modeli daha yüksek mAP50 (`0.710`), daha yüksek recall (`0.638`) ve daha stabil tracking performansı sağladığı için final model olarak seçilmiştir.
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
frame,track_id,class,confidence,x1,y1,x2,y2,center_x,center_y,direction,speed_px_per_sec,active_total,active_vehicle,active_person,unique_total,unique_vehicle,unique_person,line_vehicle_up,line_vehicle_down,line_person_up,line_person_down,line_vehicle_left,line_vehicle_right,line_person_left,line_person_right
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

## Line Crossing Counter

Line Crossing Counter, ByteTrack tarafından takip edilen `person` ve `vehicle`
nesnelerinin video üzerine çizilen sanal bir çizgiyi geçmesini sayar.
Varsayılan çizgi yataydır ve frame yüksekliğinin %50 konumuna çizilir.

Çizgi ayarları:

- `--line-orientation horizontal`
- `--line-orientation vertical`
- `--line-position 0.5`
- `--line-thickness 2`

Horizontal çizgide yönlü sayaçlar:

- `vehicle_up`
- `vehicle_down`
- `person_up`
- `person_down`

Yön anlamı:

- `Up`: Nesne çizgiyi aşağıdan yukarı geçti.
- `Down`: Nesne çizgiyi yukarıdan aşağı geçti.

Vertical çizgide yönlü sayaçlar:

- `vehicle_left`
- `vehicle_right`
- `person_left`
- `person_right`

Yön anlamı:

- `Vehicle Left`: Vehicle çizgiyi sağdan sola geçti.
- `Vehicle Right`: Vehicle çizgiyi soldan sağa geçti.
- `Person Left` / `Person Right`: Person sınıfı için aynı sol/sağ geçiş mantığı kullanılır.

Aynı `track_id` aynı yön için yalnızca bir kez sayılır. Örneğin bir nesne
horizontal çizgiyi `down` yönünde geçtikten sonra aynı ID ile tekrar `down`
yönünde sayılmaz; ancak ters yönde geçerse ayrı yön olarak sayılabilir.

Line count yalnızca takip edilen nesnenin merkez noktası çizginin bir
tarafından diğer tarafına geçtiğinde artar. Çizgiyi geçmeyen nesneler aktif
nesne sayımında görünür, ancak line crossing değerini artırmaz. Bu nedenle
çizgi konumu sahneye göre ayarlanmalıdır.

Line crossing sayımı, kısa süreli kenar tespitlerini azaltmak için
`--min-track-frames` filtresini kullanır. Bir track çizgiyi geçse bile en az bu
kadar frame takip edilmeden line count değerine eklenmez. Ayrıca person için
`--person-conf`, vehicle için `--vehicle-conf` eşiğinin altında kalan geçişler
sayılmaz.

Horizontal line y ekseninde, vertical line x ekseninde çalışır. `--line-position
0.5` görüntünün orta noktasıdır. Varsayılan konum horizontal çizgide `0.5`,
vertical çizgide `0.45` olarak seçilir. Yolun ve akışın konumuna göre `0.4`,
`0.6` veya `0.7` gibi değerler denenebilir.

Önerilen vertical demo komutu:

```powershell
python src/track_video.py --source data/sample_videos/test2.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2 --line-orientation vertical --line-position 0.45 --line-thickness 2
```

Horizontal line örneği:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2 --line-orientation horizontal --line-position 0.5 --line-thickness 2
```

Vertical line örneği:

```powershell
python src/track_video.py --source data/sample_videos/test.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2 --line-orientation vertical --line-position 0.35 --line-thickness 2
```

Panelde çizgi yönüne göre line crossing bilgileri gösterilir:

```text
Active Total: 32
Active Vehicle: 27
Active Person: 5
Line Vehicle Up: 12
Line Vehicle Down: 8
Line Person Up: 3
Line Person Down: 1
FPS: 24.8
```

Video üzerinde çizgi sarı renkle, varsayılan olarak 2 piksel kalınlıkta
çizilir ve yanında küçük `Counting Line` etiketi gösterilir. Line crossing
sayaçları CSV dosyasına da yazılır.

## ROI Zone Counting

ROI, video üzerinde belirlenen dikdörtgen ilgi bölgesidir. ROI Zone Counting
özelliği, YOLO11s 2-Class tespitlerini ByteTrack ile takip eder ve bu bölge
içindeki aktif `person` / `vehicle` sayılarını hesaplar.

Sistem ayrıca video boyunca ROI'ye giren benzersiz `track_id` değerlerini tutar.
Bir track ROI'ye ilk kez girdiğinde unique count yalnızca bir kez artar.
Track'in ROI dışından içine geçmesi `enter`, ROI içinden dışına çıkması `exit`
olayı olarak CSV dosyasına yazılır.

Bu özellik otopark, kavşak, güvenlik bölgesi ve yoğunluk analizi gibi
senaryolarda kullanılabilir.

ROI pipeline:

```text
src/roi_zone_counter.py
```

Çıktılar:

```text
outputs/videos/<video_adi>_roi.mp4
outputs/logs/roi_events.csv
```

CSV kolonları:

```text
frame,track_id,class,confidence,center_x,center_y,in_roi,event,active_roi_total,active_roi_vehicle,active_roi_person,unique_roi_total,unique_roi_vehicle,unique_roi_person
```

Varsayılan orta ROI:

```powershell
python src/roi_zone_counter.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960
```

Özel ROI:

```powershell
python src/roi_zone_counter.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --roi 500,300,1600,900 --roi-name "Highway Zone"
```

Kavşak ROI:

```powershell
python src/roi_zone_counter.py --source data/sample_videos/test4.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --roi 700,350,2300,1350 --roi-name "Intersection Zone"
```

## ROI Intrusion Alert

ROI Intrusion Alert, belirlenen ROI bölgesine ilk kez giren her benzersiz
`track_id` için alarm üretir. Aynı track ID için tekrar alarm verilmez; ancak
nesne ROI dışına çıktığında `exit` olayı CSV dosyasına yazılır.

Varsayılan ROI frame merkezindeki daha dar bir dikdörtgen alandır. Kullanıcı
`--roi x1,y1,x2,y2` verirse bu değerler aynen kullanılır. Alarm anında video
üzerinde kırmızı arka planlı büyük alert banner gösterilir ve bu mesaj
varsayılan olarak 60 frame boyunca ekranda kalır. Süre
`--alert-display-frames` ile değiştirilebilir.

Alert snapshot dosyası yalnızca ilk `enter` olayı için kaydedilir. Snapshot
kaydı varsayılan olarak açıktır; kapatmak için `--save-snapshots false`
kullanılabilir. Video başladığında zaten ROI içinde olan nesneleri alarm
dışında bırakmak için `--ignore-initial-inside` eklenebilir. İsteğe bağlı olarak
`--play-sound` argümanı ile kısa sesli uyarı da verilebilir.

Intrusion pipeline:

```text
src/roi_intrusion_alert.py
```

Çıktılar:

```text
outputs/alerts/
outputs/logs/intrusion_events.csv
outputs/videos/<video_adi>_intrusion.mp4
```

CSV kolonları:

```text
frame,track_id,class,event,confidence,center_x,center_y,roi_name,snapshot_path
```

Örnek kullanım:

```powershell
python src/roi_intrusion_alert.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --roi 500,300,1600,900 --roi-name "Restricted Zone"
```

Demo komutu:

```powershell
python src/roi_intrusion_alert.py --source data/sample_videos/test5.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --roi-name "Intrusion Zone" --ignore-initial-inside
```

Sesli alarm:

```powershell
python src/roi_intrusion_alert.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --roi 500,300,1600,900 --roi-name "Restricted Zone" --play-sound
```

## Pedestrian Zone Intrusion

Pedestrian Zone Intrusion, belirlenen `Restricted Area` ROI içine giren yalnızca
`Person` sınıfı tracklerini izler ve kişi bölgeye ilk girdiğinde alarm üretir.
Aynı kişi ROI içinde kaldığı sürece tekrar alarm verilmez; kişi çıkıp tekrar
girerse yeni `enter` olayı oluşturulur.

Video üzerinde ROI `Restricted Area` etiketiyle çizilir. ROI içine giren kişiler
kırmızı kutu ile vurgulanır ve alarm anında üstte kırmızı uyarı bandı gösterilir:

```text
ALERT:
Person entered restricted zone
Person ID 12
```

Bisikletli kişiler bazı sahnelerde model tarafından `Person` olarak
algılanabilir. Bu tür yanlış alarmları azaltmak için opsiyonel filtreler
kullanılabilir:

- `--ignore-fast-person`
- `--max-person-speed`
- `--max-person-box-area-ratio`
- `--min-person-box-aspect`
- `--max-person-box-aspect`

Bu filtreler yalnızca alarm üretimini engeller; tespit ve takip akışı devam
eder. Track ID kopması nedeniyle aynı kişiye yakın konumda kısa süre içinde yeni
ID atanırsa `--reentry-cooldown-frames` ve `--duplicate-distance-threshold`
parametreleri duplicate alert üretimini azaltır.

ROI alanı sahneye göre daraltılmalıdır. Yaya geçidi, kaldırım veya normal yaya
akışının olduğu bölgeler `Restricted Area` yapılırsa normal yayalar da intrusion
olarak sayılır.

Pedestrian intrusion pipeline:

```text
src/pedestrian_zone_intrusion.py
```

Çıktılar:

```text
outputs/logs/pedestrian_intrusion_events.csv
outputs/alerts/
outputs/videos/<video_adi>_pedestrian_intrusion.mp4
```

CSV kolonları:

```text
frame,track_id,center_x,center_y,event,snapshot_path,filtered_reason
```

`filtered_reason` değeri `none`, `fast_person`, `large_box`, `aspect_ratio` veya
`duplicate_alert` olabilir. Sadece `none` olan girişler gerçek alarm, snapshot ve
kırmızı alert banner üretir.

Örnek kullanım:

```powershell
python src/pedestrian_zone_intrusion.py --source data/sample_videos/test5.mp4 --model models/yolo11s_2class_960_best.pt
```

Özel ROI:

```powershell
python src/pedestrian_zone_intrusion.py --source data/sample_videos/test5.mp4 --model models/yolo11s_2class_960_best.pt --roi 500,300,1600,900
```

Bisikletli/person karışmasını ve ID switch tekrar alarmlarını azaltan örnek:

```powershell
python src/pedestrian_zone_intrusion.py --source data/sample_videos/test7.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --ignore-fast-person --max-person-speed 180 --reentry-cooldown-frames 120
```

## Crowd Detection

Crowd Detection, belirlenen ROI içindeki kişi yoğunluğunu takip eder. Sistem
yalnızca `Person` sınıfını analiz eder; `Vehicle` tespitleri bu modda
kullanılmaz. ROI içindeki aktif kişi sayısı eşiklere göre `normal`, `warning`
veya `crowd_alert` durumuna çevrilir.

Bu özellik güvenlik, kalabalık yönetimi, kampüs/etkinlik izleme ve İHA tabanlı
gözetleme senaryoları için kullanılabilir. Warning durumunda sarı uyarı,
crowd alert durumunda kırmızı alarm gösterilir. Aynı durum için sürekli alarm
spam oluşmaması amacıyla `--cooldown-frames` kullanılır.

Varsayılan sayım modu `--count-mode active_tracks` değeridir. Böylece crowd
durumu video boyunca biriken benzersiz ID sayısına göre değil, o anda ROI içinde
görünen ve en az `--min-track-age` kadar stabilize olmuş aktif tracklere göre
hesaplanır. Bu, yoğun kalabalıklarda ByteTrack ID kopmalarından kaynaklanan
şişirilmiş sayımları azaltır. Varsayılan `--min-track-age` değeri `15` frame'dir.

Video okunabilirliğini artırmak için kişi ID etiketleri varsayılan olarak
gizlidir. Sadece kutular ve isteğe bağlı track çizgileri gösterilir. ID
etiketlerini görmek için `--show-person-ids` kullanılabilir. ByteTrack kayıp
track tamponu `--max-lost-frames` ile ayarlanabilir.

Yoğun kalabalık videoları için varsayılan eşikler daha yüksek tutulmuştur:
`--warning-threshold 25`, `--crowd-threshold 40`. Bu değerler sahneye, kamera
açısına ve ROI boyutuna göre ayarlanmalıdır.

Panelde şu dashboard metrikleri gösterilir:

```text
Current Persons
Peak Persons
Average Persons
Density Level: LOW / MEDIUM / HIGH
```

Alert banner yalnızca durum yükseldiğinde gösterilir:

- `NORMAL -> WARNING`
- `WARNING -> CROWD ALERT`
- `NORMAL -> CROWD ALERT`

Crowd detection pipeline:

```text
src/crowd_detection.py
```

Çıktılar:

```text
outputs/videos/<video_adi>_crowd_detection.mp4
outputs/logs/crowd_detection_events.csv
outputs/logs/<video_adi>_crowd_summary.json
outputs/alerts/<video_adi>_frame000123_crowd_alert.jpg
```

CSV kolonları:

```text
frame,time_sec,active_persons_in_roi,unique_persons_in_roi,status,event,snapshot_path
```

Özet JSON dosyasında ayrıca `peak_persons`, `average_persons`,
`peak_persons_in_roi`, `average_persons_in_roi`, `density_level_peak`,
`stable_track_min_age`, `crowd_duration_sec` ve `warning_duration_sec`
metrikleri bulunur.

Varsayılan ROI:

```powershell
python src/crowd_detection.py --source data/sample_videos/test8.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960
```

Özel ROI:

```powershell
python src/crowd_detection.py --source data/sample_videos/test8.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960 --roi 500,300,1600,900 --roi-name "Crosswalk Zone"
```

Daha hassas eşik:

```powershell
python src/crowd_detection.py --source data/sample_videos/test8.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960 --warning-threshold 15 --crowd-threshold 25
```

ID etiketleriyle debug görünümü:

```powershell
python src/crowd_detection.py --source data/sample_videos/test8.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960 --show-person-ids
```

## Wrong Way Detection

Wrong Way Detection, beklenen trafik yönünü kullanıcıdan alır ve ByteTrack ile
takip edilen `Vehicle` nesnelerinin bu yönün tam tersine hareket edip etmediğini
tespit eder. Person sınıfı `--show-person` ile ekranda çizilebilir; ancak
wrong-way alarmı varsayılan olarak yalnızca vehicle sınıfı için üretilir.

Sistem her `track_id` için merkez noktası geçmişi tutar. Track en az
`--min-track-frames` kadar gözlemlendikten sonra ilk ve son merkez arasındaki
yer değiştirme hesaplanır. Hareket `--direction-threshold` altında kalıyorsa
`stable` kabul edilir ve alarm üretilmez. Beklenen yönün yalnızca tam tersi yön
wrong-way olarak işaretlenir; yan yönler varsayılan olarak ihlal sayılmaz.

Wrong-way event oluştuğunda video üzerinde kırmızı kutu, büyük alert banner,
snapshot ve CSV log üretilir. Aynı `track_id` için yalnızca bir kez alert
oluşturulur. Bu özellik güvenlik, trafik denetimi ve İHA tabanlı yol izleme
senaryoları için kullanılabilir.

Wrong-way pipeline:

```text
src/wrong_way_detection.py
```

Çıktılar:

```text
outputs/alerts/<video_adi>_frame000123_vehicle_id12_wrong_way.jpg
outputs/logs/wrong_way_events.csv
outputs/videos/<video_adi>_wrong_way.mp4
```

CSV kolonları:

```text
frame,track_id,class,confidence,direction,expected_direction,event,center_x,center_y,snapshot_path
```

Sağa doğru trafik beklenen yol:

```powershell
python src/wrong_way_detection.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --expected-direction right
```

Sola doğru trafik beklenen yol:

```powershell
python src/wrong_way_detection.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --expected-direction left
```

Yukarı doğru trafik beklenen yol:

```powershell
python src/wrong_way_detection.py --source data/sample_videos/test5.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --expected-direction up
```

## Speed Violation Alert

Speed Violation Alert, ByteTrack ile takip edilen `Vehicle` nesnelerinin piksel
tabanlı hızını hesaplar ve belirlenen `px/s` limitini aşan araçlar için alarm
üretir. Bu sistem gerçek km/h hesabı yapmaz; hız değeri video üzerindeki merkez
noktalarının piksel/saniye hareketinden türetilen göreli bir ölçümdür.

Track en az `--min-track-frames` kadar gözlemlendikten sonra
`speed_px_per_sec` hesaplanır. Yanlış pozitifleri azaltmak için hız değerleri
rolling median ile yumuşatılır ve karar aşamasında
`smoothed_speed_px_per_sec` kullanılır. Bir aracın alarm üretmesi için
yumuşatılmış hızın en az `--violation-frames` ardışık değerlendirmede
`--speed-limit` üstünde kalması gerekir.

Video başlangıcında sahnede zaten bulunan araçların toplu alarm üretmesini
azaltmak için `--startup-grace-frames` kullanılır. Varsayılan olarak aynı
`track_id` için yalnızca bir kez alarm üretilir; bu davranış
`--no-one-alert-per-track` ile kapatılabilir. `--cooldown-frames` parametresi
tekil alarm modu kapatıldığında aynı track için tekrar alarm sıklığını sınırlar.

Başlangıçta zaten sahnede bulunan araçlar varsayılan olarak alarm dışı bırakılır.
`--ignore-initial-tracks` davranışı, ilk `--startup-grace-frames` boyunca görülen
track ID değerlerini başlangıç track'i olarak işaretler ve bu araçlar için speed
violation alarmı, kırmızı kutu, snapshot ve alert banner üretmez. Bu özellik
video başında oluşan toplu yanlış alarmları azaltır. Başlangıçtaki araçların da
değerlendirilmesi istenirse `--disable-ignore-initial-tracks` kullanılabilir.

Varsayılan hız limiti `360 px/s` olarak seçilmiştir. Bu değer video
çözünürlüğü, FPS, kamera açısı, drone irtifası ve sahne ölçeğine bağlıdır.
Farklı videolarda uygun limit yeniden kalibre edilmelidir.

İhlal yapan araçlar kırmızı kutu ile gösterilir, üstte alert banner görünür,
confirmed violation oluştuğunda snapshot kaydedilir ve CSV log oluşturulur.

Speed violation pipeline:

```text
src/speed_violation_alert.py
```

Çıktılar:

```text
outputs/alerts/<video_adi>_frame000542_vehicle_id12_speed_violation.jpg
outputs/logs/speed_violations.csv
outputs/videos/<video_adi>_speed_violation.mp4
```

CSV kolonları:

```text
frame,track_id,class,speed_px_per_sec,smoothed_speed_px_per_sec,speed_limit,direction,event,center_x,center_y,snapshot_path
```

Önerilen 4K otoyol demo komutu:

```powershell
python src/speed_violation_alert.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-limit 360 --min-track-frames 15 --speed-window 5 --violation-frames 3 --startup-grace-frames 60 --one-alert-per-track
```

Başlangıçtaki araçları da değerlendirmek için:

```powershell
python src/speed_violation_alert.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-limit 360 --min-track-frames 15 --speed-window 5 --violation-frames 3 --startup-grace-frames 60 --one-alert-per-track --disable-ignore-initial-tracks
```

## Demo Video

Final demo videosu olarak yoğun otoyol trafiği içeren yüksek çözünürlüklü
`test3.mp4` kullanılmıştır. Bu video tracking, line crossing ve active count
özelliklerini birlikte göstermek amacıyla seçilmiştir. Demo sırasında araçlar
`Vehicle` sınıfı olarak tespit edilir, ByteTrack ile takip edilir ve dikey
sanal çizgiyi geçiş yönlerine göre sayılır.

Demo komutu:

```powershell
python src/track_video.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --speed-threshold 2 --line-orientation vertical --line-position 0.45 --line-thickness 2
```

## Traffic Flow Analysis

Traffic Flow Analysis, final YOLO11s 2-Class modeli ve ByteTrack takip
çıktılarını kullanarak video sonunda yapısal trafik raporları üretir. Sistem
`Vehicle` ve `Person` tracklerini analiz eder; her track için yön, ortalama hız,
maksimum hız, ilk/son frame ve süre bilgilerini çıkarır.

Bu bölümdeki hız değerleri gerçek km/h değildir. Track merkez noktalarının
video üzerindeki piksel/saniye hareketinden hesaplanan göreli `px/s`
değerleridir.

Bu analiz video üzerine çizim yapmaz; amaç demo videosundan sonra rapor,
zaman çizelgesi ve grafik üretmektir. Zaman bazlı trafik yoğunluğu
`--timeline-window` ile belirlenen saniyelik aralıklara göre hesaplanır ve en
yoğun zaman aralığı `peak_traffic` olarak JSON özetine yazılır.

Traffic flow pipeline:

```text
src/traffic_flow_analysis.py
```

Üretilen çıktılar:

```text
outputs/reports/<video_adi>_flow_summary.json
outputs/reports/<video_adi>_flow_tracks.csv
outputs/reports/<video_adi>_flow_timeline.csv
outputs/reports/<video_adi>_flow_timeline.png
outputs/reports/<video_adi>_flow_directions.png
```

Örnek kullanım:

```powershell
python src/traffic_flow_analysis.py --source video.mp4
```

Final model ile örnek kullanım:

```powershell
python src/traffic_flow_analysis.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --timeline-window 5
```

## Traffic Heatmap

Traffic Heatmap özelliği, video boyunca tespit edilen nesnelerin merkez
noktalarını bir yoğunluk matrisine işler. Böylece person veya vehicle
nesnelerinin sahnede en yoğun geçtiği bölgeler görselleştirilebilir.

Kırmızı bölgeler yüksek yoğunluğu, mavi bölgeler düşük yoğunluğu temsil eder.
Bu özellik traffic flow analizi, surveillance senaryoları ve UAV analytics
çalışmaları için kullanılabilir.

İki farklı heatmap modu bulunur:

**Density heatmap**, nesnelerin video boyunca geçtiği yolları gösterir.
"Araçlar nerelerden geçti?" sorusunu cevaplar ve trafik akış güzergahlarını
anlamak için uygundur.

**Occupancy heatmap**, nesnelerin daha uzun süre bulunduğu veya yavaşladığı
bölgeleri vurgular. ByteTrack ile track geçmişi kullanılarak yaklaşık hareket
miktarı hesaplanır; yavaşlayan veya duran nesneler heatmap üzerinde daha yüksek
ağırlık bırakır. Trafik ışığı, kavşak merkezi, bekleme alanı ve yoğunluk analizi
için daha uygundur.

Heatmap pipeline:

```text
src/generate_heatmap.py
```

Üretilen çıktılar:

```text
outputs/heatmaps/<video_adi>_density_heatmap.png
outputs/heatmaps/<video_adi>_density_overlay.png
outputs/heatmaps/<video_adi>_occupancy_heatmap.png
outputs/heatmaps/<video_adi>_occupancy_overlay.png
outputs/logs/<video_adi>_heatmap_points.csv
```

CSV kolonları:

```text
frame,class,confidence,center_x,center_y,mode,weight
```

Density vehicle heatmap:

```powershell
python src/generate_heatmap.py --source data/sample_videos/test4.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --class-filter vehicle --mode density --alpha 0.45 --blur-radius 35 --point-radius 8
```

Occupancy vehicle heatmap:

```powershell
python src/generate_heatmap.py --source data/sample_videos/test4.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --class-filter vehicle --mode occupancy --alpha 0.45 --blur-radius 35 --point-radius 8
```

Person heatmap:

```powershell
python src/generate_heatmap.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.35 --imgsz 960 --class-filter person --mode density --alpha 0.45
```

All classes:

```powershell
python src/generate_heatmap.py --source data/sample_videos/test3.mp4 --model models/yolo11s_2class_960_best.pt --conf 0.40 --imgsz 960 --class-filter all --mode density --alpha 0.45
```

`--sample-rate` parametresiyle her kaç frame'de bir analiz yapılacağı
ayarlanabilir. Örneğin `--sample-rate 5`, her 5 frame'den birini işler ve uzun
videolarda daha hızlı özet heatmap üretir. `--blur-radius` son heatmap
yumuşatma miktarını, `--point-radius` her merkez noktasının etki alanını,
`--max-weight` ise occupancy modunda tek noktanın bırakabileceği maksimum
ağırlığı kontrol eder.

## Dashboard Report

Final dashboard modülü, proje içinde üretilen trafik ve güvenlik analitiği
çıktılarını tek bir görsel raporda toplar. Traffic Flow Analysis JSON/CSV
raporları, speed violation logları, wrong-way olayları, ROI intrusion kayıtları,
pedestrian intrusion kayıtları, crowd detection summary dosyaları ve varsa
heatmap görselleri referans olarak kullanılır.

Dashboard pipeline:

```text
src/generate_project_dashboard.py
```

Varsayılan kullanım:

```powershell
python src/generate_project_dashboard.py
```

Özel klasörlerle kullanım:

```powershell
python src/generate_project_dashboard.py --reports-dir outputs/reports --logs-dir outputs/logs --heatmaps-dir outputs/heatmaps --output-dir outputs/dashboard
```

Üretilen çıktılar:

```text
outputs/dashboard/project_dashboard.png
outputs/dashboard/project_summary.json
```

Dashboard PNG içinde şu metrikler ve grafikler yer alır:

- Total Vehicles
- Total Persons
- Peak Active Vehicles
- Average Vehicle Speed px/s
- Speed Violations
- Wrong Way Events
- ROI Intrusions
- Pedestrian Intrusions
- Peak Crowd Density
- Crowd Alert Events
- Vehicle direction distribution
- Traffic flow timeline
- Event counts bar chart
- Crowd/person density summary

Eksik log veya rapor dosyaları hata oluşturmaz; ilgili metrikler `N/A` olarak
gösterilir. Böylece dashboard, yalnızca mevcut analiz çıktılarıyla da
üretilebilir.

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
│   ├── alerts/
│   ├── dashboard/
│   ├── heatmaps/
│   ├── logs/
│   ├── reports/
│   ├── videos/
│   ├── v8n_4class_50/
│   ├── v8s_4class_50/
│   ├── yolo11n_4class_960/
│   └── yolo11s_4class_960/
├── src/
│   ├── crowd_detection.py
│   ├── convert_visdrone_2class.py
│   ├── convert_visdrone_4class.py
│   ├── generate_project_dashboard.py
│   ├── generate_heatmap.py
│   ├── inference_video.py
│   ├── pedestrian_zone_intrusion.py
│   ├── roi_intrusion_alert.py
│   ├── roi_zone_counter.py
│   ├── speed_violation_alert.py
│   ├── track_video.py
│   ├── traffic_flow_analysis.py
│   └── wrong_way_detection.py
├── data_2class.yaml
├── data_4class.yaml
├── requirements.txt
└── README.md
```

`outputs/alerts/`, `outputs/dashboard/`, `outputs/heatmaps/`, `outputs/logs/`,
`outputs/reports/` ve `outputs/videos/` klasörleri çalışma sırasında otomatik
oluşturulur. Runtime çıktı dosyaları GitHub deposuna dahil edilmez; eğitim
sonuç görselleri ise benchmark ve model karşılaştırması amacıyla repoda tutulur.

## Important Notes

- Hız değerleri gerçek km/h değildir; piksel tabanlı `px/s` değerleridir.
- Sonuçlar kamera açısı, video çözünürlüğü, perspektif, irtifa ve sahne ölçeğine bağlıdır.
- Bu proje gerçek zamanlı İHA analitiği için geliştirilmiş modüler bir prototiptir.
- Güvenlik ve trafik alarm modülleri karar destek amaçlıdır; kritik kullanımda saha kalibrasyonu gerekir.

## Final Project Status

Bu proje; detection, tracking, traffic analytics, security alerts, crowd
monitoring ve report generation özelliklerini içeren modüler bir UAV analytics
sistemi haline getirilmiştir. Her modül bağımsız script olarak çalışır, ortak
YOLO11s 2-Class final modelini kullanır ve sonuçlarını CSV, JSON, PNG veya MP4
formatında dışa aktarır.

## Gelecek Çalışmalar

- ByteTrack parametrelerinin farklı sahneler için optimize edilmesi
- Yeniden kimliklendirme ile uzun süreli nesne ID takibi
- Kamera hareketi telafisi
- Piksel hızını gerçek dünya hızına çevirmek için kamera kalibrasyonu
- Nesne yörüngelerinin kaydedilmesi
- ROI (Region of Interest) sayımı
- Trafik yoğunluğu heatmap üretimi
- Çoklu ROI (Region of Interest) bölgeleri
- ROI bazlı heatmap analizi
- Çoklu line crossing bölgeleri
- Canlı kamera ve RTSP akış desteği
- Streamlit tabanlı web dashboard
- SAHI ile küçük nesne tespiti optimizasyonu

## Sonuç

Şu ana kadarki en başarılı model **YOLO11s 2-Class** modelidir:

- mAP50 = 0.710
- mAP50-95 = 0.407

Bu model proje içerisinde nesne tespiti, ByteTrack tabanlı takip, aktif nesne
sayımı, yön analizi, piksel tabanlı hız tahmini, trafik analitiği ve güvenlik
alarm modülleri için kullanılmaktadır.

Sistem artık nesne tespiti, çoklu nesne takibi, aktif nesne sayımı, yön
analizi, piksel tabanlı hız tahmini, line crossing, ROI analizi, intrusion
alert, speed violation, wrong-way detection, crowd monitoring, heatmap ve final
dashboard raporlaması sunmaktadır. Final demo yoğun trafik ve kalabalık
videoları üzerinde başarıyla test edilmiştir.
