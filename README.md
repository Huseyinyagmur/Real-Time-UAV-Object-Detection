# UAV Object Detection System

Drone/İHA görüntülerinden insan ve araç tespiti yapmayı amaçlayan gerçek
zamanlı görüntü işleme projesi.

Bu aşamada proje, VisDrone2019-DET veri setini Ultralytics YOLO biçimine
dönüştürür. Model eğitimi henüz yapılmaz.

## Desteklenen sınıflar

VisDrone sınıflarının tamamı korunur ve YOLO için `0-9` aralığına dönüştürülür:

| YOLO ID | Sınıf |
| ---: | --- |
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

VisDrone'daki `ignored regions` (`class_id=0`) ve `score=0` kayıtları YOLO
etiketlerine yazılmaz.

## Kurulum

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ham veri seti şu dizinlerde bulunmalıdır:

```text
dataset/raw/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    ├── images/
    └── annotations/
```

## Dönüştürme

Proje kök dizininden:

```powershell
python src/convert_visdrone_to_yolo.py
```

Yalnızca belirli bölümleri dönüştürmek için:

```powershell
python src/convert_visdrone_to_yolo.py --splits train val
```

Özel kaynak ve çıktı yolları da verilebilir:

```powershell
python src/convert_visdrone_to_yolo.py `
  --raw-dir D:\datasets\VisDrone `
  --output-dir D:\datasets\VisDrone-YOLO
```

Betik tekrar çalıştırılabilir. Aynı boyuttaki mevcut görüntüler yeniden
kopyalanmaz; etiketler güncel annotation içeriğinden yeniden oluşturulur.

## Üretilen yapı

```text
dataset/yolo/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Her YOLO etiketi şu normalize edilmiş biçimdedir:

```text
class_id x_center y_center width height
```

`dataset/yolo/dataset.yaml`, sonraki aşamada Ultralytics eğitim komutuna
verilecek veri seti yapılandırmasıdır.
