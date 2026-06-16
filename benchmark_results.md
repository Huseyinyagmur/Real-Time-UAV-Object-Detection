# Model Karşılaştırması

| Model                | Epoch | Giriş Boyutu | Precision | Recall | mAP50 | mAP50-95 |
| -------------------- | ----- | ------------ | --------- | ------ | ----- | -------- |
| YOLOv8n              | 50    | 640          | 0.634     | 0.433  | 0.470 | 0.279    |
| YOLOv8s              | 50    | 640          | 0.688     | 0.514  | 0.552 | 0.342    |
| YOLO11n              | 50    | 960          | 0.682     | 0.524  | 0.568 | 0.359    |
| YOLO11s (4-Class)    | 50    | 960          | 0.747     | 0.582  | 0.638 | 0.415    |
| YOLO11s (2-Class) ⭐ | 50    | 960          | 0.787     | 0.638  | 0.710 | 0.407    |

### Benchmark Değerlendirmesi

YOLOv8n, YOLOv8s, YOLO11n ve YOLO11s modelleri VisDrone2019 veri seti üzerinde karşılaştırılmıştır.

4 sınıflı (Person, Car, Truck, Bus) yapıdan 2 sınıflı (Person, Vehicle) yapıya geçilmesiyle birlikte sınıflar arası karışıklık azalmış ve genel tespit başarısı artmıştır.

En yüksek mAP50 sonucu YOLO11s (2-Class) modeli ile elde edilmiştir:

- Precision: 0.787
- Recall: 0.638
- mAP50: 0.710
- mAP50-95: 0.407

Vehicle sınıfında mAP50 = 0.818 elde edilmiştir. Person sınıfı ise drone görüntülerindeki küçük nesneler nedeniyle daha zor olmasına rağmen mAP50 = 0.602 seviyesine ulaşmıştır.

Bu nedenle proje kapsamında final model olarak YOLO11s (2-Class) seçilmiştir.
