from core.yolo_inference import YOLOInference
from pathlib import Path
import time
from core.tracking_config import TrackingConfig
def measure_inference_time(image_path:Path,inference:YOLOInference)->float:

    start=time.perf_counter()
    inference.predict(image_path)
    end=time.perf_counter()
    times=end-start
    return times

def main():
    image_path=Path("../dataset/yolo_2class/images/val/0000001_02999_d_0000005.jpg")
    config=TrackingConfig()
    inference=YOLOInference(
        model_path=config.model_path,
        confidence=config.confidence,
        image_size=config.image_size,
        class_ids=config.class_ids
    )
    result=measure_inference_time(image_path,inference)
    print(result)

if __name__=="__main__":
    main()