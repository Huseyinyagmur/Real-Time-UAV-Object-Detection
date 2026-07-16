from core.yolo_inference import YOLOInference
from pathlib import Path
import time
from core.tracking_config import TrackingConfig
from tools.error_analysis import load_images
def measure_inference_time(image_path:Path,inference:YOLOInference)->float:

    start=time.perf_counter()
    inference.predict(image_path)
    end=time.perf_counter()
    times=end-start
    return times
def benchmark_dataset(image_paths:list[Path],inference:YOLOInference)->dict:
    inference_times=[]
    for image_path in image_paths:
        temp=measure_inference_time(image_path,inference)
        inference_times.append(temp)
    total_time=sum(inference_times)
    max_time=max(inference_times)
    min_time=min(inference_times)
    average_latency=total_time/len(inference_times)
    fps=len(inference_times)/total_time
    return{
        "total_time":total_time,
        "max_latency" :max_time,
        "min_latency ":min_time,
        "average_latency":average_latency,
        "fps":fps
    }

def main():
    dataset_path=Path("../dataset/yolo_2class")
    image_directory=dataset_path/"images"/"val"
    image_path=load_images(image_directory)
    config=TrackingConfig()
    inference=YOLOInference(
        model_path=config.model_path,
        confidence=config.confidence,
        image_size=config.image_size,
        class_ids=config.class_ids
    )
    metrics=benchmark_dataset(image_path,inference)
    print(metrics["fps"])
    print(metrics["average_latency"])

if __name__=="__main__":
    main()