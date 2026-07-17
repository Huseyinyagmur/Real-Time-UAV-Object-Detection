
from pathlib import Path
from core.tracking_config import  TrackingConfig
from core.yolo_inference import YOLOInference
from tools.error_analysis  import analyze_dataset,calculate_metrics
from tools.performance_benchmark import benchmark_dataset
def evaluate_model(model_path:Path,image_paths:list[Path])->dict:
    config=TrackingConfig()
    inference=YOLOInference(
        model_path=model_path,
        confidence=config.confidence,
        image_size=config.image_size,
        class_ids=config.class_ids
    )
    tp,fp,ce,fn=analyze_dataset(image_paths,inference)
    metrics=calculate_metrics(tp,fp,fn)
    benchmark=benchmark_dataset(image_paths,inference)
    return {
    "model": model_path.stem,
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1_score"],
    "fps": benchmark["fps"],
    "average_latency": benchmark["average_latency"],
    "classification_error": ce
    }
