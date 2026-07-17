
from pathlib import Path
from core.tracking_config import  TrackingConfig
from core.yolo_inference import YOLOInference
from tools.error_analysis  import analyze_dataset,calculate_metrics
from tools.performance_benchmark import benchmark_dataset
from tools.error_analysis import load_images
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
def compare_models(model_paths:list[Path],image_paths:list[Path])->list[dict]:
    results=[]
    for model_path in model_paths:
        result=evaluate_model(model_path,image_paths)
        results.append(result)

    return results

def print_model_comparison(results:list[dict])->None:
    print("\n========== MODEL COMPARISON ==========")
    print(
        f"{'Model':<30}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
        f"{'FPS':<10}"
        f"{'Latency(ms)':<15}"
        f"{'CE':<6}"
    )
    print("-" * 79)

    for result in results:
        print(
            f"{result['model']:<30}"
            f"{result['precision'] * 100:<12.2f}"
            f"{result['recall'] * 100:<12.2f}"
            f"{result['f1_score'] * 100:<12.2f}"
            f"{result['fps']:<10.2f}"
            f"{result['average_latency'] * 1000:<15.2f}"
            f"{result['classification_error']:<6}"
        )

def main():
    dataset_path=Path("../dataset/yolo_2class/images/val")
    image_paths=load_images(dataset_path)

    model_paths=[
        Path("../models/yolo11s_2class_960_best.pt"),
        Path("../models/yolo11s_4class_960_best.pt"),
        Path("../models/yolo11n_4class_960_best.pt")
    ]
    results=compare_models(model_paths,image_paths)
    print_model_comparison(results)



if __name__=="__main__":
    main()