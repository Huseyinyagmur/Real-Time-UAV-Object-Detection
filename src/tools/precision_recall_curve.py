from pathlib import Path
from core.yolo_inference import YOLOInference
from core.tracking_config import TrackingConfig
from tools.error_analysis import analyze_dataset,calculate_metrics,load_images
import matplotlib.pyplot as plt
import numpy as np
def calculate_precision_recall_curve(inference:YOLOInference,image_paths:list[dict],start_confidence:float =0.20,end_confidence:float=0.90,step:float =0.02)->list[tuple[float, float, float]]:
    pr_points=[]

    for confidence in np.arange(start_confidence,end_confidence+step,step):
        tp,fp,_,fn,_=analyze_dataset(image_paths,inference,confidence)
        metrics=calculate_metrics(tp,fp,fn)
        pr_points.append((confidence,metrics["precision"],metrics["recall"]))
    return pr_points
def plot_precision_recall_curve(pr_points:list[tuple[float,float,float]], title:str="Precision-Recall Curve",save_path=None):
    precision=[]
    recall=[]
    for _,pre,rec in pr_points:
        precision.append(pre)
        recall.append(rec)
    plt.figure(figsize=(8,6))
    plt.plot(recall,precision,linewidth=2,label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    plt.show()
    plt.close()
def main():
    config=TrackingConfig()
    dataset_path=Path("../dataset/yolo_2class/images/val")
    image_paths=load_images(dataset_path)
    inference=YOLOInference(
        model_path=config.model_path,
        image_size=config.image_size,
        confidence=config.confidence,
        class_ids=config.class_ids
    )
    pr_points=calculate_precision_recall_curve(inference,image_paths)
    plot_precision_recall_curve(pr_points)

if __name__=="__main__":
    main()