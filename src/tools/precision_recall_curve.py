from pathlib import Path
from core.yolo_inference import YOLOInference
from core.tracking_config import TrackingConfig
from tools.error_analysis import analyze_dataset,calculate_metrics
import numpy as np
def calculate_precision_recall_curve(inference:YOLOInference,image_paths:list[dict],start_confidence:float =0.20,end_confidence:float=0.90,step:float =0.02)->list[tuple[float, float, float]]:
    pr_points=[]

    for confidence in np.arange(start_confidence,end_confidence+step,step):
        tp,fp,ce,fn,matched=analyze_dataset(image_paths,inference,confidence)
        metrics=calculate_metrics(tp,fp,fn)
        pr_points.append((confidence,metrics["precision"],metrics["recall"]))
    return pr_points

def main():
    config=TrackingConfig()
    inference=YOLOInference()

if __name__=="__main__":
    main()