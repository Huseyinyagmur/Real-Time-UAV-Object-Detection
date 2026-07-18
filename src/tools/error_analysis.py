import cv2
import csv
from core.yolo_inference import YOLOInference
from pathlib import Path
from core.tracking_config import TrackingConfig

def load_images(image_directory:Path)->list[Path]:
    image_paths=[]
    for file in image_directory.iterdir():
        if file.suffix.lower() in [".jpg",".png",".jpeg"]:
            image_paths.append(file)
    return image_paths

def extract_predictions(result):
    predictions=[]
    boxes=result.boxes
    for class_id,confidence,bbox in zip(boxes.cls,boxes.conf,boxes.xyxy):
        predictions.append(
            {
                "class_id":int(class_id.item()),
                "confidence":confidence.item(),
                "bbox":bbox.tolist()
            }
        )
    return predictions

def load_ground_truth(label_path: Path):
    labels=[]

    with label_path.open("r",encoding="utf-8") as file:
        for line in file:
            parts=line.split()
            class_id=int(parts[0])
            bbox=[float(value) for value in parts[1:]]
            labels.append({
                "class_id":class_id,
                "bbox":bbox
            })
    return labels

def yolo_to_xyxy(bbox:list[float],image_width:int,image_height:int)->list[float]:
    x_center,y_center,width,height=bbox
    #yolo koordinatlarını piksel koordinatlarına dönüştürüyor.
    x_center*=image_width
    y_center*=image_height
    width*=image_width
    height*=image_height
    #merkez koordinatlarını  köşe koordinatlarına çevirir
    x1=x_center-width/2
    x2=x_center+width/2
    y1=y_center-height/2
    y2=y_center+height/2
    return [x1,y1,x2,y2]

def calculate_iou(bbox1:list[float],bbox2:list[float])->float:
    x1_min,y1_min,x1_max,y1_max=bbox1
    x2_min,y2_min,x2_max,y2_max=bbox2
    intersection_left=max(x1_min,x2_min)
    intersection_right=min(x1_max,x2_max)
    intersection_top=max(y1_min,y2_min)
    intersection_bottom=min(y1_max,y2_max)
    intersection_width=max(0.0,intersection_right-intersection_left)
    intersection_height=max(0.0,intersection_bottom-intersection_top)
    intersection_area=intersection_width*intersection_height
    bbox1_area=(x1_max-x1_min)*(y1_max-y1_min)
    bbox2_area=(x2_max-x2_min)*(y2_max-y2_min)
    union_area=bbox1_area+bbox2_area-intersection_area
    if union_area==0:
        return 0.0
    return intersection_area/union_area

def find_best_match(prediction:dict,ground_truth:list[dict],matched_labels:list[dict])->tuple[float,dict|None]:
    best_iou=0.0
    best_label=None
    for label in ground_truth:
        if label in matched_labels:
            continue    
        iou=calculate_iou(
        prediction["bbox"],
            label["bbox"]
        )
        if iou > best_iou:
            best_iou=iou
            best_label=label
    return best_iou,best_label
def classify_prediction(
        prediction:dict,
        ground_truth:list[dict],
        matched_labels:list[dict],
        iou_threshold:float=0.5,
):
    best_iou,best_label=find_best_match(prediction,ground_truth,matched_labels)
    if best_iou<iou_threshold:
        return "False Positive",None
    else:
        if best_label is not None and best_label["class_id"] == prediction["class_id"]:
            return "True Positive", best_label
        return "Classification Error", best_label


def analyze_image(predictions:list[dict],ground_truth:list[dict])->tuple[int,int,int,int,list[tuple[int,int]]]:
    matched_labels=[]
    matches=[]
    true_positive=0
    false_positive=0
    classification_error=0
    false_negative=0
    for prediction in predictions:
        status,label=classify_prediction(prediction,ground_truth,matched_labels)
        if status=="True Positive":
            if label is not None:
                matched_labels.append(label)
                matches.append((label["class_id"],prediction["class_id"]))
            true_positive+=1
        elif status=="False Positive":
            false_positive+=1
        elif status=="Classification Error":
            if label is not None:
                matched_labels.append(label)
                matches.append((label["class_id"],prediction["class_id"]))
            classification_error+=1
    for label in ground_truth:
        if label not in matched_labels:
            false_negative+=1
    return true_positive,false_positive,classification_error,false_negative,matches
def save_error_image(
        image_path:Path,
        result,
        false_positive:int,
        false_negative:int

):
    if false_positive>0 or false_negative>0:

        annotated_image=result.plot()
        PROJECT_ROOT=Path(__file__).resolve().parents[2]
        output_directory=PROJECT_ROOT/"output"/"errors"
        save_path=output_directory / image_path.name

        output_directory.mkdir(
            parents=True,    #klasör yoksa oluştur varsa hata verme
            exist_ok=True
        )
        cv2.imwrite(
            str(save_path),
            annotated_image
        )

def write_csv_row(csv_writer,image_path:Path,true_positive:int,false_positive:int,classification_error:int,false_negative:int):
    csv_writer.writerow([
        image_path.name,
        true_positive,
        false_positive,
        classification_error,
        false_negative
        ]
    )
def load_csv_report(report_path:Path)->list[dict]:
    report=[]
    with report_path.open("r",encoding="utf-8") as file:
        csv_reader=csv.DictReader(file) #satır satır okur
        for row in csv_reader:
            report.append({
                "image_name":row["image_name"],
                "true_positive":int(row["true_positive"]),
                "false_positive":int(row["false_positive"]),
                "classification_error":int(row["classification_error"]),
                "false_negative":int(row["false_negative"])
            })
    return report

def find_worst_images(report:list[dict],metric:str,top_k:int=10)->list[dict]:
    sorted_report=sorted(
        report,
        key=lambda row:row[metric],
        reverse=True
    )
    return sorted_report[:top_k]

def print_worst_images(worst_images:list[dict],metric:str)->None:
    print(f"\n========== {metric.upper()} ==========")
    for index,image in enumerate(worst_images,start=1):
        print(f"{index}."  
                f"{image['image_name']}"
                f"{metric}={image[metric]}")

def calculate_metrics(true_positive:int,false_positive:int,false_negative:int)->dict:

    if true_positive+false_positive==0:
        precision=0.0
    else:
        precision=true_positive/(true_positive+false_positive)
    if true_positive+false_negative==0:
        recall=0.0
    else:
        recall=true_positive/(true_positive+false_negative)
    if precision+recall==0:
        f1_score=0.0
    else:
        f1_score=2*precision*recall/(precision+recall)
    return {
        "precision":precision,
        "recall":recall,    
        "f1_score":f1_score
    }
   
def analyze_dataset(image_paths:list[Path],inference:YOLOInference)->tuple[int,int,int,int]:
    total_true_positive=0
    total_false_positive=0
    total_false_negative=0
    total_classification_error=0
    project_root=Path(__file__).resolve().parents[2]
    output_directory=project_root/"output"
    output_directory.mkdir(parents=True,exist_ok=True)
    csv_path=output_directory/"report.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as file:
        csv_writer=csv.writer(file)
        csv_writer.writerow([
            "image_name",
            "true_positive",
            "false_positive",
            "classification_error",
            "false_negative"]
        )

        for image_path in image_paths:
            results=inference.predict(image_path)
            result=results[0]
            predictions=extract_predictions(result)
            label_path=Path("../dataset/yolo_2class/labels/val")/f"{image_path.stem}.txt"
            ground_truth=load_ground_truth(label_path)
            image_height,image_width=result.orig_shape
            for label in ground_truth:
                label["bbox"]=yolo_to_xyxy(
                    label["bbox"],
                    image_width,
                    image_height
                )
            true_positive,false_positive,classification_error,false_negative=analyze_image(predictions,ground_truth)
            write_csv_row(csv_writer,image_path,true_positive,false_positive,classification_error,false_negative,)
            save_error_image(image_path,result,false_positive,false_negative)
            total_true_positive+=true_positive
            total_false_negative+=false_negative
            total_false_positive+=false_positive
            total_classification_error+=classification_error
    return (total_true_positive,total_false_positive,total_classification_error,total_false_negative)

def analyze_class_metrics(predictions:list[dict],ground_truth:list[dict])->dict:
    matched_labels=[]
    class_metrics={
        0:{
            "tp":0,
            "fp":0,
            "fn":0
        },
        1:{
            "tp":0,
            "fp":0,
            "fn":0
        }
    }
    for prediction in predictions:
        status,label=classify_prediction(prediction,ground_truth,matched_labels)
        if status=="True Positive":
            class_id=prediction["class_id"]
            class_metrics[class_id]["tp"]+=1
            if label is not None:
                matched_labels.append(label)
        elif status=="False Positive":
            class_id=prediction["class_id"]
            class_metrics[class_id]["fp"]+=1
        elif status=="Classification Error":
            prediction_class=prediction["class_id"]
            class_metrics[prediction_class]["fp"]+=1
            matched_labels.append(label)
            label_class=label["class_id"]
            class_metrics[label_class]["fn"]+=1
    for label in ground_truth:
        if label not in matched_labels:
            class_id=label["class_id"]
            class_metrics[class_id]["fn"]+=1
    return class_metrics
def calculate_class_metrics(class_metrics:dict)->dict:
    metrics={}
    for class_id,values in class_metrics.items():
        tp=values["tp"]
        fp=values["fp"]
        fn=values["fn"]
        if tp+fp==0:
            precision=0.0
        else:
            precision=tp/(tp+fp)
        if tp+fn==0:
            recall=0.0
        else:
            recall=tp/(tp+fn)
        if precision+recall==0:
            f1_score=0.0
        else:
            f1_score=2*precision*recall/(precision+recall)
        metrics[class_id]={
            "precision":precision,
            "recall":recall,
            "f1_score":f1_score
        }
    return metrics
def analyze_dataset_class_metrics(image_paths:list[Path],inference:YOLOInference)->dict:
    total_metrics = {
        0: {"tp":0,"fp":0,"fn":0},
        1: {"tp":0,"fp":0,"fn":0}
    }

    for image_path in image_paths:
        results=inference.predict(image_path)
        result=results[0]
        predictions=extract_predictions(result)
        label_path=Path("../dataset/yolo_2class/labels/val")/f"{image_path.stem}.txt"
        ground_truth=load_ground_truth(label_path)
        image_width,image_height=result.orig_shape
        for label in ground_truth:
            label["bbox"]=yolo_to_xyxy(
                label["bbox"],
                image_width,
                image_height
            )
        image_metrics=analyze_class_metrics(predictions,ground_truth)
        for class_id in total_metrics:
            total_metrics[class_id]["tp"] += image_metrics[class_id]["tp"]
            total_metrics[class_id]["fp"] += image_metrics[class_id]["fp"]
            total_metrics[class_id]["fn"] += image_metrics[class_id]["fn"]
    return total_metrics
def print_class_metrics(metrics:dict)->None:
    print("\n========== PER-CLASS METRICS ==========")

    class_names = {
        0: "Person",
        1: "Vehicle"
    }

    for class_id,values in metrics.items():
        print(f"{class_names[class_id]}")
        print(f"Precision : {values['precision'] * 100:.2f}%")
        print(f"Recall    : {values['recall'] * 100:.2f}%")
        print(f"F1 Score  : {values['f1_score'] * 100:.2f}%")
        print("-" * 40)
def main():
    dataset_path=Path("../dataset/yolo_2class/images/val")
    image_paths=load_images(dataset_path)
    # print(len(image_paths))
    config=TrackingConfig()
    inference=YOLOInference(model_path=config.model_path,
                            image_size=config.image_size,
                            confidence=config.confidence,
                            class_ids=config.class_ids)
    tp,fp,ce,fn=analyze_dataset(
         image_paths,
         inference
     )
    class_counts=analyze_dataset_class_metrics(
        image_paths,
        inference
    )

    metrics = calculate_class_metrics(class_counts)
    print_class_metrics(metrics)

    # print("\n========== MODEL METRICS ==========")
    # print(f"Precision : {metrics['precision']:.2%}")
    # print(f"Recall    : {metrics['recall']:.2%}")
    # print(f"F1 Score  : {metrics['f1_score']:.2%}")
    # print(f"True Positive:{tp},False Positive:{fp},Classification Error:{ce},False Negative:{fn}")
    # project_root=Path(__file__).resolve().parents[2]
    # report=load_csv_report(
    #     project_root/"output"/"report.csv"
    # )
    # worst_fp=find_worst_images(
    #     report,"false_positive"
    # )
    # print_worst_images(
    #     worst_fp,
    #     "false_positive"
    # )
    # worst_fn=find_worst_images(
    #     report,"false_negative"
    # )
    # print_worst_images(worst_fn,"false_negative")
    # worst_ce=find_worst_images(
    #     report,"classification_error"
    # )
    # print_worst_images(worst_ce,"classification_error")
    # for image_path in image_paths:
    #     results=inference.predict(image_path)
    #     result=results[0]
        # print(type(result))
        # print(result)
        # print(dir(result))
        # boxes=result.boxes
        # print(type(boxes))
        # print(boxes)
        # print(dir(boxes))
        # predictions=extract_predictions(result)
        # print(predictions[0])
        # print(len(predictions))
        # label_path=Path("../dataset/yolo_2class/labels/val")/ f"{image_path.stem}.txt"
        # ground_truth = load_ground_truth(label_path)
        # print(ground_truth[0])
        # print(len(ground_truth))
        # image_height,image_width=result.orig_shape
        # for label in ground_truth:
        #     label["bbox"]=yolo_to_xyxy(
        #         label["bbox"],
        #         image_width,
        #         image_height
        #     )
        # print(ground_truth[0])
        # best_iou,best_label=find_best_match(predictions[0],ground_truth)
        # print(best_iou)
        # print(best_label)
        # temp=classify_prediction(predictions[0],ground_truth,0.5)
        # print(temp)
        # true_positive,false_positive,classification_error,false_negative=analyze_image(predictions,ground_truth)
        # print(f"true_positive={true_positive},false_positive={false_positive},classification_error={classification_error},false_negative={false_negative}")
        # break
        # annotated_image = result.plot()

        # print(type(annotated_image))

if __name__=="__main__":
    main()