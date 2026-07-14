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


def main():
    dataset_path=Path("../dataset/yolo_2class/images/val")
    image_paths=load_images(dataset_path)
    # print(len(image_paths))
    config=TrackingConfig()
    inference=YOLOInference(model_path=config.model_path,
                            image_size=config.image_size,
                            confidence=config.confidence,
                            class_ids=config.class_ids)
    for image_path in image_paths:
        results=inference.predict(image_path)
        result=results[0]
        # print(type(result))
        # print(result)
        # print(dir(result))
        boxes=result.boxes
        # print(type(boxes))
        # print(boxes)
        # print(dir(boxes))
        predictions=extract_predictions(result)
        # print(predictions[0])
        # print(len(predictions))
        label_path=Path("../dataset/yolo_2class/labels/val")/ f"{image_path.stem}.txt"
        ground_truth = load_ground_truth(label_path)
        print(ground_truth[0])
        print(len(ground_truth))
        break

if __name__=="__main__":
    main()