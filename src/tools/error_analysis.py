from core.yolo_inference import YOLOInference
from pathlib import Path
from core.tracking_config import TrackingConfig
def load_images(image_directory:Path)->list[Path]:
    image_paths=[]
    for file in image_directory.iterdir():
        if file.suffix.lower() in [".jpg",".png",".jpeg"]:
            image_paths.append(file)
    return image_paths
def main():
    dataset_path=Path("../dataset/yolo_2class/images/val")
    image_paths=load_images(dataset_path)
    print(len(image_paths))
    config=TrackingConfig()
    inference=YOLOInference(model_path=config.model_path,
                            image_size=config.image_size,
                            confidence=config.confidence,
                            class_ids=config.class_ids)
    for image_path in image_paths:
        results=inference.predict(image_path)
        print(type(results))
        print(len(results))
        break


if __name__=="__main__":
    main()