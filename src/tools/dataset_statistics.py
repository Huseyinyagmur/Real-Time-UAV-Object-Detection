
from pathlib import Path

def count_images(image_directory: Path)->int:
    image_count=0
    for file in image_directory.iterdir():
        if file.suffix.lower() in {".jpg",".jpeg",".png"}:
            image_count += 1
    return image_count

def count_objects(label_directory:Path):
    person_count=0
    vehicle_count=0
    for label_file in label_directory.glob("*.txt"):
        with label_file.open("r",encoding="utf-8") as file:
            for line in file:
                parts=line.split()
                class_id=int(parts[0])
                if(class_id==0):
                    person_count+=1
                elif(class_id==1):
                    vehicle_count+=1
    return person_count,vehicle_count
def main():
    dataset_path = Path("dataset/yolo_2class")
    train_images_path=dataset_path /"images"/"train"
    valid_images_path=dataset_path /"images"/"val"
    test_images_path=dataset_path /"images"/"test"
    valid_image_count=count_images(valid_images_path)
    train_image_count=count_images(train_images_path)
    test_image_count=count_images(test_images_path)
    train_labels_path=dataset_path /"labels"/"train"
    person_count,vehicle_count=count_objects(train_labels_path)
    print(f"Train Images:{train_image_count},Valid Images:{valid_image_count},Test Images:{test_image_count}")
    print(f"Vehicle Count:{vehicle_count},Person Count:{person_count}")
if __name__=="__main__":
    main()