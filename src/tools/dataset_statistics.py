
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
def calculate_ratios(person_count:int,vehicle_count:int)->tuple[float,float]:
    total_objects=person_count+vehicle_count
    if(total_objects==0):
        return 0.0,0.0
    person_ratio=person_count/total_objects*100
    vehicle_ratio=vehicle_count/total_objects*100
    return person_ratio,vehicle_ratio
def main():
    dataset_path = Path("dataset/yolo_2class")
    train_images_path=dataset_path /"images"/"train"
    valid_images_path=dataset_path /"images"/"val"
    test_images_path=dataset_path /"images"/"test"
    valid_image_count=count_images(valid_images_path)
    train_image_count=count_images(train_images_path)
    test_image_count=count_images(test_images_path)
    train_labels_path=dataset_path /"labels"/"train"
    valid_labels_path=dataset_path /"labels"/"val"
    test_labels_path=dataset_path /"labels"/"test"
    test_person_count,test_vehicle_count=count_objects(test_labels_path)
    valid_person_count,valid_vehicle_count=count_objects(valid_labels_path)
    train_person_count,train_vehicle_count=count_objects(train_labels_path)
    train_person_ratio,train_vehicle_ratio=calculate_ratios(train_person_count,train_vehicle_count)
    valid_person_ratio,valid_vehicle_ratio=calculate_ratios(valid_person_count,valid_vehicle_count)
    test_person_ratio,test_vehicle_ratio=calculate_ratios(test_person_count,test_vehicle_count)
    print(f"Train Images:{train_image_count},Valid Images:{valid_image_count},Test Images:{test_image_count}")
    print(f"Train Vehicle Count:{train_vehicle_count}, Train Person Count:{train_person_count}")
    print(f"Valid Vehicle Count:{valid_vehicle_count},Valid Person Count:{valid_person_count}")
    print(f"Test Vehicle Count:{test_vehicle_count},Test Person Count:{test_person_count}")
    print(f"Train Person Ratio:{train_person_ratio},Train Vehicle Ratio:{train_vehicle_ratio}")
    print(f"Valid Person Ratio:{valid_person_ratio},Valid Vehicle Ratio:{valid_vehicle_ratio}")
    print(f"Test Person Ratio:{test_person_ratio},Test Vehicle Ratio:{test_vehicle_ratio}")

if __name__=="__main__":
    main()