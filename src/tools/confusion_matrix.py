import numpy as np

def build_confusion_matrix(matches,class_names):
    num_classes=len(class_names)
    confusion_matrix=np.zeros((num_classes,num_classes),dtype=int)

    for gt_class,pred_class in matches:
        confusion_matrix[gt_class][pred_class]+=1
    return confusion_matrix

def print_confusion_matrix(confusion_matrix, class_names):
    print("\n========== CONFUSION MATRIX ==========\n")

    print(f"{'GT\\Pred':<12}", end="")
    for class_name in class_names.values():
        print(f"{class_name:<10}", end="")
    print()

    for gt_class, class_name in class_names.items():
        print(f"{class_name:<12}", end="")

        for pred_class in range(len(class_names)):
            print(f"{confusion_matrix[gt_class][pred_class]:<10}", end="")

        print()
def main():


if __name__=="__main__":
    main()