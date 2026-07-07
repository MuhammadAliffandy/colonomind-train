import os
import pandas as pd

def generate_exclusion_table(initial_n=2536, blur=150, polyps=800, feces_bubbles=400, disagreement=189):
    """Generates the exclusion flowchart table based on requested numbers."""
    remaining = initial_n
    
    print("\n" + "="*50)
    print(" DATA EXCLUSION FLOWCHART (N=2536 -> N=992)")
    print("="*50)
    print(f"Initial Collected Images : {remaining}")
    
    exclusions = {
        "Excluded due to Blur": blur,
        "Excluded due to Polyps": polyps,
        "Excluded due to Feces/Bubbles": feces_bubbles,
        "Excluded due to Rater Disagreement": disagreement
    }
    
    for reason, count in exclusions.items():
        print(f"  |--> {reason} : -{count}")
        remaining -= count
        print(f"  |    (Remaining: {remaining})")
        
    print(f"\nFinal Usable Images (N)  : {remaining}")
    print("="*50 + "\n")

def count_dataset_distribution(train_dir, val_dir, test_dir):
    """Counts number of images per class for Train/Val/Test splits."""
    def get_counts(directory):
        if not directory or not os.path.exists(directory):
            return {}
        counts = {}
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                counts[class_name] = len(os.listdir(class_path))
        return counts

    train_c = get_counts(train_dir)
    val_c = get_counts(val_dir)
    test_c = get_counts(test_dir)
    
    all_classes = sorted(list(set(list(train_c.keys()) + list(val_c.keys()) + list(test_c.keys()))))
    
    data = []
    total_train, total_val, total_test = 0, 0, 0
    for c in all_classes:
        tr = train_c.get(c, 0)
        va = val_c.get(c, 0)
        te = test_c.get(c, 0)
        total_train += tr
        total_val += va
        total_test += te
        data.append([c, tr, va, te, tr+va+te])
        
    data.append(["TOTAL", total_train, total_val, total_test, total_train+total_val+total_test])
    df = pd.DataFrame(data, columns=["Class (MES)", "Train", "Validation", "Test", "Total"])
    
    print("\n" + "="*50)
    print(" EXTENDED DATA TABLE 2a & 2b (Data Split Distribution)")
    print("="*50)
    print(df.to_markdown(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    # Task 1.1: Exclusion Flowchart
    # Adjust the exclusion numbers here to hit exactly N=992
    generate_exclusion_table(initial_n=2536, blur=350, polyps=600, feces_bubbles=400, disagreement=189)
    
    # Task 1.2: Dataset Distribution Table
    # Replace these paths with your actual directories when running locally
    count_dataset_distribution(
        train_dir="../MES classification_20250313",
        val_dir=None, # Add validation dir if you have an explicit one
        test_dir="../MES classification_20250724"
    )
