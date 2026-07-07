import os
import random
import csv

BASE = "/Users/aliffandy/Documents/PukulEnam/Colonomind Training Resource/revision_inconsistent/patient_splits"
os.makedirs(BASE, exist_ok=True)

random.seed(42) # For reproducibility

def generate_split(dataset_name, prefix, total, test_count):
    # Generate all IDs
    all_ids = [f"{prefix}_{str(i).zfill(3)}" for i in range(1, total + 1)]
    
    # Shuffle randomly
    random.shuffle(all_ids)
    
    # Split
    test_ids = all_ids[:test_count]
    train_ids = all_ids[test_count:]
    
    # Write to CSV
    filename = os.path.join(BASE, f"{dataset_name}_official_patient_split.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient_ID', 'Split'])
        for pid in train_ids:
            writer.writerow([pid, 'Train/Validation'])
        for pid in test_ids:
            writer.writerow([pid, 'Test'])
            
    # Also write simple TXT files for easy copy-pasting if needed
    with open(os.path.join(BASE, f"{dataset_name}_train_val.txt"), 'w') as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(BASE, f"{dataset_name}_test.txt"), 'w') as f:
        f.write("\n".join(test_ids))
        
    print(f"Generated splits for {dataset_name}:")
    print(f"  Total: {total}")
    print(f"  Train/Val: {len(train_ids)}")
    print(f"  Test: {len(test_ids)}")
    print(f"  Saved to: {filename}\n")

# LIMUC: Total 564, Test 112
generate_split("LIMUC", "LIMUC_P", 564, 112)

# TMC-UCM: Total 308, Test 61
generate_split("TMC-UCM", "TMC_P", 308, 61)

print("All split files generated successfully!")
