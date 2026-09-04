import os
from collections import Counter
from dgx_dataloader import load_all_images, load_tmc_ucm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def count_dataset(name, labels):
    counter = Counter(labels)
    total = len(labels)
    print(f"\n[{name}] Total Test Images: {total}")
    print(f"  - MES 0: {counter.get('MES0', 0)}")
    print(f"  - MES 1: {counter.get('MES1', 0)}")
    print(f"  - MES 2: {counter.get('MES2', 0)}")
    print(f"  - MES 3: {counter.get('MES3', 0)}")
    
def main():
    base_dir = ".." # Adjust if you run this from outside src/
    
    # 1. TMC-UCM (Test Split directly loaded)
    print("Loading TMC-UCM...")
    try:
        _, _, tmc_labels, _ = load_tmc_ucm(f'{base_dir}/Dataset/TMC-UCM', split_filter='Test')
        count_dataset("TMC-UCM", tmc_labels)
    except Exception as e:
        print(f"Error loading TMC-UCM: {e}")
        
    # 2. NTUH (20% Test Split)
    print("\nLoading NTUH...")
    try:
        ntuh_paths = [f'{base_dir}/Dataset+Code/MES classification_20250313', f'{base_dir}/Dataset+Code/MES classification_20250724']
        _, _, ntuh_labels_all, _ = load_all_images(ntuh_paths, 'NTUH')
        _, _, _, _, _, ntuh_test_labels, _, _ = train_test_split(
            range(len(ntuh_labels_all)), range(len(ntuh_labels_all)), ntuh_labels_all, range(len(ntuh_labels_all)), 
            test_size=0.2, random_state=42, stratify=ntuh_labels_all
        )
        count_dataset("NTUH", ntuh_test_labels)
    except Exception as e:
        print(f"Error loading NTUH: {e}")
        
    # 3. LIMUC
    print("\nLoading LIMUC...")
    try:
        limuc_paths = [f'{base_dir}/Dataset/LIMUC/test_set']
        _, _, limuc_test_labels, _ = load_all_images(limuc_paths, 'LIMUC')
        count_dataset("LIMUC", limuc_test_labels)
    except Exception as e:
        print(f"Error loading LIMUC: {e}")
        
if __name__ == "__main__":
    main()
