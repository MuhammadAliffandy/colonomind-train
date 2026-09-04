import os
import argparse

def count_raw_files(base_path, class_folders):
    counts = {}
    total = 0
    for cls_folder in class_folders:
        dir_path = os.path.join(base_path, cls_folder)
        if os.path.exists(dir_path):
            # Only count actual image files (ignoring hidden files)
            files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.startswith('.')]
            counts[cls_folder] = len(files)
            total += len(files)
        else:
            counts[cls_folder] = 0
            print(f"  [Warning] Directory not found: {dir_path}")
    return counts, total

def count_txt_file(txt_path):
    counts = {'MES0': 0, 'MES1': 0, 'MES2': 0, 'MES3': 0}
    total = 0
    if not os.path.exists(txt_path):
        print(f"  [Warning] Text file not found: {txt_path}")
        return counts, total
        
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip().upper()
            if not line: continue
            if 'MES0' in line: counts['MES0'] += 1
            elif 'MES1' in line: counts['MES1'] += 1
            elif 'MES2' in line: counts['MES2'] += 1
            elif 'MES3' in line: counts['MES3'] += 1
            total += 1
    return counts, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/raid/D13K48009/Clara/new_drive", help="Path to the new drive dataset folder")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    print("========================================")
    print("📊 FULL DATASET COUNTER (ALL SPLITS)")
    print("========================================\n")
    
    # 1. TMC-UCM (Parse from txt files)
    print("[1] TMC-UCM")
    tmc_splits = ['train.txt', 'test.txt']
    tmc_total_all = 0
    tmc_combined = {'MES0': 0, 'MES1': 0, 'MES2': 0, 'MES3': 0}
    tmc_base = os.path.join(base_dir, 'Dataset', 'TMC-UCM')
    
    for split in tmc_splits:
        txt_path = os.path.join(tmc_base, split)
        counts, tot = count_txt_file(txt_path)
        tmc_total_all += tot
        for k in counts: tmc_combined[k] += counts[k]
        print(f"  - {split}: {tot} images")
    print(f"  => TOTAL TMC-UCM: {tmc_total_all} images")
    for k, v in tmc_combined.items(): print(f"     {k}: {v}")
    
    # 2. NTUH
    print("\n[2] NTUH")
    ntuh_paths = [
        os.path.join(base_dir, 'Dataset+Code', 'MES classification_20250313'),
        os.path.join(base_dir, 'Dataset+Code', 'MES classification_20250724')
    ]
    ntuh_total_all = 0
    ntuh_combined = {'MES0': 0, 'MES1': 0, 'MES2': 0, 'MES3': 0}
    for p in ntuh_paths:
        counts, tot = count_raw_files(p, ['MES0', 'MES1', 'MES2', 'MES3'])
        ntuh_total_all += tot
        for k in counts: ntuh_combined[k] += counts[k]
        print(f"  - Folder {os.path.basename(p)}: {tot} images")
    print(f"  => TOTAL NTUH: {ntuh_total_all} images")
    for k, v in ntuh_combined.items(): print(f"     {k}: {v}")
    
    # 3. LIMUC
    print("\n[3] LIMUC")
    limuc_splits = ['train_and_validation_sets', 'test_set']
    limuc_total_all = 0
    limuc_combined = {'Mayo 0': 0, 'Mayo 1': 0, 'Mayo 2': 0, 'Mayo 3': 0}
    for split in limuc_splits:
        base = os.path.join(base_dir, 'Dataset', 'LIMUC', split)
        counts, tot = count_raw_files(base, ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'])
        limuc_total_all += tot
        for k in counts: limuc_combined[k] += counts[k]
        print(f"  - {split}: {tot} images")
    print(f"  => TOTAL LIMUC: {limuc_total_all} images")
    for k, v in limuc_combined.items(): print(f"     {k.replace('Mayo ', 'MES')}: {v}")

if __name__ == "__main__":
    main()
