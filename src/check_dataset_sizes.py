import os

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

def main():
    base_dir = ".." # Adjust if needed
    
    print("========================================")
    print("📊 RAW DATASET COUNTER (BYPASSING CACHE)")
    print("========================================\n")
    
    # 1. TMC-UCM Test
    tmc_base = os.path.join(base_dir, 'Dataset', 'TMC-UCM', 'Test')
    tmc_counts, tmc_total = count_raw_files(tmc_base, ['MES0', 'MES1', 'MES2', 'MES3'])
    print(f"[TMC-UCM] Total Test Images: {tmc_total}")
    for k, v in tmc_counts.items(): print(f"  - {k}: {v}")
    
    # 2. NTUH (20% of Total)
    print("\n[NTUH] (Note: This is Total. Test split will be 20% of this)")
    ntuh_paths = [
        os.path.join(base_dir, 'Dataset+Code', 'MES classification_20250313'),
        os.path.join(base_dir, 'Dataset+Code', 'MES classification_20250724')
    ]
    ntuh_total = 0
    ntuh_combined_counts = {'MES0': 0, 'MES1': 0, 'MES2': 0, 'MES3': 0}
    for p in ntuh_paths:
        counts, tot = count_raw_files(p, ['MES0', 'MES1', 'MES2', 'MES3'])
        ntuh_total += tot
        for k in counts: ntuh_combined_counts[k] += counts[k]
    
    print(f"[NTUH] Total Images (All Splits): {ntuh_total}")
    for k, v in ntuh_combined_counts.items(): print(f"  - {k}: {v}")
    print(f"  --> Expected Test Size (20%): {int(ntuh_total * 0.2)}")
    
    # 3. LIMUC
    limuc_base = os.path.join(base_dir, 'Dataset', 'LIMUC', 'test_set')
    limuc_counts, limuc_total = count_raw_files(limuc_base, ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3'])
    print(f"\n[LIMUC] Total Test Images: {limuc_total}")
    for k, v in limuc_counts.items(): print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
