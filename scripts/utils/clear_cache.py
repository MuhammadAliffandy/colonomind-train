import os
import shutil

cache_dir = '/home/D13K48009/raid/Clara/new_drive/Dataset_Cache'
if os.path.exists(cache_dir):
    print(f"Clearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    print("Cache cleared successfully.")
else:
    print(f"Cache directory not found: {cache_dir}")
