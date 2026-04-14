'''
More efficient implementation for combining h5 files
'''

JZ_path = '/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-gluon-split-label'
ttbar_path='/share/data1/xzcappon/datasets/ftag/super-stats/split-components/'

import glob
import os
import h5py
import numpy as np

# ---- Config ----
# The folder containing the "user.npond..." directories
base_dir = "/home/xzcappon/phd/projects/supervising/2025_2026/samples-with-gluon-split-label"
out_file = "/path/to/your/output/merged_large_dataset.h5"

# The specific datasets we want to merge based on your file dump
DATASET_KEYS = ["jets", "electrons", "flows", "tracks_ghost"]

# Tune to limit memory use
copy_block_rows = 5000 

print("Scanning for files...")

# 1. Find the directories first
search_pattern = os.path.join(base_dir, "user.npond*.h5")
candidate_dirs = sorted(glob.glob(search_pattern))

all_in_files = []

# 2. Go inside each directory and find the actual .h5 files
for d in candidate_dirs:
    if os.path.isdir(d):
        files_in_dir = glob.glob(os.path.join(d, "*.h5"))
        all_in_files.extend(files_in_dir)

# Sort for deterministic order
all_in_files = sorted(all_in_files)

if not all_in_files:
    raise FileNotFoundError("No .h5 files found! Check your base_dir path.")

print(f"Found {len(all_in_files)} files to merge.")

# ---- Pass 1: Compute total rows and define schema ----
print("Pass 1: Inspecting schema and calculating total size...")

# We'll store the shape info and total rows for each dataset
schema = {} 
total_rows = 0

# specific check on first file to get Dtypes and Shapes
with h5py.File(all_in_files[0], "r") as f0:
    for key in DATASET_KEYS:
        if key not in f0:
            raise KeyError(f"Expected dataset '{key}' not found in first file.")
        
        dset = f0[key]
        schema[key] = {
            "dtype": dset.dtype,
            "shape_tail": dset.shape[1:] # (50,) for flows, () for jets
        }

# Calculate total rows (assuming all datasets in a file have aligned row counts)
for fn in all_in_files:
    try:
        with h5py.File(fn, "r") as f:
            # We check the length of 'jets' and assume others align 
            # (standard in physics ntuples)
            total_rows += f["jets"].shape[0]
    except OSError:
        print(f"Skipping corrupt file: {fn}")

print(f"Total entries to merge: {total_rows}")

exit()

# ---- Pass 2: Create output and copy data ----
print("Pass 2: Merging data...")

with h5py.File(out_file, "w") as fout:
    
    # 1. Initialize all output datasets
    out_dsets = {}
    for key in DATASET_KEYS:
        # Final shape = (Total_Rows, ...tail_dims...)
        final_shape = (total_rows,) + schema[key]["shape_tail"]
        
        print(f"Creating '{key}' with shape {final_shape}")
        
        out_dsets[key] = fout.create_dataset(
            key,
            shape=final_shape,
            dtype=schema[key]["dtype"],
            chunks=True, # Critical for compression/speed
            compression="gzip" # Optional: saves space but slower write
        )

    # 2. Iterate and Copy
    global_offset = 0
    
    for i, fn in enumerate(all_in_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(all_in_files)}...")

        try:
            with h5py.File(fn, "r") as fin:
                # Get the number of rows in this specific file
                n_rows = fin["jets"].shape[0]
                
                # Copy every dataset in blocks
                for key in DATASET_KEYS:
                    src_dset = fin[key]
                    dst_dset = out_dsets[key]
                    
                    start = 0
                    while start < n_rows:
                        end = min(start + copy_block_rows, n_rows)
                        
                        # Write into the specific slice of the output
                        # [global_offset + start : global_offset + end]
                        dst_dset[global_offset + start : global_offset + end] = src_dset[start:end]
                        
                        start = end
                
                # Move the global pointer forward
                global_offset += n_rows
                
        except OSError:
            print(f"Error reading file {fn}, skipping...")

print(f"\nSuccess! Wrote {out_file} with {global_offset} total rows.")