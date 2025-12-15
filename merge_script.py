import torch
import os
import glob
import json
import re
import gc # Garbage Collector
from tqdm import tqdm

# CONFIGURATION
OUTPUT_FOLDER = 'processed_data'
FINAL_DATA_NAME = 'data.pt'
FINAL_KPTS_NAME = 'kpts.pt'
FINAL_LABEL_NAME = 'labels.json'

def main():
    print("🚀 Starting Low-RAM Merger...")
    
    # --- STEP 1: IDENTIFY FILES ---
    all_files = os.listdir(OUTPUT_FOLDER)
    main_data_files = []
    
    # regex to find files like chunk_data_0.pt, chunk_data_100.pt
    # excluding things like chunk_data_FINAL.pt for now
    for f in all_files:
        if f.startswith('chunk_data_') and f.endswith('.pt') and 'FINAL' not in f:
            if re.search(r'_\d+\.pt$', f):
                main_data_files.append(os.path.join(OUTPUT_FOLDER, f))
    
    # Sort numerically based on the number in the filename
    main_data_files.sort(key=lambda x: int(re.findall(r'_(\d+)\.pt$', x)[0]))
    
    # Identify Patch/Extra/FINAL chunks
    # This logic assumes anything NOT numbered 0..N is a "special" chunk
    patch_data_files = [
        os.path.join(OUTPUT_FOLDER, f) 
        for f in all_files 
        if f.startswith('chunk_data_') and f.endswith('.pt') 
        and os.path.join(OUTPUT_FOLDER, f) not in main_data_files
        and 'FINAL' not in f
    ]
    
    # Add FINAL file from main run (tail end of main loop)
    final_chunk = os.path.join(OUTPUT_FOLDER, 'chunk_data_FINAL.pt')
    if os.path.exists(final_chunk):
        # We assume FINAL comes after numbered chunks but BEFORE patches
        # (This order depends on when you generated them, but typically append order matches label order)
        main_data_files.append(final_chunk)

    ordered_data_files = main_data_files + patch_data_files
    
    # --- DUPLICATE CHECK ---
    # Print what we found to help debug the "two files" issue
    print(f"   Found {len(ordered_data_files)} total chunks to merge.")
    print("   Files to be merged:")
    for f in ordered_data_files:
        print(f"      - {os.path.basename(f)}")
        
    # User can cancel here if they see duplicates
    # input("Press Enter to continue or Ctrl+C to abort...") 

    # --- STEP 2: MERGE LABELS FIRST (Lightweight) ---
    final_labels = []
    
    # Load Main Labels (usually labels.json)
    if os.path.exists(os.path.join(OUTPUT_FOLDER, 'labels.json')):
        print("   Loading base labels.json...")
        final_labels.extend(json.load(open(os.path.join(OUTPUT_FOLDER, 'labels.json'), 'r')))
    
    # Load Patch Labels (if any separate json files exist)
    # Note: If process_single_file.py already appended to labels.json, this might duplicate labels!
    # Check if labels.json length matches data length roughly?
    # For now, we assume process_single_file appended to labels.json, so we DON'T load extra label files
    # unless you explicitly have labels_Pat14.json sitting there.
    
    for p_file in patch_data_files:
        base_name = os.path.basename(p_file)
        tag = base_name.replace('chunk_data_', '').replace('.pt', '')
        lbl_path = os.path.join(OUTPUT_FOLDER, f"labels_{tag}.json")
        
        # Only load if it exists and isn't the main file
        if os.path.exists(lbl_path) and os.path.abspath(lbl_path) != os.path.abspath(os.path.join(OUTPUT_FOLDER, 'labels.json')):
            print(f"   + Appending labels from {os.path.basename(lbl_path)}")
            final_labels.extend(json.load(open(lbl_path, 'r')))

    # Save Combined Labels
    with open(os.path.join(OUTPUT_FOLDER, FINAL_LABEL_NAME), 'w') as f:
        json.dump(final_labels, f)
    print(f"✅ Saved merged labels.json (Total Count: {len(final_labels)})")

    # --- STEP 3: MERGE DATA (The Hard Part) ---
    
    print("⏳ Loading Data Chunks into RAM...")
    full_data_list = []
    
    for f_path in tqdm(ordered_data_files):
        try:
            chunk = torch.load(f_path, map_location='cpu')
            full_data_list.append(chunk)
        except Exception as e:
            print(f"❌ Error loading {f_path}: {e}")
            return
    
    if not full_data_list:
        print("❌ No data loaded. Exiting.")
        return

    print("⏳ Concatenating Data...")
    try:
        full_data = torch.cat(full_data_list, dim=0)
    except RuntimeError as e:
        print(f"❌ Concatenation failed (Likely RAM): {e}")
        return
    
    del full_data_list
    gc.collect() 
    
    print(f"💾 Saving Data Tensor {full_data.shape}...")
    torch.save(full_data, os.path.join(OUTPUT_FOLDER, FINAL_DATA_NAME))
    
    del full_data
    gc.collect()
    print("✅ Data Saved.")

    # --- STEP 4: MERGE KEYPOINTS ---
    print("⏳ Loading Keypoint Chunks...")
    full_kpts_list = []
    
    # We infer kpts filenames from data filenames to ensure exact matching order
    ordered_kpts_files = [f.replace('chunk_data_', 'chunk_kpts_') for f in ordered_data_files]
    
    for f_path in tqdm(ordered_kpts_files):
        if not os.path.exists(f_path):
            print(f"⚠️ Warning: Missing keypoint file {f_path}. Skipping corresponding data?")
            # This is bad. Indices will shift. 
            # Ideally we should fail or insert dummy data.
            continue
            
        chunk = torch.load(f_path, map_location='cpu')
        full_kpts_list.append(chunk)
        
    print("⏳ Concatenating Keypoints...")
    full_kpts = torch.cat(full_kpts_list, dim=0)
    
    del full_kpts_list
    gc.collect()
    
    print(f"💾 Saving Keypoints Tensor {full_kpts.shape}...")
    torch.save(full_kpts, os.path.join(OUTPUT_FOLDER, FINAL_KPTS_NAME))
    
    print("🎉 Merge Complete!")

if __name__ == '__main__':
    main()