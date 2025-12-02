import torch
import os
import glob
from tqdm import tqdm

# CONFIGURATION
OUTPUT_FOLDER = 'processed_data'
FINAL_DATA_NAME = 'data.pt'
FINAL_KPTS_NAME = 'kpts.pt'

def main():
    # 1. Find all chunk files
    # We look for files matching the pattern "chunk_data_*.pt"
    data_chunks = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, 'chunk_data_*.pt')))
    kpts_chunks = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, 'chunk_kpts_*.pt')))
    
    if len(data_chunks) == 0:
        print(f"❌ Error: No chunks found in {OUTPUT_FOLDER}")
        print("Run preprocess_data.py first!")
        return

    print(f"🔎 Found {len(data_chunks)} data chunks.")
    
    # 2. Merge Data Tensors
    print("⏳ Merging Data Tensors (this might take a moment)...")
    merged_data_list = []
    
    for chunk_file in tqdm(data_chunks, desc="Loading Data Chunks"):
        # Load each chunk and append to list
        # Map location to CPU to avoid filling GPU RAM during merge
        chunk = torch.load(chunk_file, map_location='cpu')
        merged_data_list.append(chunk)
        
    # Concatenate all chunks along dimension 0 (Batch dimension)
    full_data = torch.cat(merged_data_list, dim=0)
    print(f"✅ Data Merge Complete. Shape: {full_data.shape}")
    
    # Save the massive file
    save_path_data = os.path.join(OUTPUT_FOLDER, FINAL_DATA_NAME)
    print(f"💾 Saving to {save_path_data}...")
    torch.save(full_data, save_path_data)
    
    # Free memory
    del full_data
    del merged_data_list
    
    # 3. Merge Keypoint Tensors
    print("-" * 30)
    print("⏳ Merging Keypoint Tensors...")
    merged_kpts_list = []
    
    for chunk_file in tqdm(kpts_chunks, desc="Loading Kpts Chunks"):
        chunk = torch.load(chunk_file, map_location='cpu')
        merged_kpts_list.append(chunk)
        
    full_kpts = torch.cat(merged_kpts_list, dim=0)
    print(f"✅ Keypoints Merge Complete. Shape: {full_kpts.shape}")
    
    # Save keypoints
    save_path_kpts = os.path.join(OUTPUT_FOLDER, FINAL_KPTS_NAME)
    print(f"💾 Saving to {save_path_kpts}...")
    torch.save(full_kpts, save_path_kpts)
    
    print("-" * 30)
    print("🎉 All Done! You are ready for split_data.py")

if __name__ == '__main__':
    main()