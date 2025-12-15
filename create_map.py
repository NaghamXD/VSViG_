import torch
import os
import glob
import json
import re
from tqdm import tqdm

# CONFIGURATION
OUTPUT_FOLDER = 'processed_data'
MAP_FILE = os.path.join(OUTPUT_FOLDER, 'chunk_map.json')

def main():
    print("🗺️  Building Data Map...")
    
    # 1. Identify Files in the same order as your processing/merging logic
    # (Numbered -> FINAL -> Patches)
    all_files = os.listdir(OUTPUT_FOLDER)
    main_files = []
    
    for f in all_files:
        if f.startswith('chunk_data_') and f.endswith('.pt') and 'FINAL' not in f and '60002' not in f:
            # Check if it follows the pattern chunk_data_100.pt
            if re.search(r'_\d+\.pt$', f):
                main_files.append(os.path.join(OUTPUT_FOLDER, f))
    
    # Sort numerically (100, 200, 300...)
    main_files.sort(key=lambda x: int(re.findall(r'_(\d+)\.pt$', x)[0]))
    
    # Add the "special" files in the order they were likely created
    # 1. FINAL (Leftovers from main run)
    if os.path.exists(os.path.join(OUTPUT_FOLDER, 'chunk_data_FINAL.pt')):
        main_files.append(os.path.join(OUTPUT_FOLDER, 'chunk_data_FINAL.pt'))
        
    # 2. Patch (Pat14 Sz2)
    if os.path.exists(os.path.join(OUTPUT_FOLDER, 'chunk_data_60001.pt')):
        main_files.append(os.path.join(OUTPUT_FOLDER, 'chunk_data_60001.pt'))

    # 3. Build the Map
    # Map Structure: { "global_index": ["filename", local_index] }
    chunk_map = {}
    global_idx = 0
    
    for f_path in tqdm(main_files, desc="Scanning Chunks"):
        # Load only metadata if possible? No, torch.load reads the header.
        # We assume dimension 0 is batch size.
        try:
            # We map_location to CPU to be safe
            data = torch.load(f_path, map_location='cpu')
            batch_size = data.shape[0]
            filename = os.path.basename(f_path)
            
            for local_i in range(batch_size):
                # We map the global ID to the specific file and index
                chunk_map[str(global_idx)] = [filename, local_i]
                global_idx += 1
                
            del data # Free memory
        except Exception as e:
            print(f"❌ Error reading {f_path}: {e}")
            return

    # 4. Save Map
    with open(MAP_FILE, 'w') as f:
        json.dump(chunk_map, f)
        
    print(f"✅ Map created for {global_idx} clips.")
    print(f"   Saved to {MAP_FILE}")

if __name__ == '__main__':
    main()