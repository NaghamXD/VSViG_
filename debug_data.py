import torch
import json
import os
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
PROCESSED_FOLDER = "processed_data"
# The split file contains the LIST of filenames to use for training
TRAIN_SPLIT_PATH = os.path.join(PROCESSED_FOLDER, 'train_labels.json')
# The labels file contains the DICT mapping filenames to probabilities
LABELS_FILE_PATH = os.path.join(PROCESSED_FOLDER, 'labels.json')

# --- RE-DEFINED DATASET CLASS ---
class vsvig_dataset(Dataset):
    def __init__(self, data_root, split_file, label_file, transform=None):
        super().__init__()
        self.root = data_root
        self._transform = transform
        
        # 1. Load the list of filenames for this split
        with open(split_file, 'r') as f:
            self.file_list = json.load(f)
            
        # 2. Load the master label dictionary
        with open(label_file, 'r') as f:
            self.label_map = json.load(f)

    def __getitem__(self, idx):
            # 1. Get the entry for this index
            entry = self.file_list[idx]
            
            # Handle case where split file contains lists [filename, label, id]
            if isinstance(entry, list):
                filename = entry[0] 
            else:
                filename = entry
            
            # 2. Get the label (Use filename EXACTLY as it appears in the JSON key)
            try:
                target = float(self.label_map[filename])
            except KeyError:
                # Fallback: try adding .pt if the key might have it in the dict
                if filename + '.pt' in self.label_map:
                    target = float(self.label_map[filename + '.pt'])
                else:
                    print(f"❌ Error: Filename '{filename}' not found in labels.json.")
                    raise

            # 3. Construct paths (Handle missing .pt extension for file loading)
            # If the json key is "pat08..." but file is "pat08....pt", we must add the extension here
            file_on_disk = filename
            if not file_on_disk.endswith('.pt'):
                file_on_disk += '.pt'

            patch_path = os.path.join(self.root, 'patches', file_on_disk)
            kpt_path = os.path.join(self.root, 'kpts', file_on_disk)
            
            # 4. Load the individual tensors
            try:
                data = torch.load(patch_path, map_location='cpu')
                kpts = torch.load(kpt_path, map_location='cpu')
            except FileNotFoundError:
                print(f"❌ Critical Error: Code looked for: {patch_path}")
                print(f"   But checking the folder, that file isn't there.")
                print(f"   (Did you delete 'patches' folder or rename files?)")
                raise

            sample = {'data': data, 'kpts': kpts}
            return sample, target, filename

    def __len__(self):
        return len(self.file_list)

# --- DEBUGGING FUNCTIONS ---
def save_patches_grid(patches_tensor, sample_id, frame_idx=0):
    """
    patches_tensor: (30, 15, 3, 32, 32)
    """
    # Select frame
    frame_patches = patches_tensor[frame_idx] 
    
    # Handle Data Types (Float 0-1 vs Byte 0-255)
    imgs = frame_patches.permute(0, 2, 3, 1).numpy()
    
    # If using the new preprocessing (notebook), data is likely normalized or float
    if imgs.max() <= 1.05: # Allow small margin
        imgs = imgs * 255.0
        
    imgs = imgs.astype(np.uint8)
    
    # Create a grid 3x5
    rows = []
    for i in range(0, 15, 5):
        batch = imgs[i:i+5]
        if len(batch) > 0:
            row = np.hstack(batch) 
            rows.append(row)
    
    if rows:
        grid = np.vstack(rows)
        os.makedirs('debug_output', exist_ok=True)
        # Use clean ID for filename
        clean_id = str(sample_id).replace('.pt', '')
        cv2.imwrite(f'debug_output/{clean_id}_frame_{frame_idx}.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  📸 Saved grid: debug_output/{clean_id}_frame_{frame_idx}.png")

def analyze_keypoints(kpts_tensor):
    print("  Keypoints Analysis:")
    print(f"    Shape: {kpts_tensor.shape}")
    print(f"    Min: {kpts_tensor.min():.1f}, Max: {kpts_tensor.max():.1f}")
    
    zeros = torch.sum((kpts_tensor[:, :, 0] == 0) & (kpts_tensor[:, :, 1] == 0))
    total = kpts_tensor.numel() / 2
    if total > 0:
        print(f"    Missing/Zero Keypoints: {zeros} / {int(total)} ({zeros/total*100:.1f}%)")

def main():
    print("🚀 Starting Debug Analysis...")
    
    if not os.path.exists(TRAIN_SPLIT_PATH):
        print(f"❌ Error: {TRAIN_SPLIT_PATH} not found.")
        return

    dataset = vsvig_dataset(
        data_root=PROCESSED_FOLDER, 
        split_file=TRAIN_SPLIT_PATH,
        label_file=LABELS_FILE_PATH
    )
    
    print(f"✅ Dataset initialized with {len(dataset)} samples.")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (sample, target, filename) in enumerate(dataloader):
        if i >= 5: break
        
        # filename is a tuple due to batching
        current_file = filename[0]
        
        print(f"\n--- Sample {i+1} : {current_file} ---")
        print(f"  Target Label: {target.item():.4f}")
        
        data = sample['data'][0]
        kpts = sample['kpts'][0]
        
        analyze_keypoints(kpts)
        save_patches_grid(data, current_file, frame_idx=0)
        save_patches_grid(data, current_file, frame_idx=15)

if __name__ == '__main__':
    main()