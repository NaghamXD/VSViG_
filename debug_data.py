import torch
import json
import os
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# --- CONFIGURATION (Match your train.py) ---
PROCESSED_FOLDER = "processed_data"
TRAIN_LABEL_PATH = os.path.join(PROCESSED_FOLDER, 'train_labels.json')

# --- RE-DEFINE DATASET CLASS (Copy from train.py) ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        with open(label_file, 'rb') as f:
            self._labels = json.load(f)
            
        map_path = os.path.join(data_folder, 'chunk_map.json')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Chunk map not found at {map_path}")
            
        with open(map_path, 'r') as f:
            self._chunk_map = json.load(f)
            
        self.last_chunk_name = None
        self.last_chunk_data = None
        self.last_chunk_kpts = None

    def __getitem__(self, idx):
        target = float(self._labels[idx][1])
        global_id = str(self._labels[idx][0])
        
        if global_id not in self._chunk_map:
            raise IndexError(f"Global ID {global_id} not found")
            
        filename, local_idx = self._chunk_map[global_id]
        
        if filename != self.last_chunk_name:
            self.last_chunk_name = filename
            data_path = os.path.join(self._folder, filename)
            kpts_path = os.path.join(self._folder, filename.replace('chunk_data', 'chunk_kpts'))
            self.last_chunk_data = torch.load(data_path, map_location='cpu')
            self.last_chunk_kpts = torch.load(kpts_path, map_location='cpu')
            
        data = self.last_chunk_data[local_idx] 
        kpts = self.last_chunk_kpts[local_idx]
        
        # SKIP NORMALIZATION FOR DEBUGGING (We want to see raw pixels if possible, 
        # or confirm raw values before norm)
        # But let's verify what comes out of the chunk directly.
        
        sample = {'data': data, 'kpts': kpts}
        return sample, target, global_id

    def __len__(self):
        return len(self._labels)

# --- DEBUGGING FUNCTIONS ---
def save_patches_grid(patches_tensor, sample_id, frame_idx=0):
    """
    patches_tensor: (30, 15, 3, 32, 32)
    We visualize patches for one specific frame.
    """
    # Select frame
    frame_patches = patches_tensor[frame_idx] # (15, 3, 32, 32)
    
    # Convert to numpy, permute to HWC, scale to 0-255
    # Input is 0-1 float (from preprocessing)
    imgs = frame_patches.permute(0, 2, 3, 1).numpy() * 255
    imgs = imgs.astype(np.uint8)
    
    # Create a grid 3x5
    rows = []
    for i in range(0, 15, 5):
        row = np.hstack(imgs[i:i+5]) # Stack 5 images horizontally
        rows.append(row)
    
    grid = np.vstack(rows) # Stack rows vertically
    
    # Save
    os.makedirs('debug_output', exist_ok=True)
    cv2.imwrite(f'debug_output/sample_{sample_id}_frame_{frame_idx}.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved extracted patches to debug_output/sample_{sample_id}_frame_{frame_idx}.png")

def analyze_keypoints(kpts_tensor):
    """
    kpts_tensor: (30, 15, 2)
    Checks for zeros, bounds, etc.
    """
    print("  Keypoints Analysis:")
    print(f"    Shape: {kpts_tensor.shape}")
    print(f"    Min: {kpts_tensor.min():.1f}, Max: {kpts_tensor.max():.1f}")
    
    # Count (0,0) keypoints (indicating missing detection)
    zeros = torch.sum((kpts_tensor[:, :, 0] == 0) & (kpts_tensor[:, :, 1] == 0))
    total = kpts_tensor.numel() / 2
    print(f"    Missing/Zero Keypoints: {zeros} / {int(total)} ({zeros/total*100:.1f}%)")
    
    # Count Center Defaults (960, 540)
    centers = torch.sum((kpts_tensor[:, :, 0] == 960) & (kpts_tensor[:, :, 1] == 540))
    print(f"    Default Center Keypoints: {centers} / {int(total)} ({centers/total*100:.1f}%)")

def main():
    print("🚀 Starting Debug Analysis...")
    dataset = vsvig_dataset(data_folder=PROCESSED_FOLDER, label_file=TRAIN_LABEL_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Shuffle to get random samples
    
    # Inspect 5 random samples
    for i, (sample, target, global_id) in enumerate(dataloader):
        if i >= 5: break
        
        # global_id is a tuple ('id',) because batch_size=1 and strings collate to lists/tuples
        current_id = global_id[0]
        
        print(f"\n--- Sample {i+1} (ID: {current_id}) ---")
        print(f"  Target Label: {target.item():.4f}")
        
        data = sample['data'][0] # Remove batch dim -> (30, 15, 3, 32, 32)
        kpts = sample['kpts'][0] # Remove batch dim -> (30, 15, 2)
        
        # 1. Analyze Keypoints
        analyze_keypoints(kpts)
        
        # 2. Visualize Patches (Frame 0 and Frame 15)
        save_patches_grid(data, current_id, frame_idx=0)
        save_patches_grid(data, current_id, frame_idx=15)

if __name__ == '__main__':
    main()