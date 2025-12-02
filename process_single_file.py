import torch
import cv2
import numpy as np
import pandas as pd
import os
import json
import math
from torchvision import transforms
from tqdm import tqdm
import re

# --- CONFIGURATION ---
TARGET_PATIENT = "Pat14"
TARGET_SEIZURE = "Sz2"
TARGET_FILENAME = "Sz2PG.mp4" # Double check this filename in your folder!

# Start numbering from a safe high number so it doesn't overwrite your main chunks
CHUNK_ID_START = 6002 

OUTPUT_FOLDER = 'processed_data'
DATASET_ROOT = 'WU-SAHZU-EMU-Video/dataset' 
EXCEL_FILE = os.path.join(DATASET_ROOT, 'Label.xlsx')
EXISTING_LABEL_FILE = os.path.join(OUTPUT_FOLDER, 'labels.json')

POSE_WEIGHTS = 'pose.pth'
PATCH_SIZE = 32                  
CLIP_FRAMES = 30                 
STRIDE = 5                       

# --- IMPORTS & HELPERS ---
try:
    from extract_patches import extract_patches
    from models.with_mobilenet import PoseEstimationWithMobileNet
    from modules.keypoints import extract_keypoints
    from modules.load_state import load_state 
except ImportError:
    print("❌ Error: Modules not found. Run this in the VSViG folder.")
    exit()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def time_to_sec(time_val):
    if pd.isna(time_val): return -1
    if hasattr(time_val, 'hour'): return time_val.hour * 3600 + time_val.minute * 60 + time_val.second
    try:
        t_str = str(time_val).strip()
        parts = t_str.split(':')
        if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
    except: return -1
    return -1

def get_label(current_time, eeg_start, clinical_start, k=5):
    if current_time < eeg_start: return 0.0
    elif current_time > clinical_start: return 1.0
    else:
        transition_len = clinical_start - eeg_start
        if transition_len <= 0: return 1.0
        x = (current_time - eeg_start) / transition_len
        return (np.exp(k * x) - 1) / (np.exp(k) - 1)

pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(POSE_WEIGHTS, map_location='cpu')
    load_state(net, checkpoint)
    net = net.to(device)
    net.eval()
    return net

pose_net = load_model()

# Use your working extract_features function
def extract_features(frame, net):
    net_input_height_size = 256
    upsample_ratio = 4
    img_h, img_w, _ = frame.shape
    scale = net_input_height_size / img_h
    
    scaled_img = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tensor_img = pose_transform(scaled_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stages_output = net(tensor_img)
        heatmaps = stages_output[-2].cpu().numpy()
    
    full_heatmap = heatmaps[0] 
    raw_18_coords = []
    
    for joint_id in range(18):
        single_heatmap = full_heatmap[joint_id, :, :]
        found_keypoints_list = [] 
        total_kpt_num = 0
        extract_keypoints(single_heatmap, found_keypoints_list, total_kpt_num)
        
        x_orig, y_orig = img_w // 2, img_h // 2
        
        if len(found_keypoints_list) > 0:
            candidates = found_keypoints_list[0]
            if len(candidates) > 0:
                best_match = max(candidates, key=lambda x: x[2])
                x_heatmap, y_heatmap = best_match[0], best_match[1]
                x_orig = int(x_heatmap * 8 / scale)
                y_orig = int(y_heatmap * 8 / scale)

        raw_18_coords.append([x_orig, y_orig])
        
    kpts_np = np.array(raw_18_coords)
    patches_np = extract_patches(frame, kpts_np, kernel_size=128, scale=0.25)
    filtered_kpts_np = np.delete(kpts_np, [1, 14, 15], axis=0)
    patches_t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0
    kpts_t = torch.from_numpy(filtered_kpts_np)
    
    return patches_t, kpts_t

# --- MAIN PATCH LOGIC ---
def main():
    print(f"🔧 PATCH MODE: Processing {TARGET_PATIENT} {TARGET_SEIZURE}...")
    
    # 1. Load Excel & Find Row
    try:
        df = pd.read_excel(EXCEL_FILE)
        # Cleanup
        df['PatID'] = df['PatID'].astype(str).str.strip()
        df['#Seizure'] = df['#Seizure'].astype(str).str.strip()
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        return
    
    row = df[(df['PatID'] == TARGET_PATIENT) & (df['#Seizure'] == TARGET_SEIZURE)]
    if row.empty:
        print(f"❌ Error: Cannot find {TARGET_PATIENT} {TARGET_SEIZURE} in Excel!")
        return

    eeg_time = time_to_sec(row.iloc[0]['EEG onset'])
    clin_time = time_to_sec(row.iloc[0]['Clinical Onset'])
    print(f"   Timestamps -> EEG: {eeg_time}s | Clinical: {clin_time}s")

    # 2. Open Video
    folder_name = TARGET_PATIENT.lower() 
    video_path = os.path.join(DATASET_ROOT, folder_name, TARGET_FILENAME)
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_len = int(5 * fps)
    
    step_interictal = int(5 * fps)
    step_seizure = int(1 * fps)
    
    all_clip_data = []
    all_clip_kpts = []
    all_labels = [] # Stores new labels only for now
    
    # --- FIND STARTING INDEX ---
    start_index_offset = 0
    if os.path.exists(EXISTING_LABEL_FILE):
        try:
            with open(EXISTING_LABEL_FILE, 'r') as f:
                existing_data = json.load(f)
                start_index_offset = len(existing_data)
                # If existing data is empty list, start at 0
                if len(existing_data) > 0:
                    # Look at the last ID to be safe? Or just append.
                    # Usually IDs are just sequential indices 0...N
                    # We can continue from len(existing_data)
                    pass
                print(f"   Appending to {start_index_offset} existing labels.")
        except:
            print("   Warning: Could not read existing labels, starting index at 0.")
    
    current_clip_idx = start_index_offset
    
    # --- PROCESSING LOOP ---
    current_frame = 0
    pbar = tqdm(total=total_frames - clip_len, desc="Processing")

    while current_frame < (total_frames - clip_len):
        clip_end_time = (current_frame + clip_len) / fps
        label = get_label(clip_end_time, eeg_time, clin_time)
        
        if label == 0.0: current_step = step_interictal
        else: current_step = step_seizure
        
        clip_patches = []
        clip_kpts = []
        valid_clip = True
        
        for i in range(0, clip_len, STRIDE):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + i)
            ret, frame = cap.read()
            if not ret: 
                valid_clip = False
                break
            patches, kpts = extract_features(frame, pose_net)
            clip_patches.append(patches)
            clip_kpts.append(kpts)
        
        if valid_clip and len(clip_patches) == CLIP_FRAMES:
            all_clip_data.append(torch.stack(clip_patches))
            all_clip_kpts.append(torch.stack(clip_kpts))
            
            # Append [Index, Label] format consistent with main script
            all_labels.append([current_clip_idx, label])
            current_clip_idx += 1

        current_frame += current_step
        pbar.update(current_step)
        
    cap.release()
    pbar.close()
    
    # 3. Save Specific Chunk & Update Labels
    if len(all_clip_data) > 0:
        chunk_name_data = f"chunk_data_{CHUNK_ID_START}.pt"
        chunk_name_kpts = f"chunk_kpts_{CHUNK_ID_START}.pt"
        
        print(f"💾 Saving chunk with {len(all_clip_data)} clips as {chunk_name_data}...")
        torch.save(torch.stack(all_clip_data), os.path.join(OUTPUT_FOLDER, chunk_name_data))
        torch.save(torch.stack(all_clip_kpts), os.path.join(OUTPUT_FOLDER, chunk_name_kpts))
        
        # 4. Append to Main Labels File
        if os.path.exists(EXISTING_LABEL_FILE):
            print(f"📝 Appending {len(all_labels)} new labels to {EXISTING_LABEL_FILE}...")
            with open(EXISTING_LABEL_FILE, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = [] # If file is corrupted/empty
                
                data.extend(all_labels)
                f.seek(0) # Rewind to start
                json.dump(data, f)
                f.truncate() # Delete anything leftover if new file is smaller (unlikely)
        else:
            # If no existing file, create one
            with open(EXISTING_LABEL_FILE, 'w') as f:
                json.dump(all_labels, f)
                
        print("✅ Done! Data saved and labels updated.")
    else:
        print("⚠️ No clips were generated.")

if __name__ == '__main__':
    main()