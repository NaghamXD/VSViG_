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

try:
    from extract_patches import extract_patches
except ImportError:
    print("❌ Error: Could not find 'extract_patches.py'.")
    print("Make sure the authors' script is saved as 'extract_patches.py' in this folder.")
    exit()

# --- 1. CONFIGURATION ---
OUTPUT_FOLDER = 'processed_data' # Where to save the .pt files
DATASET_ROOT = 'WU-SAHZU-EMU-Video/dataset' 
EXCEL_FILE = os.path.join(DATASET_ROOT, 'Label.xlsx')
POSE_WEIGHTS = 'pose.pth'
PATCH_SIZE = 32                  # From your train.py comments
CLIP_FRAMES = 30                 # 5 seconds * (30fps / stride 5)
STRIDE = 5                       # Frame sampling stride
NUM_JOINTS = 15                  # Number of joints to keep (matching train.py)

# Standard COCO Keypoint mapping (Lightweight OpenPose uses 18 usually)
# We will extract all 18 initially, then train.py filters them.
# Keypoints: 0:Nose, 1:Neck, 2:RShou, 3:RElb, 4:RWri, 5:LShou, 6:LElb, 7:LWri, ...

# --- 2. LOAD POSE MODEL (Updated for OpenPose-Lightweight) ---
# Ensure you have cloned the 'lightweight-human-pose-estimation.pytorch' repo
# and placed its folders (models, modules) next to this script.

try:
    # This is the standard path in Daniil-Osokin's repo
    from models.with_mobilenet import PoseEstimationWithMobileNet
    from modules.keypoints import extract_keypoints, group_keypoints
    from modules.load_state import load_state # They often have a helper for loading weights
except ImportError:
    print("❌ Error: Could not find 'models.with_mobilenet'.")
    print("Action: Clone the repo: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch")
    print("And copy the 'models' and 'modules' folders into your project directory.")
    exit()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def load_model():
    net = PoseEstimationWithMobileNet()
    # Use their custom loader if available, or standard torch.load
    checkpoint = torch.load(POSE_WEIGHTS, map_location='cpu') # Load to cpu first
    load_state(net, checkpoint) # This helper handles the 'module.' naming issues
    net = net.to(device)
    net.eval()
    return net

pose_net = load_model()

# Image Transformer for Pose Model
pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. HELPER FUNCTIONS ---

def time_to_sec(time_val):
    """Converts Excel time (String or Time object) to total seconds"""
    if pd.isna(time_val): return -1
    
    # If it's already a time object (e.g., 00:59:40)
    if hasattr(time_val, 'hour'):
        return time_val.hour * 3600 + time_val.minute * 60 + time_val.second
        
    # If it's a string (e.g., "0:59:40")
    try:
        t_str = str(time_val).strip()
        parts = t_str.split(':')
        if len(parts) == 3: # H:M:S
            return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        if len(parts) == 2: # M:S
            return int(parts[0])*60 + int(parts[1])
    except:
        return -1
    return -1


def get_label(current_time, eeg_start, clinical_start, k=5):
    """
    Formal Exponential labeling: (e^kx - 1) / (e^k - 1)
    where x is the ratio of time passed (0 to 1).
    """
    # Zone 1: Healthy (Before EEG start)
    if current_time < eeg_start:
        return 0.0
    
    # Zone 3: Full Seizure (After Clinical start)
    elif current_time > clinical_start:
        return 1.0
    
    # Zone 2: Transition
    else:
        transition_len = clinical_start - eeg_start
        if transition_len <= 0: return 1.0 # Safety check
        
        # Calculate x (ratio from 0.0 to 1.0)
        x = (current_time - eeg_start) / transition_len
        
        # Apply the Normalized Exponential Function
        # numerator = e^(k*x) - 1
        # denominator = e^k - 1
        label = (np.exp(k * x) - 1) / (np.exp(k) - 1)
        
        return float(label)
                
'''
def extract_features(frame, net):
    # 1. OpenPose Inference
    net_input_height_size = 256
    stride = 8
    upsample_ratio = 4
    img_h, img_w, _ = frame.shape
    scale = net_input_height_size / img_h
    
    # Resize image to fit model input
    scaled_img = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tensor_img = pose_transform(scaled_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stages_output = net(tensor_img)
        stage2_heatmaps = stages_output[-2]
        stage2_pafs = stages_output[-1]
        heatmaps = stage2_heatmaps.cpu().numpy()
        pafs = stage2_pafs.cpu().numpy() # Ensure this is numpy too
        
    # Order: heatmap, paf, stride, upsample_ratio
    total_keypoints_num, all_keypoints_by_type = extract_keypoints(
        heatmaps[0], pafs[0], upsample_ratio
    )
    
    # 2. Collect 18 Raw Keypoints
    raw_18_coords = []
    
    for joint_id in range(18):
        joint_data = all_keypoints_by_type[joint_id]
        if len(joint_data) > 0:
            # joint_data format: [x, y, score, id]
            best_match = max(joint_data, key=lambda x: x[2])
            x_orig = int(best_match[0] / scale)
            y_orig = int(best_match[1] / scale)
        else:
            x_orig, y_orig = img_w // 2, img_h // 2
        raw_18_coords.append([x_orig, y_orig])
        
    kpts_np = np.array(raw_18_coords) # Shape (18, 2)
    
    # 3. Call Authors' Patch Extraction
    # Note: We pass raw 18 keypoints; their script filters them to 15.
    patches_np = extract_patches(frame, kpts_np, kernel_size=128, scale=0.25)
    
    # 4. Handle Keypoints Filtering Manually
    # We must delete the same joints the authors did (Neck, Eyes) to match the patches
    # Indices: 1 (Neck), 14 (REye), 15 (LEye)
    filtered_kpts_np = np.delete(kpts_np, [1, 14, 15], axis=0)

    # 5. Convert to PyTorch Tensors
    # Patches: (15, 32, 32, 3) -> (15, 3, 32, 32)
    patches_t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0
    kpts_t = torch.from_numpy(filtered_kpts_np)
    
    return patches_t, kpts_t
'''
def extract_features(frame, net):
    # 1. OpenPose Inference
    net_input_height_size = 256
    upsample_ratio = 4
    img_h, img_w, _ = frame.shape
    scale = net_input_height_size / img_h
    
    scaled_img = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tensor_img = pose_transform(scaled_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stages_output = net(tensor_img)
        # Get heatmaps (Stage 2 is typically the final stage in lightweight openpose)
        stage2_heatmaps = stages_output[-2]
        heatmaps = stage2_heatmaps.cpu().numpy()
        
    # Get the first image in batch (19, H, W)
    # Transpose to (H, W, 19) so we can slice channels easily
    # Note: If heatmaps are already (Batch, C, H, W), heatmaps[0] is (C, H, W).
    # We need (H, W, C) or just (C, H, W) is fine if we index [i, :, :]
    # Let's verify shape. Usually it is (1, 19, 32, 56) or similar.
    # heatmaps[0] -> (19, 32, 56)
    
    full_heatmap = heatmaps[0] 
    
    # 2. Extract Keypoints Manually Loop
    # We iterate over 18 joints (Channel 0 to 17)
    # Channel 18 is usually background
    
    raw_18_coords = []
    
    for joint_id in range(18):
        # Slice the specific channel -> 2D array (H, W)
        single_heatmap = full_heatmap[joint_id, :, :]
        
        # We need to create a dummy list for it to append to
        # The function signature in your keypoints.py: 
        # def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
        found_keypoints_list = [] 
        total_kpt_num = 0 # Dummy counter
        
        # Call the function (It modifies found_keypoints_list in-place)
        # It expects a 2D heatmap. single_heatmap is (H, W).
        extract_keypoints(single_heatmap, found_keypoints_list, total_kpt_num)
        
        # found_keypoints_list now contains a LIST of tuples for this joint
        # The function does: all_keypoints.append(keypoints_with_score_and_id)
        # So found_keypoints_list[0] is the list of candidates [(x, y, score, id), ...]
        
        x_orig, y_orig = img_w // 2, img_h // 2 # Default center
        
        if len(found_keypoints_list) > 0:
            candidates = found_keypoints_list[0]
            if len(candidates) > 0:
                # Get best match (highest score is index 2)
                # candidate format: (x, y, score, global_id)
                # We sort by score to be safe, though extract_keypoints sorts by X usually?
                # Actually extract_keypoints sorts by X coordinate (itemgetter(0)).
                # We want highest confidence (index 2).
                best_match = max(candidates, key=lambda x: x[2])
                
                x_heatmap, y_heatmap = best_match[0], best_match[1]
                
                # Map back to original image
                # Heatmap stride is 8 in standard OpenPose Lightweight
                x_orig = int(x_heatmap * 8 / scale)
                y_orig = int(y_heatmap * 8 / scale)

        raw_18_coords.append([x_orig, y_orig])
        
    kpts_np = np.array(raw_18_coords)
    
    # 3. Extract Patches (Authors' Code)
    patches_np = extract_patches(frame, kpts_np, kernel_size=128, scale=0.25)
    
    # 4. Filter Keypoints (18 -> 15)
    # Removing Neck (1), R-Eye (14), L-Eye (15) to match the 15 joints
    filtered_kpts_np = np.delete(kpts_np, [1, 14, 15], axis=0)

    # 5. Convert to Tensor
    patches_t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0
    kpts_t = torch.from_numpy(filtered_kpts_np)
    
    return patches_t, kpts_t

def parse_filename_info(folder_name, filename):
    """
    Input: folder="pat01", filename="Sz1PG.mp4"
    Output: ("Pat01", "Sz1") OR (None, None) if invalid.
    """
    # 1. Filter out non-seizure files immediately
    # Based on your screenshot: 'free.mp4', 'no-Sz2P.mp4' should be ignored.
    clean_name = filename.lower()
    if clean_name.startswith('free') or clean_name.startswith('no'):
        return None, None

    # 2. Standardize Patient ID (pat01 -> Pat01)
    # The folder is 'pat01', but Excel uses 'Pat01'
    pat_id = folder_name.capitalize() 
    
    # 3. Extract Seizure ID using Regex
    # We look for "Sz" followed immediately by digits (e.g., Sz1, Sz10)
    # This matches "Sz1" inside "Sz1PG.mp4" or "Sz1P.mp4"
    match = re.search(r'(Sz\d+)', filename, re.IGNORECASE)
    
    if match:
        # Found something like "sz1" or "Sz01"
        raw_sz = match.group(1) 
        
        # Normalize: Ensure it looks like "Sz1" (Capital S, lowercase z, number)
        # This strips any accidental variations
        number_part = re.findall(r'\d+', raw_sz)[0]
        sz_id = f"Sz{int(number_part)}" # Sz1, Sz2, etc.
        
        return pat_id, sz_id
    
    return None, None

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    print(f"📂 Loading Excel from: {EXCEL_FILE}")
    try:
        # Load Excel
        df = pd.read_excel(EXCEL_FILE)
        # Cleanup: Ensure PatID is string and stripped of spaces
        df['PatID'] = df['PatID'].astype(str).str.strip()
        # Cleanup: Ensure #Seizure is formatted like "Sz1" (remove spaces)
        df['#Seizure'] = df['#Seizure'].astype(str).str.strip()
    except Exception as e:
        print(f"❌ Error loading Excel: {e}")
        return

    # Helper lists
    all_clip_data = []
    all_clip_kpts = []
    all_labels = [] 
    clip_counter = 0
    
    # Get patient folders (pat01, pat02...)
    try:
        patient_folders = sorted([f for f in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, f)) and f.lower().startswith('pat')])
    except FileNotFoundError:
        print(f"❌ Error: Dataset root '{DATASET_ROOT}' not found.")
        return

    print(f"🔎 Found {len(patient_folders)} patient folders.")

    for pat_folder in patient_folders:
        print(f"📂 Entering {pat_folder}...")
        curr_pat_path = os.path.join(DATASET_ROOT, pat_folder)
        video_files = [f for f in os.listdir(curr_pat_path) if f.endswith('.mp4')]
        
        for vid_file in video_files:
            
            # --- 1. PARSE FILENAME INFO ---
            # Make sure you have the updated parse_filename_info function defined above this main()
            pat_id, sz_id = parse_filename_info(pat_folder, vid_file)
            
            # Skip if file shouldn't be processed (e.g. "free.mp4")
            if pat_id is None:
                continue
                
            # --- 2. MATCH WITH EXCEL ---
            row = df[(df['PatID'] == pat_id) & (df['#Seizure'] == sz_id)]
            
            if row.empty:
                print(f"   ⚠️  Warning: File '{vid_file}' found, but NO entry in Excel for {pat_id} - {sz_id}")
                continue
            
            # --- 3. GET TIMES ---
            eeg_time = time_to_sec(row.iloc[0]['EEG onset'])
            clin_time = time_to_sec(row.iloc[0]['Clinical Onset'])
            
            if eeg_time == -1 or clin_time == -1:
                print(f"   ❌ Error: Invalid timestamps in Excel for {pat_id} {sz_id}")
                continue

            print(f"   ✅ Processing: {pat_id} {sz_id} | File: {vid_file} | EEG: {eeg_time}s | Clinical: {clin_time}s")
            
            # ==========================================
            #  VIDEO PROCESSING
            # ==========================================
            video_path = os.path.join(curr_pat_path, vid_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   ❌ Error: Could not open video {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            clip_len = int(5 * fps) # 5 seconds
            
            # --- DYNAMIC STEP SIZES (Paper's Balancing Strategy) ---
            # Interictal (Healthy): 5s step (No overlap)
            # Seizure/Transition: 1s step (4s overlap)
            step_interictal = int(5 * fps)
            step_seizure = int(1 * fps)
            
            current_frame = 0
            
            # Progress bar for this specific video
            pbar = tqdm(total=total_frames - clip_len, desc=f"   Processing Clips")
            
            # --- MAIN VIDEO LOOP ---
            while current_frame < (total_frames - clip_len):
                
                # A. Determine Label First (to decide step size)
                # We calculate label based on the END time of the potential clip
                clip_end_time = (current_frame + clip_len) / fps
                label = get_label(clip_end_time, eeg_time, clin_time)
                
                # B. Decide Step Size based on Label
                # If Healthy (0.0), jump 5s. If Seizure (>0.0), jump 1s.
                if label == 0.0:
                    current_step = step_interictal
                else:
                    current_step = step_seizure
                
                # C. Extract Frames (The "Stride" Logic: 150 frames -> 30 frames)
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
                
                # D. Verify & Store
                if valid_clip and len(clip_patches) == CLIP_FRAMES:
                    final_clip_tensor = torch.stack(clip_patches) # Shape: (30, 15, 3, 32, 32)
                    final_kpts_tensor = torch.stack(clip_kpts)    # Shape: (30, 15, 2)
                    
                    all_clip_data.append(final_clip_tensor)
                    all_clip_kpts.append(final_kpts_tensor)
                    all_labels.append([clip_counter, label])
                    clip_counter += 1
                    
                    # E. Memory Chunk Save
                    if len(all_clip_data) >= 100:
                        torch.save(torch.stack(all_clip_data), f"{OUTPUT_FOLDER}/chunk_data_{clip_counter}.pt")
                        torch.save(torch.stack(all_clip_kpts), f"{OUTPUT_FOLDER}/chunk_kpts_{clip_counter}.pt")
                        all_clip_data = []
                        all_clip_kpts = []

                # F. Move Forward
                current_frame += current_step
                pbar.update(current_step)

            pbar.close()
            cap.release()

    # --- SAVE LEFTOVERS (Critical Fix) ---
    # This saves the clips remaining in the buffer after all patients are processed
    if len(all_clip_data) > 0:
        print(f"💾 Saving final chunk of {len(all_clip_data)} clips...")
        torch.save(torch.stack(all_clip_data), f"{OUTPUT_FOLDER}/chunk_data_FINAL.pt")
        torch.save(torch.stack(all_clip_kpts), f"{OUTPUT_FOLDER}/chunk_kpts_FINAL.pt")
        
    # Save final labels map
    with open(f"{OUTPUT_FOLDER}/labels.json", 'w') as f:
        json.dump(all_labels, f)
    
    print("✅ Done! All patient folders processed.")

if __name__ == '__main__':
    main()