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

# Paper Params
PATCH_SIZE = 32                  # From your train.py comments
CLIP_FRAMES = 30                 # 5 seconds * (30fps / stride 5)
STRIDE = 5                       # Frame sampling stride
SIGMA_SCALE = 0.3                  # Number of joints to keep (matching train.py)

# Time Limits (VSViG Paper Constraints)
MAX_INTERICTAL_MIN = 30          # "30 min interictal periods before EEG onset" [cite: 275]
MAX_ICTAL_MIN = 2                # "period < 2min after clinical onset" [cite: 567]

# Standard COCO Keypoint mapping (Lightweight OpenPose uses 18 usually)
# We will extract all 18 initially, then train.py filters them.
# Keypoints: 0:Nose, 1:Neck, 2:RShou, 3:RElb, 4:RWri, 5:LShou, 6:LElb, 7:LWri, 8:MidHip, 9:RHip, 10:RKnee,
#  11:RAnk, 12:LHip, 13:LKnee, 14:LAnk, 15:REye, 16:LEye, 17:REar, 18:LEar

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
                
def gen_gaussian_kernel(size, sigma):
    """
    Generates the Gaussian heatmap for fusion.
    Adapted from VSViG paper Eq 1 logic.
    """
    # Create a grid of (x,y) coordinates
    kernel = np.fromfunction(
        lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), 
        (size, size)
    )
    # Normalize to 0-1 range as per typical heatmap usage in fusion
    kernel = kernel / np.max(kernel)
    return kernel

def extract_patches_integrated(img, sorted_kpts_15, kernel_size=128, scale=0.25):
    """
    Corrected version of extract_patches.
    - Uses the SORTED 15 keypoints (does not delete indices internally).
    - Implements the Fusion Strategy: Patch * Gaussian 
    """
    img_h, img_w, _ = img.shape
    
    # Pad image to handle boundary keypoints
    pad_img = np.zeros((img_h + kernel_size*2, img_w + kernel_size*2, 3), dtype=np.uint8)
    pad_img[kernel_size:-kernel_size, kernel_size:-kernel_size, :] = img
    
    # 1. Generate Gaussian Kernel (Sigma=0.3 relative to patch size)
    # Note: VSViG paper says sigma=0.3 relative to patch size. 
    # If Kernel is 128, sigma is 128*0.3 = 38.4. 
    # Then we resize by 0.25 to get 32x32.
    sigma = kernel_size * SIGMA_SCALE
    kernel = gen_gaussian_kernel(kernel_size, sigma)
    
    # Expand kernel to 3 channels for element-wise multiplication with RGB
    kernel = np.expand_dims(kernel, 2).repeat(3, axis=2)
    
    patches = []
    
    # 2. Iterate over the 15 correct VSViG joints
    for idx in range(15):
        kx, ky = sorted_kpts_15[idx]
        
        # Adjust coordinates for padding (shift by kernel_size)
        # Center of the patch is the keypoint
        y_start = int(ky + kernel_size - 0.5 * kernel_size)
        y_end   = int(ky + kernel_size + 0.5 * kernel_size)
        x_start = int(kx + kernel_size - 0.5 * kernel_size)
        x_end   = int(kx + kernel_size + 0.5 * kernel_size)
        
        # Crop
        raw_patch = pad_img[y_start:y_end, x_start:x_end, :]
        
        # 3. FUSION: "fuse the raw RGB frames" via multiplication [cite: 165, 168]
        # We normalize raw patch to 0-255 float for math, then mult by 0-1 kernel
        fused_patch = raw_patch.astype(np.float32) * kernel
        
        # 4. Resize to final size (128 -> 32)
        # Using INTER_LINEAR as per your snippet
        resized_patch = cv2.resize(fused_patch, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255 range strictly before outputting
        # (Your snippet used a min-max norm per patch, standard is usually just ensuring range)
        # We will keep it simple: clip and cast.
        resized_patch = np.clip(resized_patch, 0, 255)
        
        patches.append(resized_patch)
        
    return np.array(patches) # (15, 32, 32, 3)

def extract_features(frame, net, prev_kpts=None):
    # 1. OpenPose Inference
    net_input_height_size = 256
    img_h, img_w, _ = frame.shape
    scale = net_input_height_size / img_h
    scaled_img = cv2.resize(frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tensor_img = pose_transform(scaled_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stages_output = net(tensor_img)
        heatmaps = stages_output[-2].cpu().numpy()[0] # (19, H, W)
    
    # 2. Extract Keypoints (Iterate 18 COCO points)
    raw_18_coords = []
    for joint_id in range(18):
        single_heatmap = heatmaps[joint_id, :, :]
        found_keypoints_list = [] 
        extract_keypoints(single_heatmap, found_keypoints_list, 0)
        
        # Fallback Logic
        x_orig, y_orig = img_w // 2, img_h // 2
        if prev_kpts is not None:
            x_orig, y_orig = prev_kpts[joint_id]
            
        if len(found_keypoints_list) > 0:
            candidates = found_keypoints_list[0]
            if len(candidates) > 0:
                best_match = max(candidates, key=lambda x: x[2])
                x_orig = int(best_match[0] * 8 / scale)
                y_orig = int(best_match[1] * 8 / scale)

        raw_18_coords.append([x_orig, y_orig])
        
    kpts_np = np.array(raw_18_coords)
    
    # 3. Filter & Reorder to VSViG Format (15 points)
    # VSViG Order: Head(Nose,REye,LEye), RArm, RLeg, LArm, LLeg
    # COCO Indices: 0, 14, 15... (Excluding 1, 16, 17)
    vsvig_indices = [0, 14, 15, 2, 3, 4, 8, 9, 10, 5, 6, 7, 11, 12, 13]
    sorted_kpts_np = kpts_np[vsvig_indices]
    
    # 4. Extract Patches (Corrected Function)
    patches_np = extract_patches_integrated(frame, sorted_kpts_np, kernel_size=128, scale=0.25)
    
    # 5. Convert to Tensor (Normalize 0-1)
    patches_t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0
    kpts_t = torch.from_numpy(sorted_kpts_np)
    
    return patches_t, kpts_t, kpts_np

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
    
    # 3. Specific Exclusions - this video has to be skipped
    # Exclude "Sz4" specifically for "Pat05"
    if pat_id == "Pat05" and "sz4" in clean_name:
        print(f"   🚫 Skipping excluded file: {filename} for {pat_id}")
        return None, None
    
    # 4. Extract Seizure ID using Regex
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
    
    # Get patient folders
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
            # 1. Parse Filename
            pat_id, sz_id = parse_filename_info(pat_folder, vid_file)
            
            # Skip if file shouldn't be processed
            if pat_id is None:
                continue
                
            # 2. Match with Excel
            row = df[(df['PatID'] == pat_id) & (df['#Seizure'] == sz_id)]
            
            if row.empty:
                print(f"   ⚠️  Warning: File '{vid_file}' found, but NO entry in Excel for {pat_id} - {sz_id}")
                continue
            
            # 3. Get Times
            eeg_time = time_to_sec(row.iloc[0]['EEG onset'])
            clin_time = time_to_sec(row.iloc[0]['Clinical Onset'])
            
            if eeg_time == -1 or clin_time == -1:
                print(f"   ❌ Error: Invalid timestamps in Excel for {pat_id} {sz_id}")
                continue
            # ==========================================
            #  TIME TRUNCATION (VSViG Paper Rules)
            # ==========================================
            # Interictal: Max 30 mins before EEG onset [cite: 275]
            # Ictal: Max 2 mins after clinical onset [cite: 567]

            start_seconds = max(0, eeg_time - (MAX_INTERICTAL_MIN * 60))
            end_seconds = clin_time + (MAX_ICTAL_MIN * 60)
            
            print(f"   ✅ {pat_id} {sz_id} | Window: {start_seconds}s to {end_seconds}s")
            
            # ==========================================
            #  VIDEO PROCESSING
            # ==========================================
            video_path = os.path.join(curr_pat_path, vid_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Convert time window to frames
            start_frame = int(start_seconds * fps)
            end_frame = int(end_seconds * fps)
            if end_frame > total_vid_frames: end_frame = total_vid_frames
            
            clip_len = int(5 * fps) # 5 seconds raw duration
            
            # Step sizes (Overlap Strategy)
            # Interictal: No overlap (Step = 5s) [cite: 573]
            # Ictal/Transition: 4s overlap (Step = 1s) [cite: 573]
            step_interictal = int(5 * fps)
            step_seizure = int(1 * fps) 
            
            current_frame = start_frame
            pbar = tqdm(total=end_frame - start_frame, desc=f"   Processing")
            
            # FIX: Initialize once per video file
            last_known_kpts = None
            
            while current_frame < (end_frame - clip_len):
                
                # A. Determine Label First (to decide step size)
                clip_mid_time = (current_frame + clip_len/2) / fps
                # Use end time for label definition usually, but for step size check mid or start is fine
                label = get_label(clip_mid_time, eeg_time, clin_time)
                
                # B. Decide Step Size based on Label
                if label == 0.0:
                    current_step = step_interictal
                else:
                    current_step = step_seizure
                
                # --- YOUR "FRAME -1" STRATEGY ---
                # If we are overlapping (step < clip_len) OR if it's the very first frame,
                # we need to ensure the state is causally correct.
                
                # Case 1: Overlapping (Seizure Mode)
                # We jumped back in time. 'last_known_kpts' currently holds future data.
                # We must fetch the true "past" (Frame -1).
                if current_step < clip_len and current_frame > 0:
                    # temporarily jump back 1 frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                    ret_prev, frame_prev = cap.read()
                    
                    if ret_prev:
                        # Run inference ONLY to get keypoints (ignore patches)
                        # We pass prev_kpts=None here because frame-1 is our "best guess" reset point.
                        # Ideally, OpenPose works well enough on a single frame to re-orient.
                        _, _, raw_kpts_init = extract_features(frame_prev, pose_net, prev_kpts=None)
                        last_known_kpts = raw_kpts_init
                    else:
                        # If reading frame-1 fails, fall back to None (Center)
                        last_known_kpts = None
                
                # Case 2: First Frame of Video
                elif current_frame == 0:
                    last_known_kpts = None

                # Case 3: Interictal (Continuous)
                # We leave 'last_known_kpts' exactly as it is. 
                # The end of Clip A (Frame 150) is the input for Clip B (Frame 151).

                # C. # Extract Clip (Stride 5: 150 frames -> 30 frames)
                clip_patches = []
                clip_kpts = []
                valid_clip = True
                
                # We need exactly 30 frames extracted over the 5s window
                # Range 0 to clip_len, stepping by STRIDE (5)
                for i in range(0, clip_len, STRIDE):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + i)
                    ret, frame = cap.read()
                    if not ret: 
                        valid_clip = False
                        break
                    
                    # Pass state and receive updated state
                    patches, kpts, raw_kpts_np = extract_features(frame, pose_net, prev_kpts=last_known_kpts)
                    
                    # Update state for next frame iteration
                    last_known_kpts = raw_kpts_np
                    
                    clip_patches.append(patches)
                    clip_kpts.append(kpts)
                
                # D. Verify & Store
                if valid_clip and len(clip_patches) == CLIP_FRAMES:
                    final_clip_tensor = torch.stack(clip_patches) # (30, 15, 3, 32, 32)
                    final_kpts_tensor = torch.stack(clip_kpts)    # (30, 15, 2)
                    
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