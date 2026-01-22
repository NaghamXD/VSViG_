import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import cv2
import torch
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION (UPDATE THESE PATHS) ---
# NOTE: Update 'PROJECT_ROOT' to the folder containing your 'Label.xlsx', 'pose.pth', and 'data-root'
PROJECT_ROOT = '.'  # Current directory, or change to: '/path/to/your/project'

# If you are still running this on Colab, uncomment the next two lines:
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# PROJECT_ROOT = '/content/drive/MyDrive/final project' 

DATA_ROOT    = 'WU-SAHZU-EMU-Video/dataset' 
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'processed_data')
PATCHES_DIR  = os.path.join(OUTPUT_DIR, 'patches')
KPTS_DIR     = os.path.join(OUTPUT_DIR, 'kpts')
LABELS_FILE  = os.path.join(DATA_ROOT, 'Label.xlsx')
POSE_WEIGHTS = os.path.join(PROJECT_ROOT, 'pose.pth')

# VSViG Paper Settings
FINAL_PATCH_SIZE = 32
FUSION_SIZE = 128
SIGMA_SCALE = 0.3
SIGMA = FUSION_SIZE * SIGMA_SCALE 

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Processing using device: {DEVICE}")

# --- 2. SETUP DIRECTORIES & PATHS ---
os.makedirs(PATCHES_DIR, exist_ok=True)
os.makedirs(KPTS_DIR, exist_ok=True)

# Add project root to path to find 'models' and 'modules'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✅ Paths ready:")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_ROOT   :", DATA_ROOT)
print("OUTPUT_DIR  :", OUTPUT_DIR)

# --- 3. LOAD LABELS ---
if not os.path.exists(LABELS_FILE):
    print(f"❌ Error: Label file not found at {LABELS_FILE}")
    sys.exit(1)

df = pd.read_excel(LABELS_FILE)

# Handle merged PatID/Seizure Type cells
df[['PatID', 'Seizure Type']] = df[['PatID', 'Seizure Type']].ffill()

# Standardize text
df['PatID'] = df['PatID'].astype(str).str.lower().str.strip()
df['Seizure Type'] = df['Seizure Type'].astype(str).str.strip()

print(f"✅ Excel loaded: {df.shape}")
print(df.head())

# --- 4. LOAD POSE MODEL ---
try:
    from models.with_mobilenet import PoseEstimationWithMobileNet
    from modules.keypoints import extract_keypoints, group_keypoints
    from modules.load_state import load_state
except ImportError:
    print("❌ Error: Could not import pose model modules. Ensure 'models' and 'modules' folders are in PROJECT_ROOT.")
    sys.exit(1)

net = PoseEstimationWithMobileNet().to(DEVICE).eval()
if not os.path.exists(POSE_WEIGHTS):
    print(f"❌ Error: Pose weights not found at {POSE_WEIGHTS}")
    sys.exit(1)

load_state(net, torch.load(POSE_WEIGHTS, map_location=DEVICE))
print("✅ Pose model loaded successfully.")

# --- 5. PREPARE GAUSSIAN KERNEL & JOINT INDICES ---
# 1. Generate High-Res Gaussian Kernel (128x128)
xs = np.arange(FUSION_SIZE, dtype=np.float32)
ys = np.arange(FUSION_SIZE, dtype=np.float32)
xx, yy = np.meshgrid(xs, ys, indexing='xy')

# Center at 63.5 for 128x128 grid
gauss = np.exp(-((xx - 63.5)**2 + (yy - 63.5)**2) / (2 * (SIGMA**2)))
gauss = gauss / np.max(gauss) # Normalize to [0,1]
g_filter_high = np.repeat(gauss[:, :, None], 3, axis=2).astype(np.float32)

# 2. VSViG Joint Order (15 Joints)
JOINT_INDICES = [
    0,         # Nose
    14, 15,    # Eyes
    2, 3, 4,   # Right Arm
    5, 6, 7,   # Left Arm  <-- SWAPPED to match code
    8, 9, 10,  # Right Leg <-- SWAPPED to match code
    11, 12, 13 # Left Leg
]

print(f"✅ High-Res Gaussian ({FUSION_SIZE}x{FUSION_SIZE}) + JOINT_INDICES ready.")

# --- 6. HELPER FUNCTIONS ---

def get_sec(time_val):
    if pd.isna(time_val):
        return None
    if hasattr(time_val, 'hour'):
        return int(time_val.hour) * 3600 + int(time_val.minute) * 60 + int(time_val.second)
    try:
        parts = str(time_val).split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
    except:
        return None
    return None

def calculate_label_formal(current_time, eeg_start, clinical_start, k=5):
    """
    Formal VSViG Exponential labeling.
    """
    if current_time < eeg_start:
        return 0.0 # Interictal

    if current_time > clinical_start:
        return 1.0 # Ictal

    # Transition period exponential growth
    transition_len = clinical_start - eeg_start
    if transition_len <= 0: return 1.0

    x = (current_time - eeg_start) / transition_len
    label = (np.exp(k * x) - 1) / (np.exp(k) - 1)
    return float(label)

def process_maps_to_coords(heatmap, paf, scale, orig_h, orig_w):
    """
    Processes pre-computed heatmaps to extract coordinates.
    """
    all_kpts, total = [], 0
    for i in range(18):
        total += extract_keypoints(heatmap[i], all_kpts, total)

    coords = np.full((18, 2), -1.0, dtype=np.float32)

    # PAF Grouping
    paf_hwc = paf.transpose(1, 2, 0)
    poses, all_res = group_keypoints(all_kpts, paf_hwc)

    if poses is not None and len(poses) > 0:
        # Select best pose (most detected joints)
        best_idx = max(range(len(poses)), key=lambda p: int(np.sum(poses[p][:18] != -1)))
        pose = poses[best_idx]

        for i in range(18):
            if pose[i] != -1:
                kpt = all_res[int(pose[i])]
                # Project back: (Heatmap_Coord * Stride 8.0) / Scale
                coords[i] = [float(kpt[0]) * 8.0 / scale, float(kpt[1]) * 8.0 / scale]
        return coords

    # Fallback: Heatmap Maxima
    for i in range(18):
        _, conf, _, max_loc = cv2.minMaxLoc(heatmap[i])
        if conf > 0.1:
            coords[i] = [float(max_loc[0]) * 8.0 / scale, float(max_loc[1]) * 8.0 / scale]

    return coords

def extract_clip_tensors_batched(cap, fps, t_start_sec):
    """
    Extracts a 30-frame clip using batch pose inference and high-fidelity fusion.
    """
    start_frame = int(round(t_start_sec * fps))
    end_frame   = int(round((t_start_sec + 5.0) * fps))

    # 1. Read Frames
    frame_idxs = np.linspace(start_frame, end_frame - 1, 30).astype(int)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret: break
        if i in frame_idxs:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) == 30: break

    if len(frames) < 30: return None, None

    # 2. Batch Pose Inference
    h, w, _ = frames[0].shape
    scale = 256.0 / float(h)
    batch_inp = [cv2.resize(f, (0,0), fx=scale, fy=scale) for f in frames]
    batch_inp = np.stack(batch_inp).astype(np.float32)
    batch_inp = (batch_inp - 128.0) / 256.0 # Normalize

    batch_tensor = torch.from_numpy(batch_inp).permute(0, 3, 1, 2).to(DEVICE)
    with torch.no_grad():
        out = net(batch_tensor)
        all_heatmaps = out[-2].cpu().numpy()
        all_pafs = out[-1].cpu().numpy()

    # 3. Patch Extraction & Fusion
    clip_patches, clip_kpts = [], []
    pad = 64 # Padding for 128x128 crops

    for i in range(30):
        vsvig_coords = process_maps_to_coords(all_heatmaps[i], all_pafs[i], scale, h, w)
        vsvig_coords = vsvig_coords[JOINT_INDICES]
        clip_kpts.append(vsvig_coords)

        # Pad frame
        p_img = cv2.copyMakeBorder(frames[i], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        p_batch = np.zeros((15, 32, 32, 3), dtype=np.float32)

        for j, (x, y) in enumerate(vsvig_coords):
            if x < 0 or y < 0: continue

            x_p, y_p = int(round(x + pad)), int(round(y + pad))

            # Extract 128x128 patch
            crop = p_img[y_p-64:y_p+64, x_p-64:x_p+64, :].astype(np.float32)

            if crop.shape == (128, 128, 3):
                # FUSION + DOWNSAMPLE
                fused = crop * g_filter_high
                p_batch[j] = cv2.resize(fused, (32, 32), interpolation=cv2.INTER_CUBIC) / 255.0

        clip_patches.append(p_batch)

    # Final shape: (T, N, C, H, W)
    patches = np.array(clip_patches).transpose(0, 1, 4, 2, 3)
    return patches, np.array(clip_kpts)

# Dictionary to store all labels
master_labels = {}

def process_video_to_vsvig(v_path, pat_id, sz_id, eeg_s, clin_s, window_start, window_end):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        print(f"❌ Skipping (cannot open): {v_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"❌ Bad FPS for: {v_path}")
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    t = max(float(window_start), 0.0)
    effective_end = min(float(window_end), duration_sec - 5.0)

    count = 0
    fail = 0

    print(f"🎬 {pat_id} | Sz={sz_id} | fps={fps:.2f} | window=[{t:.1f},{effective_end:.1f}]")

    while t <= effective_end:
        # 1. Adaptive Stride: 1s near seizure, 5s otherwise
        is_crit = (t + 5.0) > eeg_s
        stride = 1.0 if is_crit else 5.0

        # 2. Labeling
        label = calculate_label_formal(t + 5.0, eeg_s, clin_s)

        # 3. Extraction
        patches, kpts = extract_clip_tensors_batched(cap, fps, t)

        if patches is None:
            fail += 1
            t += stride
            continue

        # 4. Save
        t_int = int(round(t))
        file_id = f"{pat_id}_{sz_id}_{t_int}"

        patch_tensor = torch.from_numpy(patches).float()
        kpts_tensor  = torch.from_numpy(kpts).float()

        torch.save(patch_tensor, os.path.join(PATCHES_DIR, f"{file_id}.pt"))
        torch.save(kpts_tensor,  os.path.join(KPTS_DIR,    f"{file_id}.pt"))

        # 5. Record Label
        master_labels[file_id] = float(label)

        count += 1
        if count % 20 == 0:
            print(f"  📦 Saved {file_id} | stride={stride:.0f}s | label={label:.3f} | fails={fail}")

        t += stride

    cap.release()
    print(f"✅ Done: {pat_id} Sz={sz_id} | clips={count} | fails={fail}")
    return count

# --- 7. MAIN PROCESSING LOOP ---

total_clips = 0
missing_videos = 0
skipped_rows = 0

print("\n🚀 Starting Preprocessing Loop...")

for _, row in df.iterrows():
    p_id = str(row['PatID']).strip().lower()
    sz_id = str(row['#Seizure']).strip()
    sz_type = str(row['Seizure Type']).strip()

    eeg_s  = get_sec(row.get('EEG onset', None))
    clin_s = get_sec(row.get('Clinical Onset', None))

    if eeg_s is None or clin_s is None or clin_s <= eeg_s:
        skipped_rows += 1
        print(f"⚠️ Skip row: {p_id} seizure={sz_id} invalid times (EEG={eeg_s}, Clin={clin_s})")
        continue

    # Paper windowing parameters
    start_time = max(0.0, float(eeg_s) - 1800.0)
    end_time   = float(clin_s) + 120.0

    v_name = f"{sz_id}{sz_type}.mp4"
    v_path = os.path.join(DATA_ROOT, p_id, v_name)

    if not os.path.exists(v_path):
        missing_videos += 1
        print(f"❌ Video not found: {v_path}")
        continue

    total_clips += process_video_to_vsvig(
        v_path=v_path,
        pat_id=p_id,
        sz_id=sz_id,
        eeg_s=eeg_s,
        clin_s=clin_s,
        window_start=start_time,
        window_end=end_time
    )

print("\n🏁 Preprocessing complete.")
print(f"Total clips created: {total_clips}")
print(f"Total label entries: {len(master_labels)}")
print(f"Missing videos: {missing_videos}")
print(f"Skipped rows: {skipped_rows}")

# --- 8. SAVE LABELS ---
labels_path = os.path.join(OUTPUT_DIR, 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(master_labels, f)

print(f"✅ Saved labels to: {labels_path}")

# --- 9. VALIDATION & STATS ---
# Simple checks to ensure integrity
print("\n--- Validation Statistics ---")
patch_files = sorted(glob.glob(os.path.join(PATCHES_DIR, "*.pt")))
kpt_files   = sorted(glob.glob(os.path.join(KPTS_DIR, "*.pt")))

print(f"Patch files found: {len(patch_files)}")
print(f"Keypoint files found: {len(kpt_files)}")

if len(patch_files) != len(master_labels):
    print("⚠️ Warning: Number of files does not match label entries!")

# Label Distribution
vals = np.array(list(master_labels.values()), dtype=np.float32)
print(f"Label Mean: {vals.mean():.4f}")
print(f"Fraction Interictal (0.0): {np.mean(vals <= 1e-6):.4f}")
print(f"Fraction Ictal (1.0): {np.mean(vals >= 1.0 - 1e-6):.4f}")

# Optional: Visualize one random clip if running with a display
try:
    if len(master_labels) > 0:
        fid = random.choice(list(master_labels.keys()))
        p_path = os.path.join(PATCHES_DIR, f"{fid}.pt")
        
        if os.path.exists(p_path):
            print(f"\nVisualizing random sample: {fid}")
            p = torch.load(p_path, map_location="cpu").numpy() # (30,15,3,32,32)
            
            JOINT_NAMES = [
                "Nose","L-Eye","R-Eye",
                "R-Shoulder","R-Elbow","R-Wrist",
                "L-Shoulder","L-Elbow","L-Wrist",
                "R-Hip","R-Knee","R-Ankle",
                "L-Hip","L-Knee","L-Ankle"
            ]
            
            t_idx = 0 
            fig, axes = plt.subplots(3, 5, figsize=(12, 7))
            axes = axes.ravel()
            for j in range(15):
                patch = p[t_idx, j].transpose(1,2,0)
                patch = np.clip(patch, 0, 1)
                axes[j].imshow(patch[..., ::-1]) # RGB/BGR
                axes[j].set_title(JOINT_NAMES[j], fontsize=9)
                axes[j].axis("off")
            plt.suptitle(f"{fid} | label={master_labels[fid]:.3f}")
            plt.tight_layout()
            plt.show()
except Exception as e:
    print(f"Visualization skipped: {e}")