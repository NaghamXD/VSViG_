import json
import random
import os

# CONFIGURATION
OUTPUT_FOLDER = 'processed_data'
LABEL_FILE = os.path.join(OUTPUT_FOLDER, 'labels.json')

def main():
    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: {LABEL_FILE} not found. Run preprocessing first.")
        return

    with open(LABEL_FILE, 'r') as f:
        all_labels = json.load(f)
        # Structure: [[clip_id, label_value], [clip_id, label_value], ...]

    # 1. Separate by Class
    # Interictal (Healthy) = 0.0
    # Transition/Ictal (Seizure) > 0.0
    healthy_clips = [item for item in all_labels if item[1] == 0.0]
    transition_clips = [item for item in all_labels if (item[1] > 0.0 and item[1] < 1.0)]
    seizure_clips = [item for item in all_labels if item[1] >= 1.0]

    print(f"Total Clips: {len(all_labels)}")
    print(f"  - Healthy: {len(healthy_clips)}")
    print(f"  - Seizure: {len(seizure_clips)}")
    print(f"  - Transition: {len(transition_clips)}")

    # 2. Calculate Split Sizes (20% Validation)
    num_seizure_val = int(len(seizure_clips) * 0.2)
    num_transition_val = int(len(transition_clips) * 0.2)

    # Paper says: "equivalent number of clips... extracted from interictal"
    # So if we take 100 seizure clips for val, we take 100 healthy clips for val.
    num_healthy_val = num_seizure_val + num_transition_val

    # 3. Random Shuffle & Split
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(seizure_clips)
    random.shuffle(transition_clips)
    random.shuffle(healthy_clips)

    # Create Validation Set
    val_seizure = seizure_clips[:num_seizure_val]
    val_transition = transition_clips[:num_transition_val]
    val_healthy = healthy_clips[:num_healthy_val]
    val_set = val_seizure + val_healthy + val_transition

    # Create Training Set (The rest)
    train_seizure = seizure_clips[num_seizure_val:]
    train_transition = transition_clips[num_transition_val:]
    train_healthy = healthy_clips[num_healthy_val:]
    train_set = train_seizure + train_healthy + train_transition

    # 4. Save New Label Files
    train_path = os.path.join(OUTPUT_FOLDER, 'train_labels.json')
    val_path = os.path.join(OUTPUT_FOLDER, 'val_labels.json')

    with open(train_path, 'w') as f:
        json.dump(train_set, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_set, f)

    print("-" * 30)
    print(f"✅ Data Splitting Complete")
    print(f"Train Set: {len(train_set)} clips ({len(train_seizure)} Sz, {len(train_healthy)} Healthy)")
    print(f"Val Set:   {len(val_set)} clips ({len(val_seizure)} Sz, {len(val_healthy)} Healthy)")
    print(f"Saved to: {train_path} and {val_path}")

if __name__ == '__main__':
    main()