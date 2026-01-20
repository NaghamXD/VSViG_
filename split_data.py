import json
import random
import os
from collections import defaultdict

# CONFIGURATION
OUTPUT_FOLDER = 'processed_data'
LABEL_FILE = os.path.join(OUTPUT_FOLDER, 'labels.json')

def print_duration_stats(name, count):
    """Calculates hours based on 5-second clips (VSViG Paper)"""
    seconds = count * 5
    hours = seconds / 3600
    print(f"  - {name}: {count} clips = {hours:.4f} hours")

def main():
    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: {LABEL_FILE} not found. Run preprocessing first.")
        return

    with open(LABEL_FILE, 'r') as f:
        all_labels = json.load(f)
        
    # Check if ID exists in labels
    if len(all_labels[0]) < 3:
        print("❌ Error: labels.json missing Seizure IDs.")
        print("   Update preprocessing.py to save: [clip_idx, label, unique_id]")
        return
    
    # 1. Group Data by Class and Seizure ID
    # Structure: seizure_map["Pat01_Sz1"] = {'transition': [], 'ictal': []}
    seizure_map = defaultdict(lambda: {'transition': [], 'ictal': []})
    interictal_pool = []

    for item in all_labels:
        clip_idx, label, sz_id = item
        
        if label == 0.0:
            interictal_pool.append(item)
        elif 0.0 < label < 1.0:
            seizure_map[sz_id]['transition'].append(item)
        elif label >= 1.0:
            seizure_map[sz_id]['ictal'].append(item)
   
    # 2. Split "From Each Seizure" (Paper Rule)
    # "randomly extract 20% clips from both transition and ictal periods from each seizure"
    
    val_unhealthy = []
    train_unhealthy = []
    
    random.seed(42) # Reproducibility

    print("-" * 30)
    print("🔄 Processing Split per Seizure...")
    
    for sz_id, data in seizure_map.items():
        trans_clips = data['transition']
        ictal_clips = data['ictal']
        
        # Shuffle specifically for this seizure
        random.shuffle(trans_clips)
        random.shuffle(ictal_clips)
        
        # Calculate 20% split
        n_val_trans = int(len(trans_clips) * 0.20)
        n_val_ictal = int(len(ictal_clips) * 0.20)
        
        # Split Transition
        val_unhealthy.extend(trans_clips[:n_val_trans])
        train_unhealthy.extend(trans_clips[n_val_trans:])
        
        # Split Ictal
        val_unhealthy.extend(ictal_clips[:n_val_ictal])
        train_unhealthy.extend(ictal_clips[n_val_ictal:])
        
        # (Optional Debug)
        print(f"   {sz_id}: Val={n_val_trans+n_val_ictal}, Train={len(trans_clips)+len(ictal_clips) - (n_val_trans+n_val_ictal)}")

    # 3. Handle Interictal (Healthy) Balancing
    # Paper: "equivalent number of clips... extracted from the interictal period for validation"
    
    target_val_healthy_count = len(val_unhealthy)
    
    random.shuffle(interictal_pool)
    
    # Extract equivalent number for validation
    val_healthy = interictal_pool[:target_val_healthy_count]
    
    # The rest go to training ("the rest clips are for model training")
    train_healthy = interictal_pool[target_val_healthy_count:]

    # 4. Final Sets
    val_set = val_unhealthy + val_healthy
    train_set = train_unhealthy + train_healthy
    
    # Verify we didn't lose data
    total_processed = len(val_set) + len(train_set)
    assert total_processed == len(all_labels), "Mismatch in total clip count!"

    # 5. Save
    train_path = os.path.join(OUTPUT_FOLDER, 'train_labels.json')
    val_path = os.path.join(OUTPUT_FOLDER, 'val_labels.json')

    with open(train_path, 'w') as f:
        json.dump(train_set, f)
    with open(val_path, 'w') as f:
        json.dump(val_set, f)

    # 6. Report Stats
    print("-" * 30)
    print("📊 FINAL DATASET REPORT (Matches VSViG Strategy)")
    print("-" * 30)
    print_duration_stats("Interictal (Train Pool)", len(train_healthy))
    print_duration_stats("Transition (Train Pool)", len([x for x in train_unhealthy if 0<x[1]<1]))
    print_duration_stats("Ictal (Train Pool)", len([x for x in train_unhealthy if x[1]>=1]))
    print("-" * 30)
    print(f"✅ Training Set:   {len(train_set)} clips")
    print(f"✅ Validation Set: {len(val_set)} clips")
    print(f"   - Validation Balance: {len(val_healthy)} Healthy vs {len(val_unhealthy)} Unhealthy (1:1)")
    print(f"   - Strategy: 20% taken from EACH seizure individually.")
    print("-" * 30)

if __name__ == '__main__':
    main()