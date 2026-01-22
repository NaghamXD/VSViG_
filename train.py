from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict

# --- CONFIGURATION ---
PROCESSED_FOLDER = "processed_data" 
PATH_TO_DATA_FOLDER = PROCESSED_FOLDER 

# --- PATH CONFIGS ---
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

# --- DATASET CLASS (UPDATED) ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        # Define subfolders for patches and keypoints
        self.patches_dir = os.path.join(data_folder, "patches")
        self.kpts_dir = os.path.join(data_folder, "kpts")
        
        # Check if folders exist
        if not os.path.exists(self.patches_dir) or not os.path.exists(self.kpts_dir):
            raise FileNotFoundError(f"Ensure 'patches' and 'kpts' folders exist inside {data_folder}")

        # Load Labels
        with open(label_file, 'rb') as f:
            self._labels = json.load(f)

    def __getitem__(self, idx):
        # Assuming label structure: [filename_string, label_value]
        # Example: ["pat01_Sz1_1780", 1]
        filename_base = str(self._labels[idx][0])
        target = float(self._labels[idx][1])
        
        # Construct paths
        # NOTE: Assuming files end in .pt. If they represent .npy files, change to .npy and use np.load
        patch_path = os.path.join(self.patches_dir, f"{filename_base}.pt")
        kpts_path = os.path.join(self.kpts_dir, f"{filename_base}.pt")
        
        # Load Data
        try:
            data = torch.load(patch_path, map_location='cpu') # Shape: (30, 15, 3, 32, 32)
            kpts = torch.load(kpts_path, map_location='cpu')  # Shape: (30, 15, 2)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find files for ID: {filename_base} at {patch_path}")
        
        # --- PREVIOUS FIX 1: NORMALIZE KEYPOINTS ---
        # Ensure kpts are float
        kpts = kpts.float()
        kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
        kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        
        '''        
        # --- PREVIOUS FIX 2: ADD CONFIDENCE CHANNEL (2 -> 3 CHANNELS) ---
        # Current shape: (30, 15, 2) -> We need: (30, 15, 3)
        confidence = torch.ones((30, 15, 1), dtype=kpts.dtype)
        kpts = torch.cat((kpts, confidence), dim=2)
        
        if self._transform: 
            # Flatten dimensions for transform if needed, then reshape back
            if len(data.shape) == 5:
                B_frames, P, C, H, W = data.shape 
                data = data.view(B_frames*P*C, H, W)
                data = self._transform(data)
                data = data.view(B_frames, P, C, H, W)
         '''    
        sample = {
            'data': data,
            'kpts': kpts 
        }
        return sample, target
    
    def __len__(self):
        return len(self._labels)

def train():
    train_label_path = os.path.join(PROCESSED_FOLDER, 'train_labels.json')
    val_label_path = os.path.join(PROCESSED_FOLDER, 'val_labels.json')
    
    models_to_train = ['Base'] 
    
    for m in models_to_train:
        print(f"Initializing {m} Model Training...")
        
        # 1. Setup Data
        dataset_train = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=train_label_path)
        dataset_val = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=val_label_path)
        
        # REMOVED ChunkBatchSampler
        # Standard DataLoader with shuffle=True for training
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
        
        # 2. Setup Model & Hardware
        if m == 'Base':
            model = VSViG_base() 
        elif m == 'Light':
            model = VSViG_light()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device: {device}")
        
        model = model.to(device)
        MSE = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        # --- RESUME LOGIC ---
        start_epoch = 0
        min_valid_loss = np.inf
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

        if os.path.exists(PATH_TO_LAST_CKPT):
            print(f"Found checkpoint: {PATH_TO_LAST_CKPT}. Resuming...")
            checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            min_valid_loss = checkpoint['min_valid_loss']
            
            if 'history' in checkpoint:
                history = checkpoint['history']

        elif os.path.exists(PATH_TO_BEST_MODEL):
            print(f"Found best_model.pth but no checkpoint. Loading weights only.")
            try:
                model.load_state_dict(torch.load(PATH_TO_BEST_MODEL, map_location=device))
            except:
                print("Could not load best_model.pth weights. Starting fresh.")

        epochs = 200
        
        # 3. Training Loop
        for e in range(start_epoch, epochs):
            train_loss = 0.0
            model.train()
            optimizer.zero_grad()
            print(f'\n=== Epoch: {e+1} ===')

            for batch_idx, (sample, labels) in enumerate(train_loader):
                data = sample['data'].to(device)
                kpts = sample['kpts'].to(device)
                labels = labels.float().to(device)

                outputs = model(data, kpts)
                
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                loss = MSE(outputs.float(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"\rBatch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", end="")
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f'\nTraining Loss: {avg_train_loss:.4f}')

            # 4. Validation & Saving
            if (e+1) % 5 == 0:
                valid_loss = 0.0
                RMSE_loss = 0.0
                model.eval()
                
                with torch.no_grad():
                    for sample, labels in val_loader:
                        data = sample['data'].to(device)
                        kpts = sample['kpts'].to(device)
                        labels = labels.float().to(device)
                        
                        outputs = model(data, kpts)
                        if outputs.dim() > 1 and outputs.shape[1] == 1:
                            outputs = outputs.squeeze(1)
                        
                        loss = MSE(outputs, labels)
                        valid_loss += loss.item()
                        RMSE_loss += torch.sqrt(MSE(outputs, labels)).item() * 100
                
                avg_val_loss = valid_loss / len(val_loader)
                avg_rmse = RMSE_loss / len(val_loader)
                
                history['val_loss'].append(avg_val_loss)
                history['val_rmse'].append(avg_rmse)
                
                print(f' +++ Val Loss: {avg_val_loss:.3f} | Val RMSE: {avg_rmse:.3f} +++')

                if min_valid_loss > valid_loss:
                    print(f'   -> Saving new best model to {PATH_TO_BEST_MODEL}')
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
            
            scheduler.step()

            # 5. Save Checkpoint
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_valid_loss': min_valid_loss,
                'history': history
            }, PATH_TO_LAST_CKPT)

            # 6. Save Logs
            with open(PATH_TO_LOG_FILE, 'w') as f:
                json.dump(history, f, indent=4)
                    
if __name__ == '__main__':
    train()