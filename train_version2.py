from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os

# --- CONFIGURATION ---
# Point to your processed folder
PROCESSED_FOLDER = "processed_data" 

# These variables from original script are repurposed or ignored
# We pass specific paths to the dataset class now
PATH_TO_DATA_FOLDER = PROCESSED_FOLDER 

# Where to save the best model during training
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
PATH_TO_SAVE_MODEL = "checkpoints/best_model.pth"

# --- LAZY DATASET CLASS ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        # 1. Load Labels (Train or Val list)
        with open(label_file, 'rb') as f:
            self._labels = json.load(f) # [[id, label], [id, label]...]
            
        # 2. Load the Map (Created by create_map.py)
        map_path = os.path.join(data_folder, 'chunk_map.json')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Chunk map not found at {map_path}. Run create_map.py!")
            
        with open(map_path, 'r') as f:
            self._chunk_map = json.load(f)
            
        # 3. Simple Cache
        self.last_chunk_name = None
        self.last_chunk_data = None
        self.last_chunk_kpts = None

    def __getitem__(self, idx):
        target = float(self._labels[idx][1])
        global_id = str(self._labels[idx][0]) 
        
        # Find where data lives
        if global_id not in self._chunk_map:
            # Fallback/Debug: Print error and return zero? Better to crash so you know.
            raise IndexError(f"Global ID {global_id} not found in chunk_map!")
            
        filename, local_idx = self._chunk_map[global_id]
        
        # Load Chunk (Lazy)
        if filename != self.last_chunk_name:
            self.last_chunk_name = filename
            data_path = os.path.join(self._folder, filename)
            kpts_path = os.path.join(self._folder, filename.replace('chunk_data', 'chunk_kpts'))
            
            # Load to CPU
            self.last_chunk_data = torch.load(data_path, map_location='cpu')
            self.last_chunk_kpts = torch.load(kpts_path, map_location='cpu')
            
        # Extract Clip
        data = self.last_chunk_data[local_idx] # (30, 15, 3, 32, 32)
        kpts = self.last_chunk_kpts[local_idx] # (30, 15, 2)
        
        # --- NORMALIZE KEYPOINTS (Raw Pixels -> 0.0-1.0) ---
        kpts = kpts.float()
        kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
        kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        
        # Optional Transform (if any)
        if self._transform: 
            B, P, C, H, W = data.shape 
            data = data.view(B*P*C, H, W)
            data = self._transform(data)
            data = data.view(B, P, C, H, W)
            
        sample = {
            'data': data,
            'kpts': kpts
        }
        return sample, target
    
    def __len__(self):
        return len(self._labels)

def train():
    # Load specific split files
    train_label_path = os.path.join(PROCESSED_FOLDER, 'train_labels.json')
    val_label_path = os.path.join(PROCESSED_FOLDER, 'val_labels.json')
    
    models_to_train = ['Base'] # Or ['Base', 'Light']
    
    for m in models_to_train:
        print(f"Initializing {m} Model Training...")
        
        dataset_train = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=train_label_path, transform=None)
        dataset_val = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=val_label_path, transform=None)
        
        # num_workers=0 is safer for debugging on Mac, increase to 2 or 4 if stable
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=0)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
        
        MSE = nn.MSELoss()
        epochs = 200
        min_valid_loss = np.inf
        
        if m == 'Base':
            # Pass path to dynamic order explicitly if needed by your VSViG.py modification
            # Assuming VSViG_base() handles loading internally using the global var or updated path
            model = VSViG_base() 
        elif m == 'Light':
            model = VSViG_light()
            
        # Optional: Load Pretrained Weights
        if os.path.exists('VSViG-base.pth'):
            print("Loading pretrained weights...")
            model.load_state_dict(torch.load('VSViG-base.pth', map_location='cpu'))
            
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon) acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU")
            
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        for e in range(epochs):
            train_loss = 0.0
            model.train()
            optimizer.zero_grad()
            print(f'\n=== Epoch: {e+1} ===')

            for batch_idx, (sample, labels) in enumerate(train_loader):
                data = sample['data'].to(device)
                kpts = sample['kpts'].to(device)
                labels = labels.float().to(device)

                outputs = model(data, kpts)
                
                # Squeeze outputs to match label shape if needed (B, 1) -> (B)
                outputs = outputs.squeeze() 
                
                loss = MSE(outputs.float(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() # Move zero_grad here standard practice
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"\rBatch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", end="")
            
            print(f'\nTraining Loss: {train_loss / len(train_loader):.4f}')

            # Validation
            if (e+1) % 5 == 0:
                valid_loss = 0.0
                RMSE_loss = 0.0
                model.eval()
                
                with torch.no_grad(): # Save memory
                    for sample, labels in val_loader:
                        data = sample['data'].to(device)
                        kpts = sample['kpts'].to(device)
                        labels = labels.float().to(device)
                        
                        outputs = model(data, kpts)
                        outputs = outputs.squeeze()
                        
                        loss = MSE(outputs, labels)
                        valid_loss += loss.item()
                        RMSE_loss += torch.sqrt(MSE(outputs, labels)).item() * 100
                
                avg_val_loss = valid_loss / len(val_loader)
                avg_rmse = RMSE_loss / len(val_loader)
                
                print(f' +++ Val Loss: {avg_val_loss:.3f} | Val RMSE: {avg_rmse:.3f} +++')

                if min_valid_loss > valid_loss:
                    print(f'   -> Saving new best model to {PATH_TO_SAVE_MODEL}')
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), PATH_TO_SAVE_MODEL)
            
            scheduler.step()
                    
if __name__ == '__main__':
    train()