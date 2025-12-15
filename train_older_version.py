from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os

PATH_TO_DATA = 'processed_data/'
PATH_TO_MODEL = 'VSViG-model'
PATH_TO_DYNAMIC_PARTITIONS = 'dy_point_order.pt'

class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        # 1. Load Labels
        with open(label_file, 'rb') as f:
            self._labels = json.load(f) # [[id, label], [id, label]...]
            
        # 2. Load the Map (Created by create_map.py)
        map_path = os.path.join(data_folder, 'chunk_map.json')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Chunk map not found at {map_path}. Run create_map.py!")
            
        with open(map_path, 'r') as f:
            self._chunk_map = json.load(f)
            
        # 3. Simple Cache (Optional optimization)
        # Keeps the last loaded chunk in memory to speed up sequential access
        self.last_chunk_name = None
        self.last_chunk_data = None
        self.last_chunk_kpts = None

    def __getitem__(self, idx):
        # A. Get Target and Global ID
        target = float(self._labels[idx][1])
        global_id = str(self._labels[idx][0]) # Ensure string for JSON lookup
        
        # B. Find where the data lives
        if global_id not in self._chunk_map:
            raise IndexError(f"Global ID {global_id} not found in chunk_map!")
            
        filename, local_idx = self._chunk_map[global_id]
        
        # C. Load the Chunk (Lazy Loading)
        # Optimization: Only load from disk if it's different from the last one we opened
        if filename != self.last_chunk_name:
            self.last_chunk_name = filename
            
            # Construct paths
            data_path = os.path.join(self._folder, filename)
            kpts_path = os.path.join(self._folder, filename.replace('chunk_data', 'chunk_kpts'))
            
            # Load to CPU (Model will move to GPU later)
            self.last_chunk_data = torch.load(data_path, map_location='cpu')
            self.last_chunk_kpts = torch.load(kpts_path, map_location='cpu')
            
        # D. Extract specific clip
        data = self._chunk_data[local_idx] # (30, 15, 3, 32, 32)
        kpts = self._chunk_kpts[local_idx] # (30, 15, 2)
        
        # E. Logic from original script (Reordering/Transform)
        # Note: My preprocessing script ALREADY filtered/ordered joints (18->15), 
        # so you likely DON'T need the reordering logic unless your model expects 
        # a different internal permutation. Assuming standard:
        
        # data = data.squeeze(0) # Not needed if shape is (30,...)

        '''# --- NORMALIZE KEYPOINTS (New Step) ---
        # Normalize to range [0, 1] assuming 1920x1080 resolution
        kpts = kpts.float() # Ensure float for division
        kpts[:, :, 0] = kpts[:, :, 0] / 1920.0  # X coords
        kpts[:, :, 1] = kpts[:, :, 1] / 1080.0  # Y coords'''
        
        if self._transform:
            B, P, C, H, W = data.shape # 30, 15, 3, 32, 32
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
    dy_point_order = torch.load('PATH_TO_DYNAMIC_PARTITIONS')
    data_path = PATH_TO_DATA # Inputs: Batches, Frames, Points, Channles, Height, Width (B,30,15,3,32,32)
    models = ['base', 'light']
    train_label_path = 'processed_data/train_labels.json'
    val_label_path = 'processed_data/val_labels.json'


    for m in models:
        dataset_train = vsvig_dataset(data_folder=data_path, label_file=train_label_path, transform=None)
        dataset_val = vsvig_dataset(data_folder=data_path, label_file=val_label_path, transform=None)
        #dataset_train = vsvig_dataset(data_folder=data_path, label_file=label_path, transform=None)
        #dataset_val = vsvig_dataset(data_folder=data_path, label_file=label_path, transform=None)
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True,num_workers=4)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True,num_workers=4)
        
        # criterion = nn.BCEWithLogitsLoss()
        MSE = nn.MSELoss()
        epochs = 200
        min_valid_loss = np.inf
        if m == 'Base':
            model = VSViG_base()
        elif m == 'Light':
            model = VSViG_light()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        train_loss_stack = []
        for e in range(epochs):
            train_loss = 0.0
            
            model.train()
            optimizer.zero_grad()
            print(f'===================================\n Running Epoch: {e+1} \n===================================')

            for sample, labels in train_loader:
                data = sample['data']
                kpts = sample['kpts']

                if torch.cuda.is_available():
                    data, labels, kpts = data.cuda(), labels.cuda(), kpts.cuda()
                outputs = model(data,kpts)
                # print(outputs)
                loss = MSE(outputs.float(),labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_loss_stack.append(loss.item())
            print(f'Training Loss: {train_loss:.3f}')

            if (e+1)%5 == 0:
                valid_loss = 0.0
                RMSE_loss = 0.0
                _iter = 0
                model.eval()

                for sample, labels in val_loader:
                    data = sample['data']
                    kpts = sample['kpts']
                    if torch.cuda.is_available():
                        data, labels, kpts = data.cuda(), labels.cuda(), kpts.cuda()
                    outputs = model(data,kpts)
                    loss = MSE(outputs,labels)
                    valid_loss += loss.item()
                    RMSE_loss += torch.sqrt(MSE(outputs,labels)).item()*100
                    _iter += 1
                print(f' +++++++++++++++++++++++++++++++++++\n Val Loss: {valid_loss:.3f} \t Val RMSE: {RMSE_loss/_iter:.3f} \n +++++++++++++++++++++++++++++++++++')

                if min_valid_loss > valid_loss:
                    print(f'save the model \n +++++++++++++++++++++++++++++++++++')
                    min_valid_loss = valid_loss
                    save_model_path = PATH_TO_MODEL
                    torch.save(model.state_dict(), save_model_path)
            scheduler.step()
                    
if __name__ == '__main__':
    train()
