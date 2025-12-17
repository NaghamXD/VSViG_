import torch
import numpy as np
from VSViG import VSViG_base  # Ensure this import works

# Configuration
MODEL_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_model():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Initialize the architecture
    model = VSViG_base()
    
    # 2. Load the weights
    # map_location ensures it loads on CPU if you trained on GPU but are testing on CPU
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # 3. Set to Evaluation Mode (Crucial!)
    model.to(DEVICE)
    model.eval()
    return model

def predict_new_data(model, data_tensor, kpts_tensor):
    """
    args:
        data_tensor: Shape (Frames, C, H, W) or (1, Frames, C, H, W)
        kpts_tensor: Shape (Frames, Kpts, Channels)
    """
    # Ensure batch dimension exists (Batch Size = 1)
    if data_tensor.dim() == 4:
        data_tensor = data_tensor.unsqueeze(0)
    if kpts_tensor.dim() == 2: # (Frames, Kpts) -> (1, Frames, Kpts)
         # NOTE: Your training logic added a 3rd channel (confidence) manually.
         # You must do the same here if your raw input only has 2 channels.
        pass 
        
    data_tensor = data_tensor.to(DEVICE)
    kpts_tensor = kpts_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(data_tensor, kpts_tensor)
        
        # Squeeze if necessary
        if output.dim() > 1 and output.shape[1] == 1:
            output = output.squeeze(1)
            
    return output.item()

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # 1. Load Model
    model = load_best_model()
    
    # 2. Simulate Fake Data (Replace this with your real data loading)
    # Dimensions based on your dataset: (30 frames, 15 patches, 3 channels, 32 height, 32 width)
    # Flattened view used in transform: (30, 15, 3, 32, 32)
    fake_data = torch.randn(30, 15, 3, 32, 32) 
    
    # Keypoints: (30 frames, 15 points, 3 channels)
    fake_kpts = torch.randn(30, 15, 3) 

    # 3. Get Prediction
    result = predict_new_data(model, fake_data, fake_kpts)
    print(f"Predicted Score: {result:.4f}")