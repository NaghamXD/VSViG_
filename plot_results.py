import json
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_PATH = "checkpoints/training_log.json"

def plot_history():
    if not os.path.exists(LOG_PATH):
        print(f"File {LOG_PATH} not found. Train the model first.")
        return

    with open(LOG_PATH, 'r') as f:
        history = json.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    # Create X-axis steps
    epochs = range(1, len(train_loss) + 1)
    # Validation runs every 5 epochs (based on your train.py logic)
    val_epochs = [i * 5 for i in range(1, len(val_loss) + 1)]

    plt.figure(figsize=(10, 5))
    
    # Plot Training Loss
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.6)
    
    # Plot Validation Loss
    if val_loss:
        plt.plot(val_epochs, val_loss, label='Validation Loss', color='red', marker='o')
        
        # Find best epoch
        min_val_loss = min(val_loss)
        best_val_idx = val_loss.index(min_val_loss)
        best_epoch = val_epochs[best_val_idx]
        
        plt.title(f"Train vs Val Loss (Best Epoch: {best_epoch}, Loss: {min_val_loss:.4f})")
        
        # Highlight best point
        plt.scatter(best_epoch, min_val_loss, s=100, c='green', zorder=5, label='Best Model')
        plt.annotate(f'Best: {min_val_loss:.3f}', (best_epoch, min_val_loss), xytext=(0, 10), textcoords='offset points')
    else:
        plt.title("Training Loss (No Validation Data Yet)")

    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_history()