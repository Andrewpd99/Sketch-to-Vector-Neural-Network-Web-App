import os
import sys
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset

# Add the project root directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model import SketchVectorizer

# Parameters
batch_size = 128
output_dim = 256

class SketchDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.cumulative_sizes = []
        self.preload_data()

    def preload_data(self):
        cumulative_size = 0
        start_time = time.time()
        report_interval = 5000
        for i, file in enumerate(self.npz_files):
            with np.load(file) as npz_data:
                num_images = len(npz_data['images'])
                cumulative_size += num_images
                self.cumulative_sizes.append(cumulative_size)

                # Report progress
                if (i + 1) % report_interval == 0:
                    elapsed_time = time.time() - start_time
                    elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
                    print(f"Loaded {(i + 1) * report_interval} files in {int(elapsed_minutes)}m {int(elapsed_seconds)}s")

        # Final report
        total_time = time.time() - start_time
        total_minutes, total_seconds = divmod(total_time, 60)
        print(f"Completed loading all files in {int(total_minutes)}m {int(total_seconds)}s")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        idx_within_file = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
        with np.load(self.npz_files[file_idx]) as npz_data:
            image = npz_data['images'][idx_within_file]
            return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

def setup():
    # Load processed data
    data_folder = 'data/processed/'
    npz_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npz')]

    # Create dataset and dataloader
    dataset = SketchDataset(npz_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, persistent_workers=True)

    # Initialize model
    model = SketchVectorizer(output_dim=output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return dataloader, model, device

if __name__ == "__main__":
    setup()