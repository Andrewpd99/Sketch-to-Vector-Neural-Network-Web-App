import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import psutil
import time
import gc

# Add the project root directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model import SketchVectorizer

# Parameters
batch_size = 128
output_dim = 256
max_memory_usage = 0.5  # Set maximum memory usage as a fraction (e.g., 50%)
chunk_size = 1000  # Number of files to process at once
num_workers = min(mp.cpu_count(), 8)  # Number of CPU threads to use

class SketchDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.preloaded_data = []
        self._preload_data()

    def _process_chunk(self, chunk_files):
        chunk_data = []
        for file in chunk_files:
            with np.load(file, mmap_mode='r') as npz_data:
                chunk_data.extend(npz_data['images'])
        return chunk_data

    def _preload_data(self):
        start_time = time.time()
        total_files = len(self.npz_files)
        memory_limit = max_memory_usage * psutil.virtual_memory().total / (1024 ** 2)  # MB

        # Process chunks of files to manage memory usage
        for i in range(0, total_files, chunk_size):
            chunk_files = self.npz_files[i:i + chunk_size]
            chunk_data = self._process_chunk(chunk_files)

            # Check memory usage
            while self._get_memory_usage() > memory_limit:
                print("Memory usage exceeds limit. Waiting for memory to free up...")
                time.sleep(5)
                gc.collect()  # Attempt to free up memory

            self.preloaded_data.extend(chunk_data)

            # Report progress
            if (i + chunk_size) % 5000 == 0:
                elapsed_time = time.time() - start_time
                elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
                print(f"Loaded {(i + chunk_size)} files in {int(elapsed_minutes)}m {int(elapsed_seconds)}s")

        total_time = time.time() - start_time
        total_minutes, total_seconds = divmod(total_time, 60)
        print(f"Completed loading all files in {int(total_minutes)}m {int(total_seconds)}s")

    def _get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)  # MB

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, idx):
        image = self.preloaded_data[idx]
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

def setup():
    # Load processed data
    data_folder = 'data/processed/'
    npz_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npz')]

    # Create dataset and dataloader
    dataset = SketchDataset(npz_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Initialize model
    model = SketchVectorizer(output_dim=output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return dataloader, model, device

if __name__ == "__main__":
    setup()
