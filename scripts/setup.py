import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from functools import partial
import gc
import psutil
import time

# Add the project root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model import SketchVectorizer

# Parameters
batch_size = 128
output_dim = 256
max_memory_usage = 0.5  # Set maximum memory usage as a fraction (e.g., 50%)
chunk_size = 1000  # Number of files to process at once

class SketchDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.cumulative_sizes = []
        self.preloaded_data = []
        self.preload_data()

    def process_chunk(self, chunk_files):
        chunk_data = []
        cumulative_size = 0
        for file in chunk_files:
            with np.load(file) as npz_data:
                num_images = len(npz_data['images'])
                chunk_data.append(npz_data['images'])
                cumulative_size += num_images
        return chunk_data, cumulative_size

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

    def preload_data(self):
        cumulative_size = 0
        start_time = time.time()
        report_interval = 5000
        total_files = len(self.npz_files)
        memory_limit = max_memory_usage * psutil.virtual_memory().total / (1024 ** 2)  # Convert bytes to MB

        # Use multiprocessing to process chunks
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunks = [self.npz_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
            for i, chunk_files in enumerate(chunks):
                chunk_data, chunk_cumulative_size = pool.apply(partial(self.process_chunk), args=(chunk_files,))

                # Check memory usage
                while self.get_memory_usage() > memory_limit:
                    print(f"Memory usage exceeds limit. Waiting for memory to free up...")
                    time.sleep(5)
                    gc.collect()  # Attempt to free up memory

                self.preloaded_data.extend(chunk_data)
                cumulative_size += chunk_cumulative_size
                self.cumulative_sizes.append(cumulative_size)

                # Report progress
                if (i + 1) * chunk_size >= report_interval:
                    elapsed_time = time.time() - start_time
                    elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
                    print(f"Loaded {(i + 1) * chunk_size} files in {int(elapsed_minutes)}m {int(elapsed_seconds)}s")

        total_time = time.time() - start_time
        total_minutes, total_seconds = divmod(total_time, 60)
        print(f"Completed loading all files in {int(total_minutes)}m {int(total_seconds)}s")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        idx_within_file = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
        image = self.preloaded_data[file_idx][idx_within_file]
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