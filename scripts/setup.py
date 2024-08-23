import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import psutil
import time

# Add the project root directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model import SketchVectorizer

# Parameters
batch_size = 128
output_dim = 256
num_workers = min(mp.cpu_count(), 8)  # Number of CPU threads to use
chunk_size = 5000  # Number of files to process at a time

class SketchDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self._load_first_file()

    def _load_first_file(self):
        if self.npz_files:
            self._load_file(self.npz_files[0])
    
    def _load_file(self, file):
        self.data = np.load(file, mmap_mode='r')['images']
        self.current_file_idx = 0
        self.file_index = 0

    def __len__(self):
        total_images = 0
        for file in self.npz_files:
            with np.load(file, mmap_mode='r') as npz_data:
                total_images += len(npz_data['images'])
        return total_images

    def __getitem__(self, idx):
        cumulative_size = 0
        for file_idx, file in enumerate(self.npz_files):
            with np.load(file, mmap_mode='r') as npz_data:
                num_images = len(npz_data['images'])
                if cumulative_size + num_images > idx:
                    image_idx = idx - cumulative_size
                    if file_idx != self.current_file_idx:
                        self._load_file(file)
                    image = self.data[image_idx]
                    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                cumulative_size += num_images
        raise IndexError("Index out of range.")

def load_files_in_chunks(folder, chunk_size):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
    for i in range(0, len(all_files), chunk_size):
        yield all_files[i:i + chunk_size]

def setup():
    data_folder = 'data/processed/'
    dataloader = None
    model = SketchVectorizer(output_dim=output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track memory and time
    start_time = time.time()

    for batch_files in load_files_in_chunks(data_folder, chunk_size):
        dataset = SketchDataset(batch_files)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        # Measure time and memory usage
        mem = psutil.virtual_memory()
        print(f"Memory Usage: {mem.percent}%")
        print(f"Time taken for {len(batch_files)} files: {time.time() - start_time} seconds")
        start_time = time.time()

        # Here you could process dataloader for a sample or initial checks if needed

    return dataloader, model, device

if __name__ == "__main__":
    setup()
