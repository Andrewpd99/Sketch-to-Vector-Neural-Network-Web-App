import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import psutil
import time
import concurrent.futures

# Add the project root directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model import SketchVectorizer

# Parameters
batch_size = 128
chunk_size = 5000  # Number of files to process at a time

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def process_batch(dataloader):
    model = SketchVectorizer().to(device)
    model.train()
    for data in dataloader:
        data = data.to(device, non_blocking=True)
        output = model(data)

def setup():
    data_folder = 'data/processed/'
    model = SketchVectorizer().to(device)  # Initialize without output_dim

    # Track memory and time
    start_time = time.time()

    # Process the first chunk to get an initial dataloader
    batch_files = next(load_files_in_chunks(data_folder, chunk_size))
    dataset = SketchDataset(batch_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count(), pin_memory=True)

    # Use threads for parallel processing of batches
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_batch, dataloader) for _ in range(mp.cpu_count())]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Measure time and memory usage
    mem = psutil.virtual_memory()
    print(f"Memory Usage: {mem.percent}%")
    print(f"Time taken for the first chunk of files: {time.time() - start_time} seconds")

    # Return the model, dataloader, and device for training
    return model, dataloader, device

if __name__ == "__main__":
    setup()
