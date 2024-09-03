import time
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor


"""
This tests the CPU and GPU compatibility together
"""


# Parameters
batch_size = 128
num_batches_to_test = 10
num_workers = 8  # Adjust based on your system

class DummyDataset(Dataset):
    def __init__(self, size=10000, image_size=(256, 256)):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate dummy data
        image = torch.randn(*self.image_size)
        return image

def test_dataloader_speed():
    # Initialize dataset and dataloader
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Measure dataloader speed
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_test:
            break
        batch_start_time = time.time()
        batch = batch.cuda(non_blocking=True)  # Move to GPU
        print(f"Batch [{i+1}/{num_batches_to_test}] loaded in {time.time() - batch_start_time:.4f} seconds")

    total_time = time.time() - start_time
    avg_time_per_batch = total_time / num_batches_to_test
    print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
    print(f"Total time for {num_batches_to_test} batches: {total_time:.4f} seconds")

if __name__ == "__main__":
    test_dataloader_speed()
