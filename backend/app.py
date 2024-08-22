import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Simple dataset and model for testing
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3, 224, 224)  # Example with 3x224x224 images
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*224*224, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def test_batch_size(batch_size):
    dataset = DummyDataset(1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DummyModel().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        for data, target in dataloader:
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Print memory usage
            print(f"Batch Size: {batch_size}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            break  # Remove this line to test with more batches

    except RuntimeError as e:
        print(f"Failed with batch size {batch_size}: {e}")

if __name__ == '__main__':
    # Test different batch sizes
    batch_sizes = [64, 128, 256]
    for bs in batch_sizes:
        test_batch_size(bs)
