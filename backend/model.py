import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchCNN(nn.Module):
    def __init__(self, num_classes=345):
        super(SketchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        
        self._to_linear = None
        self._get_flattened_size((1, 100, 100))  # Use 100x100 as input size
        self.fc1 = nn.Linear(self._to_linear, 512)  # Increase the size of the hidden layer
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            print(f'Feature map size after conv layers: {x.shape}')  # Debug print
            self._to_linear = x.numel()  # Flattened size
            print(f'Flattened size: {self._to_linear}')  # Debug print

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(f'Shape after conv layers: {x.shape}')  # Debug print
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print(f'Shape after flattening: {x.shape}')  # Debug print
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
