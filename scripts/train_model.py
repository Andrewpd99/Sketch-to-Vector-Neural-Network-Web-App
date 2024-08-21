import sys
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
from backend.model import SketchCNN
from sklearn.preprocessing import LabelEncoder



# Load preprocessed data
data = np.load('data/processed/processed_data.npz')
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']

# Encode string labels to integers
label_encoder = LabelEncoder()
all_labels = np.concatenate((y_train, y_val))
label_encoder.fit(all_labels)

y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)

print(f'Label Encoder classes: {label_encoder.classes_}')

# Normalize pixel values and convert numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train / 255.0, dtype=torch.float32).unsqueeze(1)  # Normalize and add channel dimension
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val / 255.0, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.long)

# Print shapes
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_val shape: {x_val.shape}')
print(f'y_val shape: {y_val.shape}')

# Create data loaders
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model with the correct number of classes
num_classes = len(label_encoder.classes_)
model = SketchCNN(num_classes=num_classes)

# Print model summary
print(model)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Adjust learning rate if needed
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler

num_epochs = 20  # Increased number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    # Inside your validation loop
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            
            # Select a single image for inspection
            img_input = inputs[0].squeeze().cpu()  # Remove channel dimension if grayscale
            img_output = outputs[0].squeeze().cpu()  # Remove channel dimension if grayscale

            # Display the images
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_input, cmap='gray')
            ax[0].set_title('Input Sketch')
            ax[0].axis('off')  # Hide axes

            ax[1].imshow(img_output, cmap='gray')
            ax[1].set_title('Generated Image')
            ax[1].axis('off')  # Hide axes

            plt.show()
            break  # Remove this break to process more images

        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    # Step the learning rate scheduler
    scheduler.step()

torch.save(model.state_dict(), 'models/sketch_cnn.pth')
print("Model saved to models/sketch_cnn.pth")