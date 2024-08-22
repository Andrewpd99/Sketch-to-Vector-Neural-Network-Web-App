import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from setup import setup  # Import the setup function

# Parameters
learning_rate = 0.001
num_epochs = 20
accumulation_steps = 2
checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoint.pth'))

def train():
    # Setup
    dataloader, model, device = setup()
    
    # Initialize optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")

    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, images in enumerate(dataloader):
            batch_start_time = time.time()
            images = images.to(device)

            # Gradient accumulation
            optimizer.zero_grad()

            with autocast():  # Enable mixed precision
                outputs = model(images)
                target_vectors = torch.randn(images.size(0), 256).to(device)
                loss = criterion(outputs, target_vectors)
                loss = loss / accumulation_steps

            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            batch_end_time = time.time()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            print(f"Batch Time: {batch_end_time - batch_start_time:.4f} seconds")

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        epoch_elapsed_minutes, epoch_elapsed_seconds = divmod(epoch_elapsed_time, 60)
        epoch_elapsed_time_str = f"{int(epoch_elapsed_minutes)}m {int(epoch_elapsed_seconds)}s"
        print(f'Epoch [{epoch+1}/{num_epochs}] completed.')
        print(f'Loss: {running_loss/len(dataloader):.4f}')
        print(f"Epoch Time: {epoch_elapsed_time_str}")

        # Print memory usage after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Free unused memory
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'sketch_vectorizer.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
