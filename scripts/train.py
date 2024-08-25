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
initial_checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'initial_setup_checkpoint.pth'))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def train():
    # Initialize the model and dataloader (no optimizer yet)
    model, dataloader, device = setup()  # Use setup to initialize model, dataloader, and device

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    start_epoch = 0

    # Check if checkpoint exists to resume from it
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")

    elif os.path.exists(initial_checkpoint_path):
        print(f"Loading initial setup state from {initial_checkpoint_path}...")
        initial_setup = torch.load(initial_checkpoint_path)
        model.load_state_dict(initial_setup['model_state_dict'])
        print(f"Starting training from initial setup state...")

    model = model.to(device)  # Move model to the device

    criterion = torch.nn.MSELoss()

    try:
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()

            for i, images in enumerate(dataloader):
                batch_start_time = time.time()
                images = images.to(device, non_blocking=True)

                # Gradient accumulation
                optimizer.zero_grad()

                with autocast():  # Enable mixed precision
                    outputs = model(images)
                    target_vectors = torch.randn(images.size(0), 256, device=device)
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
                if (i + 1) % 10 == 0:  # Print every 10 batches
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                    print(f"Batch Time: {batch_end_time - batch_start_time:.4f} seconds")

                # Save checkpoint after each batch
                save_checkpoint(model, optimizer, epoch, running_loss, checkpoint_path)

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

            # Save checkpoint every epoch
            save_checkpoint(model, optimizer, epoch, running_loss, checkpoint_path)

    except Exception as e:
        print(f"Error occurred: {e}")
        save_checkpoint(model, optimizer, epoch, running_loss, checkpoint_path)
        raise

    # Save the final model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'sketch_vectorizer.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
