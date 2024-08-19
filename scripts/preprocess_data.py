import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_data(numpy_dir, output_file, target_shape):
    data = []
    labels = []
    
    for filename in os.listdir(numpy_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(numpy_dir, filename)
            if os.path.exists(file_path):
                try:
                    np_data = np.load(file_path)
                    # Check the current shape of np_data
                    current_shape = np_data.shape
                    # Determine the shape to pad or truncate to
                    if len(current_shape) < len(target_shape):
                        current_shape = current_shape + (1,) * (len(target_shape) - len(current_shape))
                    
                    # Pad or truncate the data
                    if current_shape != target_shape:
                        if np_data.shape < target_shape:
                            # Pad with zeros
                            pad_width = [(0, max(0, t - s)) for s, t in zip(current_shape, target_shape)]
                            np_data = np.pad(np_data, pad_width, mode='constant', constant_values=0)
                        else:
                            # Truncate
                            slices = [slice(0, t) for t in target_shape]
                            np_data = np.array(np_data[tuple(slices)])
                    
                    data.append(np_data)
                    labels.append(filename.replace('.npy', ''))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File {file_path} does not exist, skipping...")
    
    if len(data) == 0:
        raise ValueError("No data loaded. Check if the files are present and correctly formatted.")
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split the data
    try:
        x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        # Save the processed data if needed
        np.savez(output_file, x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)
    except Exception as e:
        print(f"Error during train/test split: {e}")

# Example usage
numpy_dir = 'data/numpy_bitmap'
output_file = 'processed_data.npz'
target_shape = (100, 100)  # Replace with your desired shape
preprocess_data(numpy_dir, output_file, target_shape)
