import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Paths
input_folder = 'data/numpy_bitmap/'
output_folder = 'data/processed/'

# Parameters
img_size = (256, 256)  # Higher resolution for better detail capture
batch_size = 500  # Adjust based on GPU memory and CPU capacity
num_threads = 8  # Adjust based on the number of CPU cores

def preprocess_images(batch_images):
    processed_images = []
    for img_array in batch_images:
        # Convert numpy array to grayscale image
        img = Image.fromarray(img_array).convert('L')
        
        # Apply edge detection
        img_cv = np.array(img)
        edges = cv2.Canny(img_cv, 100, 200)  # Canny edge detection
        
        # Convert edges to PIL image and resize
        img_edges = Image.fromarray(edges)
        img_edges = img_edges.resize(img_size)
        
        # Apply contrast enhancement
        img_edges = ImageOps.autocontrast(img_edges)
        
        # Normalize to [0, 1]
        img_edges = np.array(img_edges) / 255.0
        processed_images.append(img_edges)
    
    return np.array(processed_images)

def process_file(file_path, label):
    loaded_images = np.load(file_path)
    for i in range(0, len(loaded_images), batch_size):
        batch_images = loaded_images[i:i+batch_size]
        processed_images = preprocess_images(batch_images)
        batch_file = os.path.join(output_folder, f'{label}_batch_{i//batch_size}.npz')
        np.savez_compressed(batch_file, images=processed_images)
        del processed_images
        gc.collect()

# Multi-threaded file processing
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            label = filename.split('.')[0]
            file_path = os.path.join(input_folder, filename)
            futures.append(executor.submit(process_file, file_path, label))
    
    # Wait for all tasks to complete
    for future in as_completed(futures):
        future.result()

print("Data preprocessing complete.")
