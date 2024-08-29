import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import svgwrite
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Paths
input_folder = 'data/numpy_bitmap/'
output_folder = 'data/processed/'
vector_output_folder = 'data/vectors/'

# Parameters
img_size = (256, 256)  # Standard size for images
batch_size = 500  # Number of images to process in each batch
num_threads = 8  # Number of parallel threads for processing

def preprocess_image(img_array):
    """Preprocesses a single image array to enhance edges and resize."""
    # Convert numpy array to grayscale image
    img = Image.fromarray(img_array).convert('L')
    
    # Apply edge detection using Canny
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)  # Adjust thresholds as needed
    
    # Resize to target size
    img_edges = Image.fromarray(edges).resize(img_size)
    
    # Enhance contrast
    img_edges = ImageOps.autocontrast(img_edges)
    
    # Normalize to [0, 1]
    img_edges = np.array(img_edges) / 255.0
    return img_edges

def convert_to_vector(img_edges):
    """Converts edge-detected images to vector format (SVG) using contours."""
    contours, _ = cv2.findContours((img_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dwg = svgwrite.Drawing(size=img_size)
    
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        dwg.add(dwg.polyline(points, stroke='black', fill='none'))
    
    return dwg.tostring()

def process_file(file_path, label):
    """Processes images from file and saves them along with their vector equivalents."""
    loaded_images = np.load(file_path)
    processed_data = []
    vector_data = []
    
    for i in range(0, len(loaded_images), batch_size):
        batch_images = loaded_images[i:i+batch_size]
        for img_array in batch_images:
            img_edges = preprocess_image(img_array)
            vector_repr = convert_to_vector(img_edges)
            processed_data.append(img_edges)
            vector_data.append(vector_repr)
        
        # Save processed images and vectors
        batch_file = os.path.join(output_folder, f'{label}_batch_{i//batch_size}.npz')
        vector_file = os.path.join(vector_output_folder, f'{label}_batch_{i//batch_size}.npz')
        np.savez_compressed(batch_file, images=np.array(processed_data))
        np.savez_compressed(vector_file, vectors=vector_data)
        processed_data.clear()
        vector_data.clear()
        gc.collect()

# Multi-threaded processing of files
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
