import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import svgwrite

# Import or define your functions here if they are in another module
# from preprocessing import preprocess_image, convert_to_vector

# Define preprocess_image and convert_to_vector functions if not imported
def preprocess_image(img_array):
    """Preprocesses a single image array to enhance edges and resize."""
    # Convert numpy array to grayscale image
    img = Image.fromarray(img_array).convert('L')
    
    # Apply edge detection using Canny
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)  # Adjust thresholds as needed
    
    # Resize to target size
    img_edges = Image.fromarray(edges).resize((256, 256))  # Use img_size if defined elsewhere
    
    # Enhance contrast
    img_edges = ImageOps.autocontrast(img_edges)
    
    # Normalize to [0, 1]
    img_edges = np.array(img_edges) / 255.0
    return img_edges

def convert_to_vector(img_edges):
    """Converts edge-detected images to vector format (SVG) using contours."""
    contours, _ = cv2.findContours((img_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dwg = svgwrite.Drawing(size=(256, 256))  # Use img_size if defined elsewhere
    
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        dwg.add(dwg.polyline(points, stroke='black', fill='none'))
    
    return dwg.tostring()

# Paths
output_folder = 'data/processed/'
vector_output_folder = 'data/vectors/'

# Limit the number of images to test
test_limit = 5  # Number of images to test

def load_test_data():
    """Loads a limited set of test data to check the preprocessing."""
    test_images = []
    for _ in range(test_limit):
        test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)  # Simulate a 28x28 grayscale image
        test_images.append(test_image)
    return np.array(test_images)

def test_preprocess_image():
    """Tests the preprocess_image function on a limited set of images."""
    test_data = load_test_data()
    for index, img_array in enumerate(test_data):
        img_edges = preprocess_image(img_array)
        assert img_edges.shape == (256, 256), f"Expected (256, 256), but got {img_edges.shape} for image {index+1}"
        assert img_edges.max() <= 1.0 and img_edges.min() >= 0.0, "Processed image should be normalized to [0, 1]"
        print(f"Preprocessing test passed for image {index+1}.")

def test_convert_to_vector():
    """Tests the convert_to_vector function on a limited set of images."""
    test_data = load_test_data()
    for index, img_array in enumerate(test_data):
        img_edges = preprocess_image(img_array)
        vector_repr = convert_to_vector(img_edges)
        assert '<svg' in vector_repr and '</svg>' in vector_repr, "The vector representation should contain valid SVG tags"
        print(f"Vector conversion test passed for image {index+1}.")

def test_saved_files():
    """Checks if the processed files are saved correctly, limited to a few files."""
    processed_files = os.listdir(output_folder)[:test_limit]
    vector_files = os.listdir(vector_output_folder)[:test_limit]

    # Check processed images
    for filename in processed_files:
        if filename.endswith('.npz'):
            data = np.load(os.path.join(output_folder, filename))
            images = data['images']
            assert images.shape[1:] == (256, 256), f"Processed images should be of size (256, 256), got {images.shape[1:]}"
            print(f"Processed data file '{filename}' verified successfully.")

    # Check vector files
    for filename in vector_files:
        if filename.endswith('.npz'):
            data = np.load(os.path.join(vector_output_folder, filename), allow_pickle=True)
            vectors = data['vectors']
            for vector in vectors:
                assert '<svg' in vector and '</svg>' in vector, "Each vector should contain valid SVG tags"
            print(f"Vector data file '{filename}' verified successfully.")

# Run tests
test_preprocess_image()
test_convert_to_vector()
test_saved_files()

print("All tests completed successfully.")
