# Sketch Recognition App

## Overview
This project is an AI-powered web application that converts hand-drawn sketches into clean vector graphics. It utilizes a convolutional neural network (CNN) to recognize and convert sketches into vector format, making it a valuable tool for designers, animators, and doodlers.

## Setup
Currently in progress
Dataset from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap  
I Did not include .npy files in repository as the files are too large.

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- Flask/FastAPI (for backend server)
- JavaScript, HTML, CSS (for frontend)
- pip install -r requirements.txt


1. **Clone the repository:**
   ```bash
   git clone https://github.com/Andrewpd99/AI-Powered-Sketch-Recognition-Web-App



## File Structure
sketch_to_vector_web_app/
│
├── backend/
│   ├── __init__.py
│   ├── app.py               # Main backend server for the web app (e.g., Flask/FastAPI)
│   ├── model.py             # Model loading and prediction functions
│   ├── utils.py             # Utility functions (e.g., for image processing or vector conversion)
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html       # Main frontend template for rendering the app
│
├── frontend/
│   ├── index.html           # Optional: Additional frontend resources if needed
│   ├── style.css
│   └── script.js
│
├── data/
│   ├── numpy_bitmap/
│   │   └── All images.npy   # Raw dataset files
│   ├── processed/           # Processed images for training
│   │   ├── processed_data.npz
│   │   └── additional_batches/  
│   └── vectors/             # New directory for storing vector data
│       └── vector_data.npz
│
├── models/
│   ├── model.pth            # Trained model weights
│   └── vectorizer_model.pth # If separate models are used for vectorizing
│
├── scripts/
│   ├── preprocess_data.py   # Script to preprocess data (resizing, edge detection, etc.)
│   ├── vectorize_data.py    # (New) Script specifically for converting processed images to vectors
│   ├── train.py             # Training script for the model
│   ├── evaluate_model.py    # Script to evaluate the model performance
│   └── setup.py             # Setup configurations, potentially for model and data pipelines
│
├── requirements.txt
├── README.md
└── .gitignore


