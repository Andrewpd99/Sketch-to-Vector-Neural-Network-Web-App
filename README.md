# Sketch Recognition App

## Overview
This project is an AI-powered web application that converts hand-drawn sketches into clean vector graphics.

## Setup
Dataset from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
Did not include .npy files in repository as the files are too large.


1. **Clone the repository:**
   ```bash
   git clone https://github.com/Andrewpd99/AI-Powered-Sketch-Recognition-Web-App




## File Structure
sketch_recognition_app/
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data/
│   ├── numpy_bitmap/
│   │   └── All images.npy
│   └── processed/
│       └── processed_data.npz
│
├── models/
│   └── model.pth
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── requirements.txt
├── README.md
└── .gitignore