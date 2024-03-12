# Barcode, QR, and Vehicle Detection using YOLOv8

This project implements a web application for detecting barcodes, QR codes, and vehicles in images using YOLOv8 models. The models have been trained separately using the Ultralytics YOLO framework.

## Models

- **Vehicle Detection (YOLOv8):** Trained on a dataset for top-view vehicle detection.
- **Barcode-QR Detection (YOLOv8):** Trained on a dataset for detecting barcodes and QR codes.

## Features

- **Model Selection:** Choose between barcode/QR code detection and vehicle detection.
- **Upload Image:** Upload an image containing objects for detection.
- **Detection:** Run detection on the uploaded image.
- **Results Display:** View the detection results directly on the web interface.

## Requirements

- Python 3.x
- Streamlit
- Pillow (PIL)
- Ultralytics YOLO (Pre-trained .pt models are used)

## Setup

1. Clone the repository:
   
   git clone https://github.com/Kushagra077/Barcoded-QR-Vehicle-Detection-using-YOLOv8.git

2. Install the required Python packages:

   pip install -r requirements.txt
   
3. Download the pre-trained YOLO model weights (vehicle.pt and barcode.pt) and place them in the root directory of the project.

4. Run the Streamlit app:
  streamlit run main.py




