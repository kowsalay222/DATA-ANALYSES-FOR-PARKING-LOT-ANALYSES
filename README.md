# Parking Lot Occupancy Detector

A computer vision-based web application for detecting and counting occupied and empty parking spaces from parking-lot images using YOLO, Flask, and OpenCV.

## Project Overview

This project uses a deep learning-based object detection model to detect parking spaces from an uploaded image and determine whether each detected space is empty or occupied. The Flask web interface allows users to upload an image and view the annotated detection result together with parking statistics.

## Features

- Detects occupied and empty parking spaces from images
- Counts total, occupied, and available parking spaces
- Calculates the parking occupancy rate
- Visualizes detection results using bounding boxes
- Provides a web interface for image upload and analysis
- Supports common image formats such as JPG and PNG
- Saves input and result images for the application workflow
- Uses a trained YOLO model for object detection

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Flask | Web application and backend |
| YOLO (Ultralytics) | Parking-space object detection |
| OpenCV | Image processing |
| HTML | Web interface |
| CSS | User interface styling |

## Project Structure

```text
parking-lot-occupancy-detector/
|
|-- app.py
|-- models/
|   |-- best.pt
|
|-- static/
|   |-- style.css
|   |-- uploads/
|
|-- templates/
|   |-- index.html
|
|-- requirements.txt
|-- README.md
```

## How It Works

1. Upload an image of a parking lot using the web interface.
2. The Flask application receives and saves the uploaded image.
3. The trained YOLO model (`best.pt`) processes the image.
4. The model detects parking-space objects.
5. Detected spaces are classified as:
   - `space-empty`
   - `space-occupied`
6. Bounding boxes are drawn on the image.
7. The application calculates total, empty, occupied, and occupancy rate values.
8. The annotated image and statistics are displayed on the web page.

## Occupancy Calculation

The occupancy rate is calculated using:

```text
Occupancy Rate = (Occupied Spaces / Total Detected Spaces) × 100
```

For example:

```text
Total Spaces    : 40
Occupied Spaces : 2
Empty Spaces    : 38
Occupancy Rate  : 5%
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kowsalay222/Parking-lot-occupancy-detector.git
```

### 2. Open the project folder

```bash
cd Parking-lot-occupancy-detector
```

### 3. Create a virtual environment

For Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scriptsctivate
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

## How to Run

Make sure the trained model is available at:

```text
models/best.pt
```

Start the Flask application:

```bash
python app.py
```

Open a browser and go to:

```text
http://127.0.0.1:5000
```

Upload a parking-lot image and click the analysis button to view the detection result.

## Model Information

The project uses a trained YOLO model stored as:

```text
models/best.pt
```

The application expects the model to provide the following parking-space classes:

```text
space-empty
space-occupied
```

The model's ability to work with different parking lots depends on the diversity of the data used during training.

## Model Limitation

A model trained mainly on one parking-lot layout, camera position, or dataset may not generalize well to completely different parking environments.

For improved performance on different parking lots, the model can be trained or fine-tuned using a more diverse dataset containing:

- Multiple parking lots
- Different camera angles
- Different parking layouts
- Different lighting conditions
- Different vehicle types
- Different image resolutions

If the model does not detect `space-empty` or `space-occupied` objects, the application cannot calculate a meaningful occupancy rate.

## Example Output

The application displays:

```text
Total Spaces       : 20
Occupied Spaces    : 14
Available Spaces   : 6
Occupancy Rate     : 70%
```

The result image contains bounding boxes around the detected parking spaces.

## Use Cases

- Smart parking management systems
- Campus and college parking monitoring
- Office parking monitoring
- Shopping mall parking analysis
- Hospital parking monitoring
- Public parking facilities
- Traffic and urban planning applications

## Future Enhancements

- Real-time video and CCTV detection
- Automatic parking-space selection
- Individual parking-slot status
- Interactive occupancy dashboard
- Historical occupancy analytics
- Database integration for storing detection results
- Parking-full notifications
- Mobile-friendly interface
- Cloud deployment
- Improved model training using diverse parking datasets

## Author

**Kowsalya S**  
AI & Data Science Student

## Repository

https://github.com/kowsalay222/Parking-lot-occupancy-detector
