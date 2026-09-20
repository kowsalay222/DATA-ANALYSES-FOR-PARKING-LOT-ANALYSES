"""
Flask web app for Parking Lot Occupancy Detection.

How it works:
1. User uploads an image via the web form (index.html)
2. Flask saves the image temporarily
3. The trained YOLO model runs inference on it
4. Boxes are drawn on the image and it's saved as a "result" image
5. The results page shows the annotated image + occupancy counts
"""

import os
import uuid
import cv2
from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/best.pt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load the trained model ONCE when the server starts ---
# (loading it inside the request handler would be very slow — it would
#  reload the whole model on every single upload)
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")


@app.route('/')
def home():
    """Show the upload form."""
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle an uploaded image: run the model, draw boxes, show results."""

    # 1. Check a file was actually submitted
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    # 2. Save the uploaded image with a unique name (avoids overwriting
    #    if two people upload a file with the same name at once)
    unique_id = uuid.uuid4().hex
    upload_filename = f"{unique_id}_input.jpg"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(upload_path)

    # 3. Run the model on the uploaded image
    results = model.predict(upload_path, conf=0.5, verbose=False)
    result = results[0]

    # 4. Count empty vs occupied spaces
    class_names = result.names
    empty_count = sum(1 for c in result.boxes.cls if class_names[int(c)] == 'space-empty')
    occupied_count = sum(1 for c in result.boxes.cls if class_names[int(c)] == 'space-occupied')
    total = empty_count + occupied_count
    occupancy_rate = round((occupied_count / total * 100), 1) if total > 0 else 0

    # 5. Draw the boxes onto the image and save the annotated version
    annotated = result.plot(conf=False, line_width=2, font_size=10)  # returns BGR image
    result_filename = f"{unique_id}_result.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, annotated)

    # 6. Show the results page
    return render_template(
        'index.html',
        result={
            'image_path': result_path,
            'empty': empty_count,
            'occupied': occupied_count,
            'total': total,
            'occupancy_rate': occupancy_rate,
        }
    )


if __name__ == '__main__':
    # debug=True auto-reloads the server when you edit code — handy while developing.
    # Turn this off (debug=False) before deploying anywhere public.
    app.run(debug=True)