import os
import numpy as np
import cv2
import nibabel as nib
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import imageio

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model("attention_enhanced_unet_model.h5", compile=False)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def process_nifti_file(file_path):
    """Process a NIfTI file and return all slices resized to 128x128."""
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"The file {file_path} is empty and cannot be processed.")
    
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()
    slices = []
    for i in range(img_data.shape[2]):
        img_slice = img_data[:, :, i]
        img_slice_resized = cv2.resize(img_slice, (128, 128))
        slices.append(img_slice_resized / 255.0)  # Normalize to [0, 1]
    return slices, img_data

def overlay_prediction_on_brain(brain_slice, prediction, output_size=(512, 512)):
    """Overlay the predicted segmentation on the brain image and resize the output."""
    # Normalize the brain slice to [0, 255] and convert to uint8
    brain_normalized = cv2.normalize(brain_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize prediction to match the brain slice dimensions
    prediction_resized = cv2.resize(prediction, (brain_normalized.shape[1], brain_normalized.shape[0]))

    # Convert the prediction to a color map
    prediction_colored = cv2.applyColorMap((prediction_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert the brain image to 3-channel grayscale (RGB)
    brain_colored = cv2.cvtColor(brain_normalized, cv2.COLOR_GRAY2BGR)

    # Overlay the prediction on the brain image
    overlay = cv2.addWeighted(brain_colored, 0.7, prediction_colored, 0.3, 0)

    # Resize the final overlay to the desired output size
    overlay_resized = cv2.resize(overlay, output_size)

    return overlay_resized



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file uploads
        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return render_template('error.html', error_message="No files uploaded.")

        # Process the first uploaded file (assuming one NIfTI file)
        file = uploaded_files[0]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            brain_slices, original_volume = process_nifti_file(file_path)
        except Exception as e:
            return render_template('error.html', error_message=f"Error processing file: {e}")

        # Generate predictions and overlay
        gif_frames = []
        for idx, brain_slice in enumerate(brain_slices):
            input_data = np.expand_dims(brain_slice, axis=(0, -1))  # Shape: (1, 128, 128, 1)
            input_data = np.concatenate([input_data] * 4, axis=-1)  # Shape: (1, 128, 128, 4)

            prediction = model.predict(input_data)[0, :, :, 0]  # Shape: (128, 128)

            # Overlay prediction on the brain slice
            overlayed_image = overlay_prediction_on_brain(brain_slice, prediction, output_size=(1024, 1024))


            # Add a scale (slice index) to the frame
            overlayed_image = cv2.putText(
                overlayed_image.copy(),
                f"Slice: {idx + 1}/{len(brain_slices)}",
                (10, 20),  # Position
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Font size
                (255, 255, 255),  # Font color (white)
                2,  # Thickness
                cv2.LINE_AA
            )

            gif_frames.append(overlayed_image)

        # Save as a GIF
        gif_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.gif')
        imageio.mimsave(gif_path, gif_frames, fps=5)

        return redirect(url_for('display_prediction', gif_filename='prediction.gif'))

    return render_template('index.html')

@app.route('/prediction')
def display_prediction():
    gif_filename = request.args.get('gif_filename', 'prediction.gif')
    gif_url = url_for('static', filename=f'uploads/{gif_filename}')
    return render_template('display.html', gif_url=gif_url)

if __name__ == '__main__':
    app.run(debug=True)
