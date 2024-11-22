import os
import numpy as np
import cv2
import nibabel as nib
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model("attention_enhanced_unet_model.h5", compile=False)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def process_nifti_file(file_path):
    """Process a NIfTI file and return all slices resized to 128x128 along with original volume."""
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()

    slices = []
    for i in range(img_data.shape[2]):
        img_slice = img_data[:, :, i]
        img_slice_resized = cv2.resize(img_slice, (128, 128))
        img_slice_resized = img_slice_resized[:, :, np.newaxis] / 255.0  # Normalize to [0, 1]
        slices.append(img_slice_resized)

    return slices, img_data

def create_4_channel_input(img_slice):
    """Create a 4-channel input for prediction by duplicating the input slice."""
    input_data = np.concatenate([img_slice] * 4, axis=-1)  # Shape: (128, 128, 4)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension: Shape: (1, 128, 128, 4)
    return input_data

def overlay_prediction_on_brain(brain_slice, prediction):
    """Overlay the predicted mask on the brain slice."""
    # Normalize brain slice to [0, 255] and convert to RGB
    brain_normalized = cv2.normalize(brain_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    brain_rgb = cv2.cvtColor(brain_normalized, cv2.COLOR_GRAY2RGB)

    # Normalize prediction and apply colormap
    prediction_normalized = (prediction * 255).astype(np.uint8)
    prediction_colored = cv2.applyColorMap(prediction_normalized, cv2.COLORMAP_JET)

    # Blend the images
    overlayed_image = cv2.addWeighted(brain_rgb, 0.7, prediction_colored, 0.3, 0)
    return overlayed_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                slices, original_volume = process_nifti_file(file_path)
                predictions = []

                for img_slice in slices:
                    input_data = create_4_channel_input(img_slice)  # Prepare input data

                    # Predict using the model
                    prediction = model.predict(input_data)[0, :, :, 0]
                    predictions.append(prediction)

                # Create an overlayed image for each slice and generate an animated GIF
                overlayed_images = []
                for i, prediction in enumerate(predictions):
                    overlayed_image = overlay_prediction_on_brain(slices[i][:, :, 0], prediction)
                    overlayed_images.append(Image.fromarray(overlayed_image))

                # Generate an animated GIF from overlayed images
                gif_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_animation.gif')
                overlayed_images[0].save(gif_path, save_all=True, append_images=overlayed_images[1:], duration=100, loop=0)

        return redirect(url_for('display_prediction'))

    return render_template('index.html')

@app.route('/prediction')
def display_prediction():
    image_url = url_for('static', filename='uploads/prediction_animation.gif')
    return render_template('display.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)

