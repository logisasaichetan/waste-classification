from flask import Flask, render_template, request, send_file
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained CNN model
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_output = None 
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']

        # Preprocess the image
        img = Image.open(image)
        img = img.resize((224,224))  # Adjust the size as per your model
        img_array = np.array(img) / 255.0  # Normalize pixel values

        # Make a prediction
        prediction = model.predict(img_array.reshape(1, 224, 224, 3))
        predicted_class = np.argmax(prediction)

        # Determine the predicted output
        predicted_output = "biodegradable" if predicted_class == 0 else "non-biodegradable"

        # Save the processed image
        processed_image_path = 'processed_image.png'
        img.save(processed_image_path)

    return render_template('index.html', predicted_output=predicted_output)
        
@app.route('/processed_image/<path:filename>')
def processed_image(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)