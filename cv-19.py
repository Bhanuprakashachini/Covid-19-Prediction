import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import numpy as np
from sklearn.model_selection import train_test_split
import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Ensure 'tmp' directory exists
os.makedirs('tmp', exist_ok=True)

class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, image_size, batch_size):
        self.file_paths = file_paths
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.indices = np.arange(len(file_paths))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices_range = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_indices_paths = [self.file_paths[i] for i in batch_indices_range]
        batch_indices_labels = [self.labels[i] for i in batch_indices_range]
        return self.data_generator(batch_indices_paths, batch_indices_labels)

    def data_generator(self, batch_indices_paths, batch_indices_labels):
        images = []
        for path in batch_indices_paths:
            image = Image.open(path).convert('RGB')
            img = image.resize(self.image_size)
            image_array = np.array(img) / 255.0
            images.append(image_array)
        images = np.array(images)
        labels = np.array(batch_indices_labels)
        return images, labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


virus = 'C:\\Users\\pc\\Desktop\\covid-19\\COVID_IEEE\\virus'
normal = 'C:\\Users\\pc\\Desktop\\covid-19\\COVID_IEEE\\normal'

viral_path = [os.path.join(virus, i) for i in os.listdir(virus)]
normal_path = [os.path.join(normal, i) for i in os.listdir(normal)]

viral_label = np.array([0] * len(viral_path))
normal_label = np.array([1] * len(normal_path))

file_paths = np.concatenate((viral_path, normal_path), axis=0)
labels = np.concatenate((viral_label, normal_label), axis=0)

image_size = (128, 128)
batch_size = 32

training = ImageGenerator(file_paths, labels, image_size, batch_size)

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(training, epochs=15)

# Save the model
cnn.save('covid_classifier_model.h5')

# Load the model
cnn = keras.models.load_model('covid_classifier_model.h5')

def load_single_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize((128, 128))  # Ensure the image is the same size
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the templates directory

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to a temporary location
        filename = secure_filename(file.filename)
        img_path = os.path.join('tmp', filename)
        file.save(img_path)

        # Load the saved image using PIL
        sample_image = load_single_image(img_path)

        # Make predictions using the CNN model
        prediction = cnn.predict(sample_image)
        class_label = np.argmax(prediction)

        # Determine result
        if class_label == 0:
            result = "positive for COVID"
        else:
            result = "negative for COVID"

        
        os.remove(img_path)

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
