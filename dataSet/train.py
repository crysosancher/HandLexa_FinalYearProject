import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import h5py

# Download and extract dataset
_URL = 'https://storage.googleapis.com/mledu-datasets/handSign_filtered.zip'
path_to_zip = tf.keras.utils.get_file('handlexa.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'handlexa')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'valide')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
# base_model.summary()

# Add layers and build the model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
EPOCHS = 10
history = model.fit(train_dataset, epochs=EPOCHS)

# Save model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save weights to HDF5
model.save_weights("model.h5")

# Convert HDF5 weights to binary files (limit to 3)
def save_weights_to_bin(h5file, limit=3):
    count = 0

    def save_dataset_as_bin(name, dataset):
        nonlocal count
        if count < limit:
            bin_file_name = f'{name}.bin'.replace('/', '_')
            with open(bin_file_name, 'wb') as bin_file:
                bin_file.write(dataset[()].tobytes())
            count += 1

    with h5py.File(h5file, 'r') as f:
        f.visititems(lambda name, obj: save_dataset_as_bin(name, obj) if isinstance(obj, h5py.Dataset) else None)

save_weights_to_bin('model.h5')

# Download the files in Google Colab
from google.colab import files

# Download model.json
files.download('model.json')

# Download binary weight files
import glob

for bin_file in glob.glob("*.bin"):
    files.download(bin_file)
