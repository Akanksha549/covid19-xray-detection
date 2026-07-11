# Detection of Covid-19 from Chest X-ray

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# Dataset Path
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["covid", "normal"]

print("Dataset Path:", DATASET_DIR)
print("Dataset Exists:", os.path.exists(DATASET_DIR))

# -------------------------------------------------------
# Image Augmentation
# -------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

# -------------------------------------------------------
# Compile
# -------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# -------------------------------------------------------
# Accuracy Plot
# -------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.legend()

plt.show()

# -------------------------------------------------------
# Loss Plot
# -------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.legend()

plt.show()

# -------------------------------------------------------
# Load Saved Model
# -------------------------------------------------------

model = load_model(MODEL_PATH)

# -------------------------------------------------------
# Test Normal Image
# -------------------------------------------------------

normal_image = os.path.join(
    DATASET_DIR,
    "normal",
    "IM-0131-0001.jpeg"
)

if os.path.exists(normal_image):

    img = image.load_img(normal_image,target_size=(299,299))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    result = model.predict(img_array)

    prediction = "COVID" if result[0][0] > 0.5 else "NORMAL"

    print("Prediction:",prediction)

    plt.imshow(img)
    plt.title(prediction)
    plt.axis("off")
    plt.show()

# -------------------------------------------------------
# Test Covid Image
# -------------------------------------------------------

covid_image = os.path.join(
    DATASET_DIR,
    "covid",
    "1-s2.0-S0929664620300449-gr2_lrg-a.jpg"
)

if os.path.exists(covid_image):

    img = image.load_img(covid_image,target_size=(299,299))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    result = model.predict(img_array)

    prediction = "COVID" if result[0][0] > 0.5 else "NORMAL"

    print("Prediction:",prediction)

    plt.imshow(img)
    plt.title(prediction)
    plt.axis("off")
    plt.show()
