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
# Training Data
# -------------------------------------------------------

train_data = train_datagen.flow_from_directory(
    directory=DATASET_DIR,
    target_size=(299, 299),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset='training',
    classes=CLASSES
)

val_data = train_datagen.flow_from_directory(
    directory=DATASET_DIR,
    target_size=(299, 299),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset='validation',
    classes=CLASSES
)

# -------------------------------------------------------
# CNN Model
# -------------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(299,299,3)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.summary()

# -------------------------------------------------------
# Compile
# -------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=2
)

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "my_modell.keras")

model.save(MODEL_PATH)

print("Model Saved:", MODEL_PATH)

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
