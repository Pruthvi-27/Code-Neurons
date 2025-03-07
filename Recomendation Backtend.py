import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os

# Define dataset paths
train_dir = "C:\\Users\\nitee\\Desktop\\New Plant Diseases Dataset\\train"
valid_dir = "C:\\Users\\nitee\\Desktop\\New Plant Diseases Dataset\\valid"

# Image Preprocessing & Augmentation
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Get class labels
class_labels = list(train_data.class_indices.keys())

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_labels), activation="softmax")  
])

# Compile model
initial_lr = 0.001
model.compile(optimizer=Adam(learning_rate=initial_lr),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train model
EPOCHS = 10
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1)

model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data),
    validation_steps=len(valid_data),
    callbacks=[lr_callback]
)

# Save trained model
model.save("plant_disease_model.h5")
print("Model training complete and saved as 'plant_disease_model.h5'.")
