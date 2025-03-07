import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\nitee\\Desktop\\plant_disease_model.h5")

# Load class labels
train_dir = "C:\\Users\\nitee\\Desktop\\New Plant Diseases Dataset\\train"

class_labels = sorted(list(os.listdir(train_dir)))  # Get folder names as class labels

# Streamlit UI
st.title("🌱 Plant Disease Detection App")
st.write("Upload an image of a leaf to detect if it's healthy or has a disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))  # Resize
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)  
    predicted_class = class_labels[predicted_class_index]  

    # Show result
    st.subheader(f"🟢 Predicted: {predicted_class}")
