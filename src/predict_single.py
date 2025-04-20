import numpy as np
from keras._tf_keras.keras.models  import load_model
from keras._tf_keras.keras.preprocessing import image
import os

async def main(uploaded_scan):
   # --- Configuration ---
   MODEL_PATH = 'saved_models\\pneumonia_model.h5'
   IMG_SIZE = (150, 150)

   # --- Load the trained model ---
   model = load_model(MODEL_PATH)

   # --- Load and preprocess the image ---
   img = image.load_img(uploaded_scan, target_size=IMG_SIZE, color_mode='grayscale')
   img_array = image.img_to_array(img)
   img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 1)
   img_array = img_array / 255.0  # normalize to match training

   # --- Predict ---
   prob = model.predict(img_array)[0][0]
   label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

   return prob, label
