import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image

# Load the trained U-Net model
try:
    with open('unet_fire_model.pkl', 'rb') as f:
        model = pickle.load(f)
        st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set the title of the app
st.title("Stove Flame Image Segmentation")

# Add a brief description
st.write("""
This application segments gas stove flames using a trained U-Net deep learning model. 
The focus is on **blue flame detection**, which is important for assessing the efficiency of a gas stove. 
Upload a stove flame image, and the app will predict the segmented flame areas using the U-Net model.
""")

# Upload an image for segmentation
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_image is not None:
    try:
        # Display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        img = np.array(img)
        img_resized = cv2.resize(img, (256, 256))
        
        # Convert the image to HSV for flame segmentation
        hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

        # Define the color range for blue flames (use appropriate HSV values)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create mask for blue flames
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Normalize and expand dimensions for the model
        img_resized = img_resized.astype('float32') / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        # Predict the segmentation mask using the loaded model
        mask_pred = model.predict(img_resized)[0]
        
        # Post-process the predicted mask
        mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255  # Binarize the output mask
        
        # Apply the blue mask to the prediction (to focus on blue flame regions)
        mask_pred = cv2.bitwise_and(mask_pred, mask_pred, mask=blue_mask)

        # Display the predicted mask
        st.image(mask_pred, caption='Predicted Mask', use_column_width=True)
        st.success("Prediction complete.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.warning("Please upload an image for segmentation.")
