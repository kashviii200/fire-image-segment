import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image

# Load the pickled model
with open('unet_fire_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set the title of the app
st.title("Fire Image Segmentation")

# Upload an image for segmentation
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))  # Resize to 256x256 as the model expects
    img_resized = img_resized.astype('float32') / 255.0  # Normalize to [0, 1]
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension for prediction
    
    # Predict the segmentation mask using the loaded model
    mask_pred = model.predict(img_resized)[0]
    
    # Normalize mask prediction to [0, 255] for better visualization
    mask_pred = (mask_pred * 255).astype(np.uint8)
    
    # Ensure the predicted mask is RGB for display in Streamlit
    if len(mask_pred.shape) == 2:
        # If the mask is grayscale, convert it to RGB by stacking
        mask_pred_rgb = np.stack([mask_pred]*3, axis=-1)
    else:
        # If the mask is already in RGB format
        mask_pred_rgb = mask_pred
    
    # Display the predicted mask as a single image
    st.image(mask_pred_rgb, caption='Predicted Mask', use_column_width=True)
