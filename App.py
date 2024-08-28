import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('face_recognition_model.h5')

model = load_model()

# Class labels (replace with actual student names)
class_labels = ['Arthur', 'Ray', 'Tep']

# Preprocessing function
def preprocess_input(img):
    img = tf.cast(img, tf.float32)
    img /= 255.0
    return img

# Define a function to predict the label of an image
def predict(image):
    img = cv2.resize(image, (128, 128))  # Resize to 128x128
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Apply preprocessing
    predictions = model.predict(img)
    return class_labels[np.argmax(predictions)]

# Start the Streamlit app
st.title('Face Recognition Web App')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Use Streamlit's camera input
    st.write("Align your face similar to the guide on the right:")
    img_file_buffer = st.camera_input("Take a picture", key="camera")

with col2:
    # Create a 128x128 guide image
    guide_image = np.ones((128, 128, 3), dtype=np.uint8) * 255  # White background
    cv2.rectangle(guide_image, (0, 0), (127, 127), (255, 0, 0), 2)  # Blue rectangle
    st.image(guide_image, caption="Face Alignment Guide", use_column_width=False, width=128)

if img_file_buffer is not None:
    # Read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # Get the dimensions of the captured image
    height, width = frame_rgb.shape[:2]
    
    # Calculate the position to crop a 128x128 square from the center
    start_y = max(0, (height - 128) // 2)
    start_x = max(0, (width - 128) // 2)
    cropped_frame = frame_rgb[start_y:start_y+128, start_x:start_x+128]
    
    # Ensure the cropped frame is exactly 128x128
    cropped_frame = cv2.resize(cropped_frame, (128, 128))
    
    # Predict the label
    label = predict(cropped_frame)
    
    # Display the label on the cropped frame
    cv2.putText(cropped_frame, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Display the cropped image with the label
    st.image(cropped_frame, channels="RGB", use_column_width=False, width=128)

# Display guide message
st.write("Please align your face within the camera view similar to the guide before taking the picture.")