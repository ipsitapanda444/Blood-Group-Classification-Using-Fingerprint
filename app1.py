import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load your pre-trained model
model = load_model('model.h5')  # Ensure 'model.h5' is in the same folder or adjust the path

# Streamlit app title
st.title('Blood Group Prediction from Fingerprint (BMP Format)')

# User inputs: Name, Phone Number, Email, Date of Birth
name = st.text_input("Enter your name:")
phone_number = st.text_input("Enter your phone number:")
email = st.text_input("Enter your email:")
dob = st.date_input("Enter your Date of Birth:")

# File uploader for fingerprint image (including BMP files)
uploaded_file = st.file_uploader("Upload a fingerprint image", type=['bmp', 'png', 'jpg', 'jpeg'])

# Predict function
def predict_blood_group(image_path):
    # Preprocess the image before feeding it into the model
    image = load_img(image_path, target_size=(64, 64), color_mode='rgb')  # Adjust size to 64x64 as expected by your model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image to [0, 1] range if necessary for your model

    # Make prediction
    prediction = model.predict(image)

    if prediction.shape[1] != 8:
        return None  # Return None if the prediction doesn't have 8 output classes

    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class to blood group (assuming 8 classes for blood types)
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']

    if predicted_class < len(blood_groups):
        return blood_groups[predicted_class]
    else:
        return None  # Return None if the prediction index is out of bounds

# If a file is uploaded, proceed to make a prediction
if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform the prediction
    prediction = predict_blood_group(file_path)

    # Display the results
    st.subheader("Prediction Results")
    st.write(f"Name: {name}")
    st.write(f"Phone Number: {phone_number}")
    st.write(f"Email: {email}")
    st.write(f"Date of Birth: {dob}")
   
    if prediction:
        st.write(f"Predicted Blood Group: {prediction}")
    else:
        st.write("Could not make a valid prediction. Please check your model or image.")
else:
    st.write("Please upload an image to get a prediction.")

# Ensure the 'uploads' folder exists for saving uploaded files
os.makedirs('uploads', exist_ok=True)
