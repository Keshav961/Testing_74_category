import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# Load trained model
MODEL_PATH = "car_parts_model_74_category.h5"  # Update with your actual model file path
model = tf.keras.models.load_model(MODEL_PATH)

# List of 74 spare part categories
class_names = [
    'AIR COMPRESSOR', 'AIR FILTER', 'ALTERNATOR', 'BALL JOINT', 'BATTERY',
    'BRAKE CALIPER', 'BRAKE PAD', 'BRAKE ROTOR', 'BULB', 'CABIN FILTER',
    'CAMSHAFT', 'CARBERATOR', 'CLOCK SPRING', 'CLUTCH PLATE', 'CLUTCH RELEASE BEARING',
    'COIL SPRING', 'CRANKSHAFT', 'CYLINDER HEAD', 'DICKY SHOCK ABSORBER', 'DIESEL FILTER',
    'DISTRIBUTOR', 'ENGINE BLOCK', 'ENGINE VALVE', 'FOG LAMP', 'FOG LAMP COVER',
    'FRONT SHOCK ABSORBER', 'FRONT STABILIZER LINK', 'FRONT WHEEL HUB', 'FUEL FILTER', 'FUEL INJECTOR',
    'FUEL PUMP', 'FUEL TANK CAP', 'FUSE BOX', 'GAS CAP', 'HEADLIGHTS',
    'HORN', 'IDLER ARM', 'IGNITION COIL', 'INSTRUMENT CLUSTER', 'LEAF SPRING',
    'LOWER TRACK CONTROL ARM', 'MUFFLER', 'OIL FILTER', 'OIL PAN', 'OIL PRESSURE SENSOR',
    'OVERFLOW TANK', 'OXYGEN SENSOR', 'PISTON', 'PRESSURE PLATE', 'RADIATOR',
    'RADIATOR FAN', 'RADIATOR HOSE', 'RADIO', 'REAR SHOCK ABSORBER', 'REAR WHEEL HUB ASSEMBLY',
    'SHIFT KNOB', 'SIDE MIRROR', 'SILENCER', 'SPARK PLUG', 'SPOILER',
    'STARTER', 'STRUT MOUNTING', 'TAIL LAMP ASSEMBLY - LH', 'TAIL LAMP ASSEMBLY - RH', 'TENSIONER',
    'THERMOSTAT ASSEMBLY', 'TORQUE CONVERTER', 'TRANSMISSION', 'VACUUM BRAKE BOOSTER', 'VALVE LIFTER',
    'WATER PUMP', 'WHEEL RIM', 'WINDOW REGULATOR', 'WIPER BLADE'
]

# Function to preprocess images
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)

# Function to predict image class
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return class_names[predicted_class_idx], confidence

# Streamlit UI
st.title("🚗 Car Spare Parts Classifier  - Batch Testing")
st.write("Upload multiple images to classify car parts.")

# Upload multiple images
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Check if new images are uploaded
if uploaded_files:
    # Clear previous results from session state
    if "results" in st.session_state:
        st.session_state.results = []  # Reset the previous results
    
    results = []  # Store results for display
    
    # Display prediction results
    st.write("## 🔍 Predictions")
    cols = st.columns(3)  # Arrange images in 3 columns

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)

        # Predict class & confidence
        predicted_label, confidence = predict_image(image)

        # Store results in session state
        results.append([uploaded_file.name, predicted_label, f"{confidence:.2f}%"])

        # Display images & predictions in structured format
        with cols[idx % 3]:  # Distribute images across columns
            st.image(image, caption=f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}%", use_container_width=True)

    # Show results in a table
    df_results = pd.DataFrame(results, columns=["Image Name", "Predicted Class", "Confidence"])
    st.write("### 📊 Summary Table")
    st.dataframe(df_results)

    # Save results in session state
    st.session_state.results = results
