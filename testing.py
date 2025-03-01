import streamlit as st
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("/home/keshav/Downloads/CaRPM/VsCode/car_parts_model_74_category.h5")  # Update with your model path

# Load class names
class_names = ['AIR COMPRESSOR',
 'AIR FILTER',
 'ALTERNATOR',
 'BALL JOINT',
 'BATTERY',
 'BRAKE CALIPER',
 'BRAKE PAD',
 'BRAKE ROTOR',
 'BULB',
 'CABIN FILTER',
 'CAMSHAFT',
 'CARBERATOR',
 'CLOCK SPRING',
 'CLUTCH PLATE',
 'CLUTCH RELEASE BEARING',
 'COIL SPRING',
 'CRANKSHAFT',
 'CYLINDER HEAD',
 'DICKY SHOCK ABSORBER',
 'DIESEL FILTER',
 'DISTRIBUTOR',
 'ENGINE BLOCK',
 'ENGINE VALVE',
 'FOG LAMP',
 'FOG LAMP COVER',
 'FRONT SHOCK ABSORBER',
 'FRONT STABILIZER LINK',
 'FRONT WHEEL HUB',
 'FUEL FILTER',
 'FUEL INJECTOR',
 'FUEL PUMP',
 'FUEL TANK CAP',
 'FUSE BOX',
 'GAS CAP',
 'HEADLIGHTS',
 'HORN',
 'IDLER ARM',
 'IGNITION COIL',
 'INSTRUMENT CLUSTER',
 'LEAF SPRING',
 'LOWER TRACK CONTROL ARM',
 'MUFFLER',
 'OIL FILTER',
 'OIL PAN',
 'OIL PRESSURE SENSOR',
 'OVERFLOW TANK',
 'OXYGEN SENSOR',
 'PISTON',
 'PRESSURE PLATE',
 'RADIATOR',
 'RADIATOR FAN',
 'RADIATOR HOSE',
 'RADIO',
 'REAR SHOCK ABSORBER',
 'REAR WHEEL HUB ASSEMBLY',
 'SHIFT KNOB',
 'SIDE MIRROR',
 'SILENCER',
 'SPARK PLUG',
 'SPOILER',
 'STARTER',
 'STRUT MOUNTING',
 'TAIL LAMP ASSEMBLY - LH',
 'TAIL LAMP ASSEMBLY - RH',
 'TENSIONER ',
 'THERMOSTAT ASSEMBLY',
 'TORQUE CONVERTER',
 'TRANSMISSION',
 'VACUUM BRAKE BOOSTER',
 'VALVE LIFTER',
 'WATER PUMP',
 'WHEEL RIM',
 'WINDOW REGULATOR',
 'WIPER BLADE']  # Replace with actual class names

# Create folder to store images
os.makedirs("predictions", exist_ok=True)

# Streamlit UI
st.title("🚗 Car Spare Parts Classifier")
st.write("Upload an image to classify the spare part.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # Display result
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Save the uploaded image with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"predictions/{timestamp}_{predicted_class}.jpg"
    img.save(img_filename)

    # Save results to CSV
    csv_filename = "predictions/predictions_log.csv"
    new_data = pd.DataFrame([[timestamp, uploaded_file.name, predicted_class, confidence]], 
                            columns=["Timestamp", "Original Filename", "Predicted Class", "Confidence"])

    if os.path.exists(csv_filename):
        new_data.to_csv(csv_filename, mode="a", header=False, index=False)
    else:
        new_data.to_csv(csv_filename, mode="w", header=True, index=False)

    st.success("Prediction saved successfully!")
