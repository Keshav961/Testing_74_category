import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "car_parts_model_74_category.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (update with actual class names)
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
 'WIPER BLADE']  # Replace with your actual class names

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🚗 Car Parts Classifier - Batch Testing")
st.write("Upload multiple images to classify car parts.")

# Upload multiple images
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Process and predict
if uploaded_files:
    results = []  # Store predictions
    
    # Display images and predictions in a grid layout
    cols = st.columns(3)  # Create 3 columns for better layout
    
    for index, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)

        # Get model prediction
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]  # Get class label
        confidence = round(np.max(predictions) * 100, 2)  # Get confidence score

        # Store results
        results.append({
            "Image Name": uploaded_file.name,
            "Predicted Class": predicted_class,
            "Confidence (%)": confidence
        })

        # Display image with prediction
        with cols[index % 3]:  # Arrange images in 3-column layout
            st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)
            st.write(f"**Predicted:** {predicted_class}  \n**Confidence:** {confidence}%")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display results in table
    st.write("### 📝 Prediction Results")
    st.dataframe(results_df, use_container_width=True)  # Use full width for better visibility

    # Download button for CSV
    csv_filename = "batch_predictions.csv"
    results_df.to_csv(csv_filename, index=False)
    st.download_button("📥 Download Results", data=open(csv_filename).read(), file_name=csv_filename, mime="text/csv")
