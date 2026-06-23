import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

MODEL_PATH = "leaf_disease_model.h5"
FILE_ID = "1mH1jCxhjuzjtdLW2A5zkfHUP-fEAMaVr"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)

st.title("🌴 Palm Oil Leaf Nutrient Stress Classification")

language = st.selectbox(
    "Select Language / భాష ఎంచుకోండి",
    ["English", "Telugu", "Hindi", "Tamil"]
)

class_names = [
    "boron",
    "healthy",
    "kalium",
    "magnesium",
    "nitrogen"
]

remedies = {

    "boron": {
        "English": "Boron deficiency detected. Use Borax or Solubor fertilizer in recommended quantity.",
        "Telugu": "బోరాన్ లోపం కనిపించింది. బోరాక్స్ లేదా సోలుబోర్ ఎరువును వాడండి.",
        "Hindi": "बोरॉन की कमी पाई गई है। Borax या Solubor खाद का उपयोग करें।",
        "Tamil": "போரான் குறைபாடு கண்டறியப்பட்டது. Borax அல்லது Solubor பயன்படுத்தவும்."
    },

    "healthy": {
        "English": "Leaf looks healthy. No treatment required.",
        "Telugu": "ఆకు ఆరోగ్యంగా ఉంది. మందు అవసరం లేదు.",
        "Hindi": "पत्ता स्वस्थ है। उपचार की आवश्यकता नहीं है।",
        "Tamil": "இலை ஆரோக்கியமாக உள்ளது. மருந்து தேவையில்லை."
    },

    "kalium": {
        "English": "Potassium deficiency detected. Use MOP or Potash fertilizer.",
        "Telugu": "పొటాషియం లోపం కనిపించింది. పొటాష్ ఎరువును వాడండి.",
        "Hindi": "पोटैशियम की कमी है। Potash fertilizer उपयोग करें।",
        "Tamil": "பொட்டாசியம் குறைபாடு உள்ளது. Potash உரம் பயன்படுத்தவும்."
    },

    "magnesium": {
        "English": "Magnesium deficiency detected. Use Magnesium sulphate.",
        "Telugu": "మెగ్నీషియం లోపం కనిపించింది. మెగ్నీషియం సల్ఫేట్ వాడండి.",
        "Hindi": "मैग्नीशियम की कमी है। Magnesium sulphate उपयोग करें।",
        "Tamil": "மக்னீசியம் குறைபாடு உள்ளது. Magnesium sulphate பயன்படுத்தவும்."
    },

    "nitrogen": {
        "English": "Nitrogen deficiency detected. Use nitrogen rich fertilizer.",
        "Telugu": "నైట్రోజన్ లోపం కనిపించింది. నైట్రోజన్ ఎరువును వాడండి.",
        "Hindi": "नाइट्रोजन की कमी है। Nitrogen fertilizer उपयोग करें।",
        "Tamil": "நைட்ரஜன் குறைபாடு உள்ளது. Nitrogen உரம் பயன்படுத்தவும்."
    }
}

uploaded_file = st.file_uploader(
    "Upload Palm Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224,224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    confidence = prediction[0][index] * 100

    disease = class_names[index]

    st.success(f"Detected Disease: {disease}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Recommended Solution 💊")

    st.write(remedies[disease][language])