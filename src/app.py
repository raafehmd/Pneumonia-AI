import streamlit as st
import asyncio
from keras._tf_keras.keras.models  import load_model
from keras._tf_keras.keras.preprocessing import image
from predict_single import main
from PIL import Image

# Streamlit UI
st.set_page_config(page_title="Pneumonia AI Predictor", page_icon="", layout="wide")

#st.image("canadian-university-dubai-seeklogo.png", use_container_width=True)


st.title("Pneumonia AI Predictor")
st.write("Enter your Image to identify")

# Sidebar for inputs
uploaded_scan = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_scan is not None:
    l_image = Image.open(uploaded_scan)
    st.image(l_image, caption="Uploaded Image", width=300)


# AI Button
if st.button("Run AI"):
    if uploaded_scan:
        try:
            with st.spinner("Analyzing image..."):                
                prob, label = asyncio.run(main(uploaded_scan))
                pass
            
            if (prob):
                # Result of AI prediction
                st.success("Image processed successfully!")
                st.write(f"Prediction: {label}")
                st.write(f"Probability of Pneumonia: {prob*100:.2f}%")
                
            else:
                st.warning("Image was not processed successfully, please try again.")
        except Exception as e:
            st.error("An error occurred while proccessing the image:")
            st.write(str(e))
    else:
        st.warning("Please upload an image.")
