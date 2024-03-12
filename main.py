import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO models
vehicle_model = YOLO('vehicle.pt')
barcode_model = YOLO('barcode.pt')

# Function to perform inference based on model selection
def perform_inference(model, uploaded_image):
    image = Image.open(uploaded_image)
    results = model(image)
    return results

# Streamlit app
def main():
    st.title("Object Detection App")

    # Sidebar - Model selection
    selected_model = st.sidebar.radio("Select Model", ("Barcode-QR", "Vehicle"))

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Run inference based on selected model
        if selected_model == "Barcode-QR":
            model = barcode_model
        else:
            model = vehicle_model

        if st.button("Run Detection"):
            with st.spinner('Running inference...'):
                results = perform_inference(model, uploaded_image)
                for result in results:
                    result.show()

if __name__ == "__main__":
    main()
