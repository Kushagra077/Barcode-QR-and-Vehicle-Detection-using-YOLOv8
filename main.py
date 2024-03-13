import streamlit as st
from PIL import Image
from ultralytics import YOLO
import dill

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
    st.title("Barcode, QR Code and Vehicle Detection Application")

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
                try:
                    # Perform inference
                    results = perform_inference(model, uploaded_image)

                    # Visualize and save results
                    for i, r in enumerate(results):
                        st.subheader(f"Result")
                        
                        # Plot results image
                        im_bgr = r.plot()  # BGR-order numpy array
                        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                        st.image(im_rgb, caption=f"Result", use_column_width=True)

                        # Save results to disk
                        #filename = f'results_{selected_model}_{i}.jpg'
                        #r.save(filename=filename)
                        #st.markdown(f"Download [**{filename}**](/{filename})")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
