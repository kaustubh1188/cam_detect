import streamlit as st
import cv2
import os
from ultralytics import YOLO
import numpy as np
import tempfile
import io
from PIL import Image
import requests

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to process a single frame
def process_frame(model, frame, conf_threshold, iou_threshold):
    res = model.predict(
        frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}
    
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    latency = round(sum(res[0].speed.values()) / 1000, 2)
    prediction_text = "Predicted "
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}(s), ' if v > 1 else f'{v} {k}, '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"
    prediction_text += f' in {latency} seconds.'

    res_frame = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
    return res_frame, prediction_text

def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("Developed by Kaustubh Patel")
    st.markdown("<div class='title'>Wildfire Detection</div>", unsafe_allow_html=True)

    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    
    selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    input_type = st.radio("Select input type:", ("Image", "Video", "Live Stream"))
    
    if input_type == "Image":
        image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))
        image = None
        if image_source == "Upload from Computer":
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
        else:
            url = st.text_input("Enter the image URL:")
            if url:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
        
        if image:
            with st.spinner("Detecting..."):
                prediction, text = process_frame(model, np.array(image), conf_threshold, iou_threshold)
                st.image(prediction, caption="Prediction", use_column_width=True)
                st.success(text)

                # Convert image to a downloadable format
                prediction = Image.fromarray(prediction)
                image_buffer = io.BytesIO()
                prediction.save(image_buffer, format='PNG')
                st.download_button(label='Download Prediction', data=image_buffer.getvalue(), file_name='prediction.png', mime='image/png')
    
    elif input_type == "Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            tfile.close()  # Make sure to close the file after writing
            video = cv2.VideoCapture(tfile.name)
            
            with st.spinner("Processing video..."):
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break

                    processed_frame, prediction_text = process_frame(model, frame, conf_threshold, iou_threshold)
                    st.image(processed_frame, channels="RGB", use_column_width=True)
                    st.write(prediction_text)

                video.release()

    elif input_type == "Live Stream":
        hls_url = st.text_input("Enter the HLS stream URL (e.g., http://localhost:8080/stream.m3u8):")
        start_button = st.button("Start Stream")

        if start_button and hls_url:
            st.write("Stream should appear below if everything is set up correctly.")
            st.video(hls_url, format="application/x-mpegURL")

if __name__ == "__main__":
    main()
