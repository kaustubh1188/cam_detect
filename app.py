import streamlit as st
import cv2
from ultralytics import YOLO
import requests
from PIL import Image
import os
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    try:
        # Check if YOLO has an option to disable signal handling
        model = YOLO(model_path)
        return model
    except ValueError as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    if model is None:
        st.error("Model not loaded properly.")
        return None, "Model loading error."
    
    try:
        # Predict objects using the model
        res = model.predict(image, conf=conf_threshold, iou=iou_threshold, device='cpu')
        
        class_name = model.model.names
        classes = res[0].boxes.cls
        class_counts = {}
        
        # Count the number of occurrences for each class
        for c in classes:
            c = int(c)
            class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

        # Generate prediction text
        prediction_text = 'Predicted '
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            prediction_text += f'{v} {k}'
            if v > 1:
                prediction_text += 's'
            prediction_text += ', '

        prediction_text = prediction_text[:-2]
        if len(class_counts) == 0:
            prediction_text = "No objects detected"

        # Calculate inference latency
        latency = sum(res[0].speed.values())  # in ms, need to convert to seconds
        latency = round(latency / 1000, 2)
        prediction_text += f' in {latency} seconds.'

        # Convert the result image to RGB
        res_image = res[0].plot()
        res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
        
        return res_image, prediction_text
    
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
        return None, "Prediction error."
