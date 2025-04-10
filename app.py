import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
    }
    .subtitle {
        font-size: 1.5rem !important;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success {
        color: green;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .danger {
        color: red;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='title'>Face Mask Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image or use your webcam to detect face masks</p>", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_face_mask_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'face_mask_finally2.keras')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, compile=True)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Detailed error: {e}")
        return None

# Load face detection model
@st.cache_resource
def load_face_detector():
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Function to detect faces and predict mask
def detect_mask(image, face_detector, mask_model):
    # Convert image to RGB if it's not already
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Create a copy for drawing on
    output_image = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    
    # If no faces detected
    if len(faces) == 0:
        return output_image, "No faces detected", []
    
    results = []
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess for the mask detector
        face = cv2.resize(face_roi, (128, 128))  # Change to 128x128
        face = img_to_array(face)
        face = face / 255.0  # Use simple scaling instead of preprocess_input
        face = np.expand_dims(face, axis=0)
        
        # Make prediction
        prediction = mask_model.predict(face, verbose=0)
        
        # Determine label and color
        mask_probability = prediction[0][0]
        
        if mask_probability > 0.5:
            label = "Mask"
            color = (0, 255, 0)  # Green
            status = "With Mask"
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red
            status = "Without Mask"
        
        # Add to results
        results.append({
            "position": (x, y, w, h),
            "status": status,
            "probability": float(mask_probability)
        })
        
        # Draw bounding box and label
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    overall_status = "All wearing masks" if all(r["status"] == "With Mask" for r in results) else "Some people are not wearing masks"
    
    return output_image, overall_status, results

# Function to process uploaded image
def process_uploaded_image(uploaded_file, face_detector, mask_model):
    try:
        # Read image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Detect mask
        result_image, status, details = detect_mask(image, face_detector, mask_model)
        
        # Convert back to RGB for display
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        return result_image, status, details
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, "Error", []

# Function to process webcam frame
def process_webcam_frame(frame, face_detector, mask_model):
    try:
        # Detect mask
        result_image, status, details = detect_mask(frame, face_detector, mask_model)
        
        return result_image, status, details
    except Exception as e:
        st.error(f"Error processing webcam frame: {str(e)}")
        return frame, "Error", []

# Main function
def main():
    # Load models
    mask_model = load_face_mask_model()
    face_detector = load_face_detector()
    
    if mask_model is None or face_detector is None:
        st.warning("Please ensure the model file exists at the specified path and try again.")
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    # Tab 1: Upload Image
    with tab1:
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Masks", key="detect_upload"):
                with st.spinner("Processing..."):
                    # Process image
                    result_image, status, details = process_uploaded_image(uploaded_file, face_detector, mask_model)
                    
                    if result_image is not None:
                        # Display results
                        st.image(result_image, caption="Detection Result", use_column_width=True)
                        
                        # Display status
                        if "not wearing" in status.lower():
                            st.markdown(f"<p class='danger'>Status: {status}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='success'>Status: {status}</p>", unsafe_allow_html=True)
                        
                        # Display details
                        if details:
                            st.subheader("Detection Details")
                            for i, detail in enumerate(details):
                                with st.expander(f"Face {i+1}"):
                                    st.write(f"Status: {detail['status']}")
                                    st.write(f"Confidence: {detail['probability']:.2f}")
    
    # Tab 2: Webcam
    with tab2:
        st.subheader("Use Webcam")
        st.write("Note: This requires camera access in your browser.")
        
        # Webcam settings
        run_webcam = st.checkbox("Enable Webcam")
        
        if run_webcam:
            # Create a placeholder for the webcam feed
            webcam_placeholder = st.empty()
            
            # Start webcam
            try:
                cap = cv2.VideoCapture(0)
                
                # Check if webcam opened successfully
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera settings.")
                else:
                    st.success("Webcam started successfully!")
                    
                    # Create stop button
                    stop_button_pressed = st.button("Stop Webcam")
                    
                    # Process frames until stop button is pressed
                    while run_webcam and not stop_button_pressed:
                        # Read frame
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Failed to capture frame from webcam.")
                            break
                        
                        # Process frame
                        result_frame, status, details = process_webcam_frame(frame, face_detector, mask_model)
                        
                        # Convert to RGB for display
                        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        webcam_placeholder.image(result_frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Display status below the frame
                        if "not wearing" in status.lower():
                            color = "red"
                        else:
                            color = "green"
                        
                        webcam_placeholder.markdown(f"<h3 style='color: {color};'>Status: {status}</h3>", unsafe_allow_html=True)
                        
                        # Add a small delay
                        time.sleep(0.1)
                        
                        # Check if stop button was pressed
                        if stop_button_pressed:
                            break
                    
                    # Release webcam
                    cap.release()
                    
            except Exception as e:
                st.error(f"Error accessing webcam: {str(e)}")
                st.info("If you're running this in a virtual environment or cloud service, webcam functionality may not be available.")

# Run the app
if __name__ == "__main__":
    main()