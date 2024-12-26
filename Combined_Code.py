import cv2
import numpy as np
import threading
from keras.models import load_model
from keras.layers import LSTM
from ultralytics import YOLO
import logging

# Set TensorFlow logging level
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Custom LSTM function to handle unsupported parameters
def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)  # Ignore the unsupported parameter
    return LSTM(*args, **kwargs)

# Load the behavioral analysis model
custom_objects = {"LSTM": custom_lstm}
behavior_model = load_model("model_best.h5", custom_objects=custom_objects, compile=False)

# Load the YOLO model for weapon detection
weapon_model = YOLO('best.pt')

# Parameters
num_frames = 30
frame_height = 64
frame_width = 64
channels = 1

# Shared webcam capture object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Error: Could not open webcam.")

# Function to preprocess frames for behavioral analysis
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.reshape(frame_height, frame_width, channels)

# Behavioral analysis thread function
def behavioral_analysis(skip_frames=5):
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam for behavioral analysis.")
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)

        label = "Normal"
        color = (0, 255, 0)  # Green for Normal

        if len(frames) == num_frames:
            # Prepare the video clip for prediction
            video_clip = np.array(frames).reshape(1, num_frames, frame_height, frame_width, channels)
            prediction = behavior_model.predict(video_clip)
            pred_value = prediction[0][0]

            # Determine the label and bounding box color
            if pred_value > 0.51:
                label = "Violence"
                color = (0, 0, 255)  # Red for Violence

            # Remove the first frame from the buffer
            frames.pop(0)

        # Add label to the frame
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Behavioral Analysis", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Weapon detection thread function
def weapon_detection(img_size=320):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam for weapon detection.")
            break

        # Predict on the frame
        results = weapon_model.predict(source=frame, imgsz=img_size, conf=0.8, show=False)

        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Draw bounding boxes on the frame for weapons
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label_name = weapon_model.names[cls_id]

                # Check if the detected object is a weapon (knife or gun)
                if label_name.lower() in ["knife", "gun"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    label = f"{label_name} {confidence:.2f}"

                    # Draw the bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for weapons
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the annotated frame
        cv2.imshow('Weapon Detection', annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start both threads
behavior_thread = threading.Thread(target=behavioral_analysis)
weapon_thread = threading.Thread(target=weapon_detection)

behavior_thread.start()
weapon_thread.start()

behavior_thread.join()
weapon_thread.join()

cap.release()
cv2.destroyAllWindows()
