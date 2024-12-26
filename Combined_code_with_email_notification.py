import cv2
import numpy as np
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from keras.models import load_model
from keras.layers import LSTM
from ultralytics import YOLO
import logging

# Set TensorFlow logging level
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'  # SMTP server for Gmail
SMTP_PORT = 587
EMAIL_SENDER = 'nandeshcollege@gmail.com'
EMAIL_PASSWORD = 'voze csed ixao rreq'  # Use an app password for security
EMAIL_RECEIVER = 'majorproject0987@gmail.com'

# Custom LSTM function to handle unsupported parameters
def custom_lstm(*args, **kwargs):
    kwargs.pop('time_major', None)  # Ignore the unsupported parameter
    return LSTM(*args, **kwargs)

# Load models
custom_objects = {"LSTM": custom_lstm}
behavior_model = load_model("model_best.h5", custom_objects=custom_objects, compile=False)
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

# Function to send email notifications
def send_email(subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Behavioral analysis thread function
def behavioral_analysis(skip_frames=5):
    global violence_detected
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
            video_clip = np.array(frames).reshape(1, num_frames, frame_height, frame_width, channels)
            prediction = behavior_model.predict(video_clip)
            pred_value = prediction[0][0]

            if pred_value > 0.51:
                label = "Violence"
                color = (0, 0, 255)  # Red for Violence
                violence_detected = True
            else:
                violence_detected = False

            frames.pop(0)

        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Behavioral Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Weapon detection thread function
def weapon_detection(img_size=320):
    global weapon_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam for weapon detection.")
            break

        results = weapon_model.predict(source=frame, imgsz=img_size, conf=0.3, show=False)
        annotated_frame = frame.copy()
        weapon_detected = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label_name = weapon_model.names[cls_id]

                if label_name.lower() in ["knife", "gun"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label = f"{label_name} {confidence:.2f}"
                    weapon_detected = True

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Weapon Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Monitor for alert and danger messages
def monitor_notifications():
    global violence_detected, weapon_detected
    violence_detected = False
    weapon_detected = False

    while True:
        if violence_detected and weapon_detected:
            send_email("Danger Alert", "Both violence and a weapon have been detected. Immediate action required!")
        elif violence_detected or weapon_detected:
            send_email("Alert", "Either violence or a weapon has been detected. Please investigate.")

# Start threads
behavior_thread = threading.Thread(target=behavioral_analysis)
weapon_thread = threading.Thread(target=weapon_detection)
notification_thread = threading.Thread(target=monitor_notifications, daemon=True)

# Start the threads for behavioral analysis, weapon detection, and email notifications
behavior_thread.start()
weapon_thread.start()
notification_thread.start()

# Join threads to keep the program running until all threads are done
behavior_thread.join()
weapon_thread.join()

cap.release()
cv2.destroyAllWindows()
