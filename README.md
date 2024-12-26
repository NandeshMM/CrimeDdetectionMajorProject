# Crime Prevention Using AI/ML with CCTV Integration

This project uses AI/ML technologies to enhance existing CCTV networks for real-time crime prevention. It integrates behavioral analysis and weapon detection to identify threats like violent behavior and weapons, sending alerts via email for immediate action.

---

## Features

- **Behavioral Analysis**: Detects violent activities using a ConvoLSTM-based model (`model_best.h5`).
- **Weapon Detection**: Identifies weapons like knives and guns using a YOLOv8 model (`best.pt`).
- **Real-Time Notifications**: Sends email alerts for detected threats.

---

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python and computer vision concepts
- Webcam or video feed access

### Step 1: Clone the Repository

Clone this repository to your local machine:

git clone https://github.com/NandeshMM/CrimeDetectionMajorProject.git
cd CrimeDetectionMajorProject

### Step 2: Install Dependencies
Install all required dependencies using the provided requirements.txt file:

Run Following Command:-
pip install -r requirements.txt

### Step 3: Download Pretrained Models
The project requires two pretrained models to function:

best.pt: YOLOv8 model for weapon detection.
model_best.h5: ConvoLSTM model for behavioral analysis.
Download the models from the links below:

YOLOv8 Model (best.pt):

Download from ML_Model_File_Location

ConvoLSTM Model (model_best.h5):

Download from ML_Model_File_Location

## Email Notification System

### Step 1: Configure Email Alerts
Update the email settings in main.py:

python
Copy code
EMAIL_SENDER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_app_password'
EMAIL_RECEIVER = 'receiver_email@gmail.com'
Note: Use an app password for better security if you're using Gmail.

### Step 2: Start the System
Run the main script to start the detection system:

### Step 3: Stop the System
Press q in the video window to stop the system.

Dependencies
The project uses the following libraries:

opencv-python: For real-time video processing.
numpy: For numerical computations.
tensorflow and keras: For behavioral analysis using ConvoLSTM.
ultralytics: For YOLOv8-based weapon detection.
smtplib: For sending email notifications.
