import cv2
import os

# Define the source and destination folders
source_folder = 'E:\\Crime_Detection_Videos_Dataset_13_Dec\\violent'
destination_folder = 'E:\\Crime_Detection_Videos_Dataset_13_Dec\\violent_frames'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(('.mp4', '.avi', '.mov')):
        # Read the video
        cap = cv2.VideoCapture(os.path.join(source_folder, filename))

        # Create a folder for frames of the current video
        video_frame_folder = os.path.join(destination_folder, os.path.splitext(filename)[0])
        if not os.path.exists(video_frame_folder):
            os.makedirs(video_frame_folder)

        frame_count = 0

        # Extract and save frames
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(video_frame_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
            else:
                break

        # Release the video capture object
        cap.release()
    else:
        continue

print("Frames extraction completed!")