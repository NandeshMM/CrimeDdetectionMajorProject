import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

def preprocess_video_frames_with_labels(video_folder, category, label, frame_size=(32, 32), sequence_length=10):
    """
    Preprocess video frames and assign labels for a specific category (violent or non-violent).
    Args:
        video_folder: Path to the folder containing video frames organized in 'violent' and 'non-violent' subfolders.
        category: The category name ('violent' or 'non-violent').
        label: The label associated with the category (1 for violent, 0 for non-violent).
        frame_size: Tuple specifying the size of each frame (height, width).
        sequence_length: Number of frames in each sequence.
    Returns:
        Tuple of numpy arrays (sequences, labels) for the given category.
    """
    sequences = []
    labels = []
    category_path = os.path.join(video_folder, category)
    print(f"Processing category: {category_path}")

    for video_folder in tqdm(os.listdir(category_path), desc=f"Processing {category} videos", unit="video"):
        video_path = os.path.join(category_path, video_folder)

        # Check if the item is a valid directory
        if os.path.isdir(video_path):
            frames = []
            for frame_file in sorted(os.listdir(video_path)):
                frame_path = os.path.join(video_path, frame_file)
                if os.path.isfile(frame_path):  # Check if it's a file (not a folder)
                    frame = load_img(frame_path, target_size=frame_size)
                    frame = img_to_array(frame) / 255.0  # Normalize pixel values
                    frames.append(frame)

            # Split into sequences of length `sequence_length`
            for i in range(len(frames) - sequence_length + 1):
                sequences.append(frames[i:i + sequence_length])
                labels.append(label)
        else:
            print(f"Skipping non-directory: {video_folder}")

    return np.array(sequences), np.array(labels)

# Step 2: Split Data by Category
def split_data_by_category(sequences, labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Split data into training, validation, and testing sets by category.
    Args:
        sequences: Numpy array of video sequences.
        labels: Numpy array of corresponding labels.
        train_ratio: Proportion of data to be used for training.
        val_ratio: Proportion of data to be used for validation.
        test_ratio: Proportion of data to be used for testing.
    Returns:
        Separate training, validation, and testing datasets for each category.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    # Split into violent and non-violent datasets
    violent_sequences = sequences[labels == 1]
    nonviolent_sequences = sequences[labels == 0]

    # Split violent data
    violent_train, violent_temp = train_test_split(violent_sequences, test_size=(val_ratio + test_ratio), random_state=42)
    violent_val, violent_test = train_test_split(violent_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Split non-violent data
    nonviolent_train, nonviolent_temp = train_test_split(nonviolent_sequences, test_size=(val_ratio + test_ratio), random_state=42)
    nonviolent_val, nonviolent_test = train_test_split(nonviolent_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    return (violent_train, violent_val, violent_test), (nonviolent_train, nonviolent_val, nonviolent_test)

# Example usage for preprocessing and splitting

# Paths for violent and non-violent video frames
video_folder_path = "E:\\Crime_Detection_Frames_Data_Temp_14_Dec"

# Preprocess video frames separately for violent and non-violent categories
violent_sequences, violent_labels = preprocess_video_frames_with_labels(video_folder_path, category='violent', label=1)
nonviolent_sequences, nonviolent_labels = preprocess_video_frames_with_labels(video_folder_path, category='non-violent', label=0)

# Combine violent and non-violent data
sequences = np.concatenate((violent_sequences, nonviolent_sequences), axis=0)
labels = np.concatenate((violent_labels, nonviolent_labels), axis=0)

# Split data into violent and non-violent sets
(violent_train, violent_val, violent_test), (nonviolent_train, nonviolent_val, nonviolent_test) = split_data_by_category(sequences, labels)

# Save the datasets to disk
output_dir = "E:\\Crime_Detection_Frames_Dataset_13_Dec"  # Using raw string for Windows path
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "violent_train.npy"), violent_train)
np.save(os.path.join(output_dir, "violent_val.npy"), violent_val)
np.save(os.path.join(output_dir, "violent_test.npy"), violent_test)

np.save(os.path.join(output_dir, "nonviolent_train.npy"), nonviolent_train)
np.save(os.path.join(output_dir, "nonviolent_val.npy"), nonviolent_val)
np.save(os.path.join(output_dir, "nonviolent_test.npy"), nonviolent_test)

print("Data saved:")
print(f"- Violent: Train: {len(violent_train)}, Validation: {len(violent_val)}, Test: {len(violent_test)}")
print(f"- Non-Violent: Train: {len(nonviolent_train)}, Validation: {len(nonviolent_val)}, Test: {len(nonviolent_test)}")
