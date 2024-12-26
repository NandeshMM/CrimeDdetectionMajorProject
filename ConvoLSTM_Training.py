import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_fscore_support
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, MaxPooling3D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf

# Function to load your sequences and labels from .npy files
def load_data():
    # Check if files exist
    for file in ['violent_train.npy', 'nonviolent_train.npy', 'violent_test.npy', 'nonviolent_test.npy']:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found. Please ensure the file is in the correct directory.")
    
    # Load preprocessed frames
    violent_train = np.load('violent_train.npy')  # Shape: [samples, timesteps, height, width, channels]
    nonviolent_train = np.load('nonviolent_train.npy')
    violent_test = np.load('violent_test.npy')
    nonviolent_test = np.load('nonviolent_test.npy')

    # Concatenate training and testing data
    train_data = np.concatenate((violent_train, nonviolent_train), axis=0)
    train_labels = np.concatenate((np.ones(len(violent_train)), np.zeros(len(nonviolent_train))), axis=0)
    test_data = np.concatenate((violent_test, nonviolent_test), axis=0)
    test_labels = np.concatenate((np.ones(len(violent_test)), np.zeros(len(nonviolent_test))), axis=0)

    return train_data, train_labels, test_data, test_labels

# Define your ConvLSTM model
def create_model(input_shape):
    model = Sequential()

    # First ConvLSTM2D layer with return_sequences=True
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())

    # Second ConvLSTM2D layer with return_sequences=False
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
    model.add(BatchNormalization())

    # Reshape the output to add a 'depth' dimension for Conv3D
    model.add(tf.keras.layers.Reshape((1, input_shape[1], input_shape[2], 64)))  # Adding depth dimension

    # 3D convolutional layer
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same'))

    # 3D MaxPooling layer
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))  # Pooling only along height and width dimensions

    # Output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(model, train_data, train_labels, test_data, test_labels, epochs=20, batch_size=8):
    history = model.fit(
        train_data, train_labels,
        validation_data=(test_data, test_labels),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

# Plot training and validation metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Evaluate model performance
def evaluate_model(model, test_data, test_labels):
    predictions = (model.predict(test_data) > 0.5).astype(int).flatten()

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-violent', 'Violent'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    report = classification_report(test_labels, predictions, target_names=['Non-violent', 'Violent'])
    print(report)

    # Precision, Recall, F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Example usage
train_data, train_labels, test_data, test_labels = load_data()

# Ensure data has 5 dimensions: [samples, timesteps, height, width, channels]
if train_data.ndim != 5:
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

# Create and train the model
model = create_model(input_shape=train_data.shape[1:])
history = train_model(model, train_data, train_labels, test_data, test_labels)

# Save the model
model_save_path = "crime_detection_model.h5"
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Plot metrics
plot_metrics(history)

# Evaluate the model
evaluate_model(model, test_data, test_labels)
