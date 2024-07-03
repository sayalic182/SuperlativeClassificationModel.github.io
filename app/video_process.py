import cv2
import os
import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import json
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Custom data generator for videos
def video_data_generator(directory, batch_size, target_size=(224, 224), shuffle=True):
    video_folders = os.listdir(directory)
    num_classes = len(video_folders)
    class_indices = {class_name: i for i, class_name in enumerate(video_folders)}

    while True:
        if shuffle:
            np.random.shuffle(video_folders)

        for class_name in video_folders:
            class_dir = os.path.join(directory, class_name)
            videos = os.listdir(class_dir)
            if shuffle:
                np.random.shuffle(videos)

            frames = []
            labels = []
            for video in videos[:batch_size // num_classes]:
                video_path = os.path.join(class_dir, video)
                frames.extend(extract_frames(video_path, target_size))
                labels.extend([class_indices[class_name]] * len(frames))

                if len(frames) >= batch_size:
                    frames = np.array(frames[:batch_size])
                    labels = np.array(labels[:batch_size])
                    yield frames, labels
                    frames = []
                    labels = []

def extract_frames(video_path, target_size):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    return frames


def capture_videos(model_name, class_name, num_videos, video_duration):
    objects_folder = os.path.join(os.getcwd(), f"static/videos/{model_name}/Objects")

    if not os.path.exists(objects_folder):
        os.makedirs(objects_folder)

    # Create the specified folder within the "Objects" directory
    folder_path = os.path.join(objects_folder, class_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Wait for 20 seconds before initializing the camera
    start_time = time.time()
    while time.time() - start_time < 20:
        pass

    cap = cv2.VideoCapture(0)

    # Set video width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_name = os.path.join(folder_path, f"video_0.mp4")
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

    count = 0

    try:
        while count < num_videos:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Display the frame
            cv2.imshow('Frame', frame)

            # Write the frame to the video
            out.write(frame)

            # Check if video duration exceeds the specified time
            if time.time() - start_time >= video_duration:
                out.release()
                count += 1
                if count < num_videos:
                    video_name = os.path.join(folder_path, f"video_{count}.mp4")
                    out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))
                    start_time = time.time()

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def train_video(model_name, objects_dir,splits_dir):
    train_dir = os.path.join(splits_dir, "train")
    test_dir = os.path.join(splits_dir, "test")
    val_dir = os.path.join(splits_dir, "validation")

    # Create Splits directory if not exists
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    # Get the list of class names
    class_names = os.listdir(objects_dir)

    # Create train, test, validation directories
    for directory in [train_dir, test_dir, val_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create subdirectories for each class in train, test, validation directories
    for class_name in class_names:
        for directory in [train_dir, test_dir, val_dir]:
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    class_names = os.listdir(objects_dir)
    print(class_names)

    # Split data into train, test, validation sets and move videos
    for class_name in class_names:
        # List videos for the current class
        class_videos = [video for video in os.listdir(os.path.join(objects_dir, class_name)) if video.endswith(".mp4")]

        # Split data into train and test sets (80% train, 20% test)
        train_videos, test_val_videos = train_test_split(class_videos, test_size=0.2, random_state=42)
        # Split remaining data into test and validation sets (50% test, 50% validation)
        test_videos, val_videos = train_test_split(test_val_videos, test_size=0.5, random_state=42)

        # Move videos to appropriate directories
        for video in train_videos:
            src = os.path.join(objects_dir, class_name, video)
            dst = os.path.join(train_dir, class_name, video)
            shutil.copy(src, dst)

        for video in test_videos:
            src = os.path.join(objects_dir, class_name, video)
            dst = os.path.join(test_dir, class_name, video)
            shutil.copy(src, dst)

        for video in val_videos:
            src = os.path.join(objects_dir, class_name, video)
            dst = os.path.join(val_dir, class_name, video)
            shutil.copy(src, dst)

    # Define the directory paths
    train_videos_dir = splits_dir + "/train"  # Update with your actual directory path
    validation_videos_dir = splits_dir + "/validation"

    # Example usage:
    batch_size = 32
    train_generator = video_data_generator(train_videos_dir, batch_size)
    validation_generator = video_data_generator(validation_videos_dir, batch_size)

    # Example of using the generators for training
    for i, (X_train, y_train) in enumerate(train_generator):
        # Perform training steps using X_train and y_train
        print(f"Batch {i+1}: Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        if i == 4:  # Stop after 5 batches for demonstration
            break

    for i, (X_val, y_val) in enumerate(validation_generator):
        # Perform validation steps using X_val and y_val
        print(f"Batch {i+1}: Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
        if i == 4:  # Stop after 5 batches for demonstration
            break

        # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')  # Assuming len(class_names) gives the number of classes
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Manually count the number of samples in the training and validation sets
    train_samples = sum(len(os.listdir(os.path.join(train_videos_dir, class_name))) for class_name in class_names)
    validation_samples = sum(len(os.listdir(os.path.join(validation_videos_dir, class_name))) for class_name in class_names)
    print(train_samples)
    print(validation_samples)

    epochs = 10  # You can adjust the number of epochs as needed
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32
    )

    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    with open("static/json/" + model_name +".json", "w") as f:
        json.dump(class_indices, f)

    # Save the trained model
    model.save("static/models/" + model_name + '.h5')


# Function to preprocess video frames for prediction
def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Function to capture video from the live camera and display predicted class
def test_video_model(model, class_names, target_size=(224, 224), confidence_threshold=0.95):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame, target_size)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(preprocessed_frame)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        if confidence > confidence_threshold:
            predicted_class_name = class_names[predicted_class_idx]
        else:
            predicted_class_name = "Unknown"

        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {predicted_class_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Live Camera', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()