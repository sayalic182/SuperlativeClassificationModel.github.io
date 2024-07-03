import cv2
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def capture_images(model_name, class_name, num_images):
    # Create the "Objects" directory if it doesn't exist
    objects_folder = os.path.join(os.getcwd(), f"media/{model_name}/Objects")
    if not os.path.exists(objects_folder):
        os.makedirs(objects_folder)

    # Create the specified folder within the "Objects" directory
    folder_path = os.path.join(objects_folder, class_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Counter for captured images
    count = 0

    # Display countdown timer on the frame
    for i in range(20, 0, -1):
        ret, frame = cap.read()
        frame_text = f"Capturing will start in: {i}"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1000)

    print("Capturing started...")

    while count < num_images:
        ret, frame = cap.read()
        
        # Resize the frame to 224x224
        frame_resized = cv2.resize(frame, (224, 224))

        # Display the frame
        cv2.imshow('Frame', frame_resized)

        # Save the resized frame to the specified folder
        image_name = os.path.join(folder_path, f"image_{count}.jpg")
        cv2.imwrite(image_name, frame_resized)

        count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def train_image(model_name, objects_folder, splits_folder, pre_trained=False):
    # Paths
    objects_dir = objects_folder
    splits_dir = splits_folder
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

    for class_name in class_names:
        for directory in [train_dir, test_dir, val_dir]:
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    # Split data into train, test, validation sets and move images
    for class_name in class_names:
        # List images for current class
        class_images = os.listdir(os.path.join(objects_dir, class_name))

        # Split data into train and test sets (80% train, 20% test)
        train_images, test_val_images = train_test_split(class_images, test_size=0.2, random_state=42)
        # Split remaining data into test and validation sets (50% test, 50% validation)
        test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        # Move images to appropriate directories
        for image in train_images:
            src = os.path.join(objects_dir, class_name, image)
            dst = os.path.join(train_dir, class_name, image)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(objects_dir, class_name, image)
            dst = os.path.join(test_dir, class_name, image)
            shutil.copy(src, dst)

        for image in val_images:
            src = os.path.join(objects_dir, class_name, image)
            dst = os.path.join(val_dir, class_name, image)
            shutil.copy(src, dst)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,  # Reduced zoom range
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # Specify this as training data
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # Specify this as validation data
    )

    if pre_trained:
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze pre-trained layers for fine-tuning
    else:
        base_model = None

    # Build model architecture (consider a deeper model for complex datasets)
    model = Sequential()
    if base_model:
        model.add(base_model)
    else:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))  # Increased number of neurons
    model.add(Dropout(0.5))
    model.add(Dense(len(train_generator.class_indices), activation='softmax'))

    # Model compilation
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        steps_per_epoch=int(train_generator.samples/train_generator.batch_size),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=int(validation_generator.samples/validation_generator.batch_size)
    )
    classes = train_generator.class_indices
    with open("jsons/" + model_name+".json", "w") as f:
        json.dump(classes, f)

    model.save("models/"+ model_name + ".h5")

# Load and preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict the class and confidence score
def predict_class(img_array, model):
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_index, confidence

# OOD detection threshold
threshold = 0.85  # Adjust as needed

# Function to detect OOD samples
def detect_ood(confidence, class_index):
    if confidence < threshold:
        return True, None  # Input is OOD
    else:
        return False, class_index  # Input is in-distribution

def test_image_model(model, class_labels):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img_array = preprocess_image(frame)

        # Predict the class and confidence score
        class_index, confidence = predict_class(img_array, model)

        # Detect OOD samples
        is_ood, _ = detect_ood(confidence, class_index)

        if is_ood:
            cv2.putText(frame, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if class_index is not None:
                class_name = list(class_labels.keys())[class_index]
                cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Live Camera', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
