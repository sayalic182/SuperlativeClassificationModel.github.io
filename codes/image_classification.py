#########################################################################
#                    Image Classification Model Code                    #
#########################################################################

########################## Required Libraries  ##########################
# pip install opencv-python                                             #
# pip install tensorflow                                                #
# pip install json                                                      #
# pip install shutil                                                    #
# ppip install scikit-learn                                             #
# pip install json                                                      #
# pip install numpy                                                     #
#                                                                       #
#########################################################################


import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# Load the trained model
model = load_model("image_classification_model.h5") # Add Your Trained Model Path

# Load class labels from JSON file
with open("class_labels.json", "r") as f: # Add path of json file
    class_labels = json.load(f)

# Load and preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict the class and confidence score
def predict_class(img_array):
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_index, confidence

# OOD detection threshold
threshold = 0.96  # Adjust as needed

# Function to detect OOD samples
def detect_ood(confidence):
    if confidence < threshold:
        return True, None  # Input is OOD
    else:
        return False, class_index  # Input is in-distribution

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img_array = preprocess_image(frame)

    # Predict the class and confidence score
    class_index, confidence = predict_class(img_array)

    # Detect OOD samples
    is_ood, _ = detect_ood(confidence)

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
