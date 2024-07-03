#########################################################################
#                    Video Classification Model Code                    #
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


# Load the trained model
model = load_model("video_classification_model.h5") # Add Your Trained Model Path

# Load class labels from JSON file
with open("class_labels.json", "r") as f: # Add path of json file
    class_labels = json.load(f)

class_names = {v: k for k, v in class_labels.items()}
test_video_model(model, class_names)