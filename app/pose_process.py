import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def collect_data(pose_name, capture_duration, settle_duration):
    cap = cv2.VideoCapture(0)
    landmarks_list = []
    start_time = time.time()
    capture_start_time = start_time + settle_duration

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the video feed during the settlement period
            current_time = time.time()
            if current_time < capture_start_time:
                cv2.putText(image, f'Please settle into your pose.....{int(capture_start_time - current_time)}s left', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif current_time - capture_start_time < capture_duration:
                landmarks = results.pose_landmarks.landmark
                landmarks_row = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
                landmarks_row.append(pose_name)
                landmarks_list.append(landmarks_row)
                cv2.putText(image, 'Recording...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                break

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return landmarks_list

def save_data(data, filename):
    columns = [f'x{i}' for i in range(33)] + [f'y{i}' for i in range(33)] + ['pose']
    df = pd.DataFrame(data, columns=columns)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:  # append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)


def train_pose(pose_data):
    # Prepare the data
    X = pose_data.iloc[:, :-1]
    y = pose_data.iloc[:, -1]

    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    return knn, le
    

def test_pose(knn, le):
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                landmarks_row = np.array([[landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]])
                pose_prediction = knn.predict(landmarks_row)
                pose_label = le.inverse_transform(pose_prediction)[0]

                # Ensure the label is a string
                pose_label_str = str(pose_label)
                
                cv2.putText(image, f"Class : {pose_label_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
