#########################################################################
#                    Pose Classification Model Code                    #
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
import mediapipe as mp
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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


knn = joblib.load(f'knn_model.pkl')  # Enter path Properly
le = joblib.load(f'label_encoder.pkl')  # Enter path Properly

test_pose(knn, le)