import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


model = joblib.load('trained_model_name')   # Enter your Model name 
label_encoder = joblib.load('label_encoder_model_name') #  Enter model Encoder Name
file1_path = 'file1.txt'    # Enter your File Path

# Read the data from the text file into a list

with open(file1_path, 'r') as file:
    a_from_file = [line.strip() for line in file]

feature_values = []
for i in a_from_file:
    value = float(input(f"Enter Value of {i} :"))
    feature_values.append(value)

# Convert input values to DataFrame
input_df = pd.DataFrame([feature_values])

# Make predictions
predictions = model.predict(input_df)

    # Inverse transform predicted labels to get the class names
predicted_classes = label_encoder.inverse_transform(predictions)

print("Prediction : ",predicted_classes[0])