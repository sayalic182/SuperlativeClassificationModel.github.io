import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def csv_trained(model_name, uploaded_file):
    # Load the dataset into a DataFrame
    df = pd.read_csv(uploaded_file)  
    # Extract features (X) and labels (Y)
    X = df.iloc[:, :-1]  # All columns except the last one
    Y = df.iloc[:, -1]   # Last column

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    # Get feature names
    feature_names = X.columns
    trained_model_name = f"models/{model_name}.pkl"
    label_encoder_model_name = f"models/label_encoder_{model_name}.pkl"

    # Save the trained model
    joblib.dump(model, trained_model_name)
    joblib.dump(label_encoder, label_encoder_model_name)

    return trained_model_name, feature_names, label_encoder_model_name


def test_model(trained_model_name, feature_values, label_encoder_model_name):
    
    model = joblib.load(trained_model_name)
    label_encoder = joblib.load(label_encoder_model_name)

    # Convert input values to DataFrame
    input_df = pd.DataFrame([feature_values])

    # Make predictions
    predictions = model.predict(input_df)

     # Inverse transform predicted labels to get the class names
    predicted_classes = label_encoder.inverse_transform(predictions)
    
    return predicted_classes
