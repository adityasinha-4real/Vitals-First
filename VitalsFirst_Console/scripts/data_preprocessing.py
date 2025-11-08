import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Encode Gender as 0/1 to keep things simple
    gender_map = {'F': 0, 'M': 1}
    df['Gender'] = df['Gender'].map(gender_map)

    # Encode the target using LabelEncoder and remember classes
    y_encoder = LabelEncoder()
    df['Final_Triage_Label'] = y_encoder.fit_transform(df['Final_Triage_Label'])

    X = df.drop('Final_Triage_Label', axis=1)
    y = df['Final_Triage_Label']

    return X, y, y_encoder

def encode_single_input(payload: dict):
    # Expecting keys: Age, Gender('M'/'F'), Body_Temperature, Heart_Rate, Respiratory_Rate,
    # Blood_Pressure_Systolic, Blood_Pressure_Diastolic, Oxygen_Saturation, Symptom_Score
    gender_map = {'F': 0, 'M': 1}
    payload = payload.copy()
    payload['Gender'] = gender_map.get(payload.get('Gender','M'), 1)
    return pd.DataFrame([payload])
