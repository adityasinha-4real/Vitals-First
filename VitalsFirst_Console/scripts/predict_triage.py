import os
import joblib
from data_preprocessing import encode_single_input

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run: python scripts/train_model.py")
        return

    rf = joblib.load(MODEL_PATH)
    y_encoder = joblib.load(ENCODER_PATH)

    print("\n=== VitalsFirst: Console Triage Prediction ===")
    try:
        age = int(input("Age: "))
        gender = input("Gender (M/F): ").strip().upper()
        temp = float(input("Body Temperature (°F): "))
        hr = int(input("Heart Rate (bpm): "))
        rr = int(input("Respiratory Rate: "))
        sys_bp = int(input("Systolic BP: "))
        dia_bp = int(input("Diastolic BP: "))
        o2 = int(input("Oxygen Saturation (%): "))
        sym = int(input("Symptom Score (0-10): "))
    except Exception as e:
        print("Input error:", e)
        return

    payload = {
        'Age': age,
        'Gender': gender,
        'Body_Temperature': temp,
        'Heart_Rate': hr,
        'Respiratory_Rate': rr,
        'Blood_Pressure_Systolic': sys_bp,
        'Blood_Pressure_Diastolic': dia_bp,
        'Oxygen_Saturation': o2,
        'Symptom_Score': sym
    }

    X = encode_single_input(payload)
    pred_enc = rf.predict(X)[0]
    pred_label = y_encoder.inverse_transform([pred_enc])[0]

    print(f"\nPredicted Triage Level: {pred_label}")

if __name__ == '__main__':
    main()
