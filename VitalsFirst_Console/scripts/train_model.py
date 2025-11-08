import os
import warnings
warnings.filterwarnings("ignore")

import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
from data_preprocessing import load_and_preprocess_data

# === File paths ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "triage.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def main():
    # Load and preprocess data
    X, y, y_encoder = load_and_preprocess_data(DATA_PATH)

    # Safe split (only stratify if each class has >= 2 samples)
    class_counts = Counter(y)
    if all(v >= 2 for v in class_counts.values()):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("⚠️ Warning: One or more classes have too few samples. Proceeding without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # === Evaluation Metrics ===
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC (macro, OvR) - requires probabilities
    try:
        y_proba = rf.predict_proba(X_test)
        roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_macro = None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=y_encoder.classes_,
        columns=y_encoder.classes_
    )

    # === Print Results ===
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"Macro ROC-AUC (OvR): {roc_auc_macro:.4f}")
    else:
        print("Macro ROC-AUC (OvR): N/A (insufficient class coverage in test set)")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=y_encoder.classes_))

    print("\nConfusion Matrix:")
    print(cm_df)

    # === Save model + encoder ===
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(y_encoder, ENCODER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved label encoder to {ENCODER_PATH}")

if __name__ == "__main__":
    main()
