# 🩺 VitalsFirst Console

## Overview
**VitalsFirst** is a machine-learning–driven triage system designed to support hospital staff in classifying patients according to medical urgency.  
This **Console version** represents the backend component of the VitalsFirst ecosystem, developed using **FastAPI** and **scikit-learn**.  
It enables healthcare institutions to predict triage categories based on vital signs and patient information, while also handling data preprocessing, model inference, and alert generation.

---

## 🎯 Objectives
- Automate patient triage through real-time data processing.  
- Assist nurses and doctors in prioritizing cases using data-driven risk scoring.  
- Provide explainable AI insights (e.g., SHAP analysis, ROC, and precision-recall curves).  
- Enable easy model retraining and integration with the web frontend.

---

## 🧠 System Architecture
The system follows a modular pipeline:
1. **Data Ingestion** — Reads CSV datasets (patients, triage, staff, alerts).  
2. **Preprocessing** — Cleans, encodes, and normalizes data via `data_preprocessing.py`.  
3. **Model Training** — Uses `train_model.py` to fit a Random Forest classifier on preprocessed triage data.  
4. **Prediction** — `predict_triage.py` loads the model (`random_forest_model.pkl`) and predicts triage categories.  
5. **API Layer** — A FastAPI interface exposes `/predict` and `/train` endpoints for external access (integration-ready).  

---

## 🏗️ Folder Structure
```
VitalsFirst_Console/
├── data/                → Input CSV datasets
├── models/              → Trained ML models
├── scripts/             → Data & ML scripts
├── requirements.txt     → Project dependencies
├── README.md            → Project documentation
├── venv/                → Virtual environment (optional)
└── .git/                → Git repository metadata
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or higher  
- pip  
- (Optional) Virtual environment  

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/VitalsFirst_Console.git
cd VitalsFirst_Console

# Create virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 1. Start the FastAPI Server
```bash
uvicorn main:app --reload
```
(Default port: `8000`)

Access the interactive API docs at:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Run Model Training Manually
```bash
python scripts/train_model.py
```

### 3. Make Predictions
You can test predictions via:
```bash
python scripts/predict_triage.py
```
or send POST requests to the `/predict` endpoint.

---

## 🧩 Key Components

| File | Description |
|------|--------------|
| `scripts/data_preprocessing.py` | Handles missing values, encodes categorical data, normalizes numeric features. |
| `scripts/train_model.py` | Trains and serializes Random Forest model and label encoder. |
| `scripts/predict_triage.py` | Loads trained model to generate predictions on new patient data. |
| `models/random_forest_model.pkl` | Saved trained model. |
| `models/label_encoder.pkl` | Encodes categorical triage labels. |
| `data/*.csv` | Sample datasets for testing and retraining. |

---

## 📊 Machine Learning Model
- **Algorithm:** Random Forest Classifier  
- **Input Features:** Patient vitals, symptoms, demographics, and condition descriptors.  
- **Target:** Triage category (e.g., Immediate, Urgent, Non-urgent).  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC.  
- **Explainability Tools:** SHAP dependence plot and feature importance visualization.

---

## 🔗 Integration
The Console backend integrates seamlessly with the **VitalsFirst Frontend**, enabling clinicians to:
- Submit patient data through a web interface.  
- Receive triage predictions in real time.  
- View alerts and scheduling updates from the same database.

---

## 🧪 Example API Request
```json
POST /predict
{
  "heart_rate": 120,
  "temperature": 102.5,
  "oxygen_saturation": 92,
  "blood_pressure": "140/90",
  "symptom": "Chest pain",
  "age": 55
}
```

**Response:**
```json
{
  "predicted_triage": "High Priority",
  "confidence": 0.87
}
```

---

## 🛡️ Future Enhancements
- Integration with IoT-based vitals monitoring devices.  
- Live SHAP explainability dashboard.  
- Cloud deployment using Docker and AWS Lambda.  
- Integration with hospital EHR systems for direct data sync.  

---

## 📜 License
This project is distributed for educational and research purposes.  
All rights reserved © 2025 VitalsFirst Team.
