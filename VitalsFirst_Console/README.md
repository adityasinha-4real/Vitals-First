# 🏥 VitalsFirst – AI-Driven Patient Triage Console System

## 📘 Project Overview
VitalsFirst is a **console-based Python project** that simulates an AI-powered patient triage and scheduling system for hospitals.  
It uses a **Random Forest Classifier** trained on synthetic medical data to predict patient triage levels — **Stable**, **Urgent**, or **Critical** — based on vital signs and symptom severity.  

This project demonstrates basic **machine learning workflow implementation in Python** without requiring advanced ML knowledge, making it ideal for students and beginners in AI courses.

---

## 🧩 Project Structure
```
VitalsFirst_Console/
│
├── data/
│   ├── patients.csv          # Patient demographic and medical info
│   ├── triage.csv            # Training data for ML model
│   ├── staff.csv             # Doctor/Nurse/Admin allocation data
│   ├── schedule.csv          # Appointments and scheduling data
│   └── alerts.csv            # System-generated notifications
│
├── models/
│   ├── random_forest_model.pkl   # Trained Random Forest model
│   └── label_encoder.pkl         # Encoded label mapping
│
├── scripts/
│   ├── data_preprocessing.py     # Data encoding and helper functions
│   ├── train_model.py            # Model training and evaluation script
│   └── predict_triage.py         # Console prediction program
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       # For Windows
# or
# source venv/bin/activate   # For macOS/Linux
```

### 2️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Random Forest Model
```bash
python scripts/train_model.py
```

**Output Example:**
```
⚠️ Warning: One or more classes have too few samples. Proceeding without stratify.
Accuracy: 0.96
Saved model to models/random_forest_model.pkl
Saved label encoder to models/label_encoder.pkl
```

---

## 🧠 How the Model Works
The **Random Forest Classifier** is trained using patient vital signs and symptom scores.  
It learns simple patterns, for example:
- High temperature + rapid heart rate + low oxygen = **Critical**
- Mild symptoms + normal vitals = **Stable**

Each triage label helps prioritize medical attention efficiently.

---

## 🧮 Prediction (Console-Based)
After training, run:
```bash
python scripts/predict_triage.py
```

Then enter patient details when prompted, for example:
```
Age: 60
Gender (M/F): M
Body Temperature (°F): 102
Heart Rate (bpm): 112
Respiratory Rate: 28
Systolic BP: 155
Diastolic BP: 95
Oxygen Saturation (%): 90
Symptom Score (0-10): 8
```

**Output:**
```
Predicted Triage Level: Critical
```

---

## 📊 Sample Model Output
| Metric | Score |
|---------|--------|
| Accuracy | ~96% |
| Supported Labels | Stable, Urgent, Critical |
| Algorithm | Random Forest Classifier (n_estimators=120) |

---

## 🧰 Key Libraries Used
| Library | Purpose |
|----------|----------|
| `pandas` | CSV data handling |
| `scikit-learn` | ML model training and evaluation |
| `joblib` | Model serialization |
| `os` | Path management |

---

## 💡 Learning Outcomes
By completing this project, students learn how to:
- Preprocess data for ML models  
- Train, evaluate, and save models in Python  
- Use Random Forest Classifier for classification problems  
- Build and run a console-based AI system in VS Code  

---

## 🚀 Future Enhancements
- Add a **Streamlit-based UI** for a web interface  
- Integrate **database (SQLite/MySQL)** for real data storage  
- Implement **real-time alert and scheduling modules**  
- Expand to multi-user access (Nurse, Doctor, Admin, Patient login)

---

## 👨‍💻 Author & Credits
Developed by: *[Your Name]*  
Course: **Artificial Intelligence – Semester Project (Python)**  
Version: **VitalsFirst Console Edition (v1.0)**  
Python Version: **3.11.9**
