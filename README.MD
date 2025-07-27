# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using various techniques like **data preprocessing, SMOTE (Synthetic Minority Oversampling Technique), and Random Forest classification**.

---

## **📌 Project Overview**
- Credit card fraud detection is a classic **imbalanced classification problem**, where fraudulent transactions are extremely rare compared to genuine ones.
- This project demonstrates:
  - Data preprocessing with **Pandas & NumPy**.
  - Handling class imbalance using **SMOTE**.
  - Model training with **Random Forest**.
  - Evaluation using **precision, recall, and F1-score** instead of plain accuracy.

---

## **📂 Project Structure**
ccfraud_detection/
│
├── fraud_detection/
│ ├── fraud_detection.py # Main script for data preprocessing, training & evaluation
│
├── requirements.txt # List of dependencies
└── README.md # Project documentation

---

## **⚙️ Tech Stack**
- **Python 3.9+**
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imblearn` (for SMOTE)
  - `matplotlib` / `seaborn` (for visualization)

---

📊 Model Evaluation
Since fraud detection is an imbalanced classification problem, accuracy is not a reliable metric.

We evaluate using:
Precision
Recall
F1-Score
Confusion Matrix

📌 Future Improvements
Try XGBoost or LightGBM for better performance.
Add real-time anomaly detection.
Deploy the model using Flask/Django as a web service.

👨‍💻 Author
Pranshu Ranjan
GitHub: @NeilMorales
