# fraud_detection.py
# Credit Card Fraud Detection using Random Forest + SMOTE
# Author: [Your Name]

# -------------------
# Step 1: Import Libraries
# -------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Step 2: Load Dataset
# -------------------
df = pd.read_csv(r"C:\Users\Pranshu\OneDrive\Desktop\Pranshu vit python 23bds0249\ccfraud_detection\creditcard.csv")  # Dataset is one level up from this folder
print("Dataset Shape:", df.shape)
print(df['Class'].value_counts())

# -------------------
# Step 3: Data Preprocessing
# -------------------
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

# -------------------
# Step 4: Train-Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# -------------------
# Step 5: Handle Class Imbalance with SMOTE
# -------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_res))

# -------------------
# Step 6: Train Random Forest Model
# -------------------
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# -------------------
# Step 7: Model Evaluation
# -------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("AUC Score:", roc_auc_score(y_test, y_pred_proba))

# -------------------
# Step 8: Precision-Recall Curve
# -------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, marker='.', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()

# -------------------
# Step 9: Feature Importance (Optional)
# -------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices[:10]], y=X.columns[indices[:10]])
plt.title('Top 10 Feature Importances')
plt.show()
