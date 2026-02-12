# =====================================================
# MACHINE LEARNING USING EXCEL FILE (VS CODE VERSION)
# =====================================================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 2. Load Excel File
# -------------------------------

file_path = r"Book1.xlsx"   # If file is in same folder

df = pd.read_excel(file_path)

print("Dataset Loaded Successfully!\n")
print(df.head())

# -------------------------------
# 3. Create Classification Target
# -------------------------------

# Create Breach_Flag:
# 1 = High Risk (above median)
# 0 = Low Risk (below median)

median_value = df["Sum of Breach_Flag"].median()

df["Breach_Flag"] = np.where(
    df["Sum of Breach_Flag"] > median_value,
    1,
    0
)

print("\nTarget Created Successfully!")

# -------------------------------
# 4. Encode zone_id
# -------------------------------

le = LabelEncoder()
df["zone_id_encoded"] = le.fit_transform(df["zone_id"])

# -------------------------------
# 5. Define Features and Target
# -------------------------------

X = df[["zone_id_encoded", "Sum of Breach_Flag"]]
y = df["Breach_Flag"]

# -------------------------------
# 6. Split Data (70% Train / 30% Test)
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nData Split Completed!")

# -------------------------------
# 7. Train Random Forest Model
# -------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# -------------------------------
# 8. Model Testing
# -------------------------------

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Feature Importance
# -------------------------------

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

# -------------------------------
# 10. Future Prediction Example
# -------------------------------

print("\nFuture Prediction Example")

# Example new data (Change values as needed)
new_data = pd.DataFrame({
    "zone_id_encoded": [2],
    "Sum of Breach_Flag": [120]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: HIGH RISK ZONE")
else:
    print("Prediction: LOW RISK ZONE")

# -------------------------------
# END
# -------------------------------
