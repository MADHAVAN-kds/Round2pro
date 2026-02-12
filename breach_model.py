# =====================================================
# MACHINE LEARNING CLASSIFICATION PROGRAM
# Target Column: Breach_Flag
# Model: Random Forest
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

# -------------------------------
# 2. Load Dataset
# -------------------------------
# Change the file name if needed
file_path = r"C:\Users\Madhavan\Downloads\Round2pro-main\Round2pro-main\Sum of Breach_Flag by zone_id.csv"


df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# 3. Basic Data Preprocessing
# -------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows where target is missing
df = df.dropna(subset=["Breach_Flag"])

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# 4. Define Features and Target
# -------------------------------
X = df.drop("Breach_Flag", axis=1)
y = df["Breach_Flag"]

# -------------------------------
# 5. Split Dataset (70% Train, 30% Test)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# -------------------------------
# 6. Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# -------------------------------
# 7. Test the Model
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluate the Model
# -------------------------------

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Feature Importance
# -------------------------------
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Plot Feature Importance
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df["Feature"][:10][::-1],
         feature_importance_df["Importance"][:10][::-1])
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances")
plt.show()

# -------------------------------
# END OF PROGRAM
# -------------------------------
