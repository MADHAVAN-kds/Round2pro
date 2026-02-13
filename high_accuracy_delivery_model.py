# ======================================================
# HIGH ACCURACY DELIVERY ML MODEL
# NO DATA DELETION | FULL DATASET USED
# TARGET ACCURACY ≈ 0.9666
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# 1. LOAD DATASET (NO ROWS REMOVED)
# ======================================================

df = pd.read_csv("delivery_data.csv")

print("Dataset Shape:", df.shape)
print("No rows removed.")

# ======================================================
# 2. HANDLE MISSING VALUES (WITHOUT DELETING ANYTHING)
# ======================================================

# Fill numeric columns with median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled without deleting data.")

# ======================================================
# 3. CREATE TARGET VARIABLE (NO COLUMN REMOVED)
# ======================================================

df["Delivery_Performance_Risk"] = (df["ontime_rate"] < 0.90).astype(int)

print("Target Created.")

# ======================================================
# 4. ENCODE CATEGORICAL DATA (KEEP ALL COLUMNS)
# ======================================================

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ======================================================
# 5. DEFINE FEATURES (USE ALL ORIGINAL FEATURES)
# ======================================================

X = df.drop("Delivery_Performance_Risk", axis=1)
y = df["Delivery_Performance_Risk"]

# ======================================================
# 6. TRAIN TEST SPLIT (DATA NOT DELETED)
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# 7. RANDOM FOREST (TUNED FOR HIGH ACCURACY)
# ======================================================

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [300, 400],
    "max_depth": [20, None],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ======================================================
# 8. PREDICTION
# ======================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=======================================")
print("FINAL MODEL ACCURACY:", round(accuracy, 4))
print("=======================================")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ======================================================
# 9. CONFUSION MATRIX
# ======================================================

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ======================================================
# 10. FEATURE IMPORTANCE
# ======================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# ======================================================
# 11. FUTURE PREDICTION EXAMPLE
# ======================================================

future_sample = X_test.iloc[0:1]
future_prediction = best_model.predict(future_sample)

if future_prediction[0] == 1:
    print("\n⚠ Future Delivery Performance Risk Detected")
else:
    print("\n✓ Future Delivery Performance Stable")
