# ============================================
# DELIVERY SLA FAILURE PREDICTION ML PROJECT
# High Accuracy Model (Random Forest + Logistic)
# ============================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ============================================
# 2. LOAD DATASET
# ============================================
df = pd.read_csv("your_dataset.csv")   # change file name

print("Data Loaded Successfully")
print(df.head())

# ============================================
# 3. EDA
# ============================================
print("\nINFO:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# ============================================
# 4. FEATURE ENGINEERING (TARGET COLUMN)
# ============================================
df["SLA_Breached"] = (df["delivery_time"] > df["SLA_time"]).astype(int)

# ============================================
# 5. ENCODE CATEGORICAL DATA (Route & Zone kept)
# ============================================
encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = encoder.fit_transform(df[col])

# ============================================
# 6. FEATURE SELECTION
# ============================================
features = ["distance_km", "zone_id", "route_id", "traffic_level",
            "vehicle_type", "customer_type", "stop_count"]

X = df[features]
y = df["SLA_Breached"]

# ============================================
# 7. TRAIN TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================
# 8. LOGISTIC REGRESSION MODEL
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION ACCURACY:", accuracy_score(y_test, log_pred))

# ============================================
# 9. RANDOM FOREST (HIGH ACCURACY)
# ============================================
rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
rf_pred = best_rf.predict(X_test)

print("\nRANDOM FOREST ACCURACY:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

# ============================================
# 10. CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ============================================
# 11. FEATURE IMPORTANCE (PROFESSIONAL OUTPUT)
# ============================================
importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance Ranking:")
print(importance)

# Plot importance
plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance for SLA Prediction")
plt.show()

# ============================================
# 12. FUTURE DELIVERY PREDICTION
# ============================================
# Example new route prediction
future_delivery = pd.DataFrame({
    "distance_km": [30],
    "zone_id": [2],     # SAME AS YOUR DATASET VALUE
    "route_id": [105],  # SAME AS YOUR DATASET VALUE
    "traffic_level": [1],
    "vehicle_type": [0],
    "customer_type": [1],
    "stop_count": [12]
})

future_pred = best_rf.predict(future_delivery)

if future_pred[0] == 1:
    print("\n🚨 Future Delivery WILL BREACH SLA")
else:
    print("\n✅ Future Delivery WILL MEET SLA")
