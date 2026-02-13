import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")


def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    return df




def feature_engineering(df):

   
    df.fillna(df.median(numeric_only=True), inplace=True)

  
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

   
    df = pd.get_dummies(df, drop_first=True)

    if "ontime_rate" in df.columns:
        df["Delivery_Risk"] = (df["ontime_rate"] < 0.90).astype(int)

  
    if "delivery_time" in df.columns and "expected_time" in df.columns:
        df["delivery_delay"] = df["delivery_time"] - df["expected_time"]

   
    if "distance" in df.columns and "delivery_time" in df.columns:
        df["delivery_speed"] = df["distance"] / (df["delivery_time"] + 1)

    print("Feature Engineering Completed")

    return df




def split_data(df):

    X = df.drop("Delivery_Risk", axis=1)
    y = df["Delivery_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train-Test Split Completed")

    return X_train, X_test, y_train, y_test, X, y




def build_model():

    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42
        ))
    ])

    print("Model Pipeline Created")

    return model_pipeline



def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)
    print("Model Training Completed")

    return model



def evaluate_model(model, X_test, y_test, X, y):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n==============================")
    print("TEST ACCURACY:", round(accuracy, 4))
    print("ROC-AUC SCORE:", round(roc_auc, 4))
    print("==============================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(cm)

    # Cross Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\nCross Validation Accuracy:", round(cv_scores.mean(), 4))


def compare_models(X_train, X_test, y_train, y_test):

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    log_acc = accuracy_score(y_test, log_model.predict(X_test))

    print("\nLogistic Regression Accuracy:", round(log_acc, 4))




if __name__ == "__main__":

    DATA_PATH = "delivery_data.csv"   # Change path if needed

    df = load_data(DATA_PATH)

    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, X, y = split_data(df)

    model = build_model()

    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test, X, y)

    compare_models(X_train, X_test, y_train, y_test)

    print("\nModel Execution Completed Successfully.")
