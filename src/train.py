# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    # Load processed data
    data = pd.read_csv("../data/processed/features.csv")
    X = data[['Recency', 'Frequency', 'Monetary']]
    y = data['is_high_risk']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and log with MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_model()