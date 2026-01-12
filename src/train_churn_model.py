import mlflow
import mlflow.sklearn
import pandas as pd
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

if __name__ == "__main__":
    # Load data (already DVC-tracked)
    df = pd.read_csv("data/raw/churn.csv")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow.set_experiment("churn_model_training")
    git_commit = get_git_commit()

    with mlflow.start_run():
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("git_commit", git_commit)

        mlflow.log_metric("accuracy", accuracy)

        # Log the model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnClassifier"
        )

        print(f"Training complete | accuracy={accuracy}")
