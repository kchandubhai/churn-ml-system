import mlflow
import mlflow.sklearn
import pandas as pd
import subprocess

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/raw/churn.csv")

    # Drop rows without target (supervised learning rule)
    df = df.dropna(subset=["churn"])

    # Separate features & target
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    # Explicit feature groups
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Full pipeline (preprocessing + model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    accuracy = pipeline.score(X_test, y_test)

    # Log to MLflow
    mlflow.set_experiment("churn_pipeline_training")
    git_commit = get_git_commit()

    with mlflow.start_run():
        mlflow.log_param("model_type", "logistic_regression_pipeline")
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_metric("accuracy", accuracy)

        # Log the ENTIRE pipeline
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="ChurnClassifierPipeline"
        )

        print(f"Pipeline training complete | accuracy={accuracy}")
