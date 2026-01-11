import mlflow
import random
import time
import subprocess

def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def train_dummy_model(alpha: float, beta: int):
    time.sleep(1)
    accuracy = round(random.uniform(0.7, 0.9), 4)
    loss = round(random.uniform(0.1, 0.3), 4)
    return accuracy, loss


if __name__ == "__main__":
    alpha = 0.01
    beta = 10

    git_commit = get_git_commit()

    mlflow.set_experiment("churn_dummy_experiments")

    with mlflow.start_run():
        # Reproducibility metadata
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("beta", beta)
        mlflow.log_param("git_commit", git_commit)

        accuracy, loss = train_dummy_model(alpha, beta)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        print(
            f"Run complete | accuracy={accuracy}, loss={loss}, commit={git_commit}"
        )
