import mlflow
import random
import time

def train_dummy_model(alpha: float, beta: int):
    """
    Fake training function to demonstrate MLflow tracking.
    """
    time.sleep(1)  # simulate work
    accuracy = round(random.uniform(0.7, 0.9), 4)
    loss = round(random.uniform(0.1, 0.3), 4)
    return accuracy, loss


if __name__ == "__main__":
    alpha = 0.01
    beta = 10

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("beta", beta)

        # Run fake training
        accuracy, loss = train_dummy_model(alpha, beta)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        print(f"Run completed | accuracy={accuracy}, loss={loss}")
