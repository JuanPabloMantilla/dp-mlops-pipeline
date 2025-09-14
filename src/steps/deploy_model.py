# src/steps/deploy_model.py

import subprocess
from zenml import step

@step
def deploy_model(deploy_decision: bool, run_id: str):
    """Deploys the model as a local MLflow prediction server."""
    if deploy_decision:
        model_uri = f"runs:/{run_id}/model"
        
        print(f"Deploying model from URI: {model_uri}")

        # Command to serve the MLflow model
        command = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--env-manager", "local"
        ]
        
        # Run the command as a background process
        subprocess.Popen(command)
        print("Model deployment server started in the background.")
    else:
        print("Skipping deployment.")