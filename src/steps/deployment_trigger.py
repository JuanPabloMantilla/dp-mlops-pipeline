# src/steps/deployment_trigger.py

from zenml import step

@step
def deployment_trigger(accuracy: float) -> bool:
    """Decides whether to deploy the model based on its accuracy."""
    deploy_decision = accuracy > 0.75
    print(f"Deployment decision is: {deploy_decision}")
    return deploy_decision