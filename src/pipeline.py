# src/pipeline.py

from zenml import pipeline
from .steps.ingest_data import ingest_data
from .steps.process_data import process_data
from .steps.train_model import train_model
from .steps.evaluate_model import evaluate_model
from .steps.deployment_trigger import deployment_trigger
from .steps.deploy_model import deploy_model

@pipeline
def dp_training_deployment_pipeline():
    """Defines the full training and deployment pipeline."""
    raw_data = ingest_data()
    
    # process_data now returns 4 simple artifacts
    X_train_proc, X_test_proc, y_train, y_test = process_data(df=raw_data)
    
    # train_model returns the weights path and the MLflow run ID
    weights_path, run_id = train_model(
        X_train_processed=X_train_proc, y_train=y_train
    )
    
    # evaluate_model takes the weights and run_id to evaluate and log
    accuracy = evaluate_model(
        weights_path=weights_path,
        run_id=run_id,
        X_test_processed=X_test_proc,
        y_test=y_test,
    )
    
    # deployment_trigger makes a decision based on accuracy
    deployment_decision = deployment_trigger(accuracy=accuracy)
    
    # deploy_model takes the decision and the run_id to serve the correct model
    deploy_model(
        deploy_decision=deployment_decision,
        run_id=run_id,
    )