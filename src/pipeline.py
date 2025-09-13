from zenml import pipeline
from .steps.ingest_data import ingest_data
from .steps.process_data import process_data
from .steps.train_model import train_model
from .steps.evaluate_model import evaluate_model

@pipeline
def dp_training_pipeline():
    """Defines the full training pipeline."""
    raw_data = ingest_data()
    X_train, X_test, y_train, y_test, _ = process_data(df=raw_data)
    weights_path = train_model(X_train_processed=X_train, y_train=y_train)
    evaluate_model(weights_path=weights_path, X_test_processed=X_test, y_test=y_test)