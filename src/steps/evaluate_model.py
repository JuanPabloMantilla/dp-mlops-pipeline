# src/steps/evaluate_model.py

import tensorflow as tf
import numpy as np
from zenml import step
import mlflow

@step
def evaluate_model(
    weights_path: str, 
    run_id: str,
    X_test_processed: np.ndarray, 
    y_test: np.ndarray
) -> float:
    """Recreates architecture, loads weights, evaluates, and logs the metric."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_test_processed.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    _, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Use the provided run_id to log the metric to the correct experiment run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", accuracy)

    return accuracy