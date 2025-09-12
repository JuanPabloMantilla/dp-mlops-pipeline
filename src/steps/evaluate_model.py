import tensorflow as tf
import numpy as np
from zenml import step
import mlflow

@step
def evaluate_model(weights_path: str, X_test_processed: np.ndarray, y_test: np.ndarray):
    """Recreates the model architecture, loads weights, and evaluates."""
    print("Evaluating model...")
    # Re-create the exact same model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_test_processed.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Load the trained weights from the provided path
    model.load_weights(weights_path)
    
    # Compile the model with a standard optimizer (not needed for training, just for .evaluate())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Evaluate and log to the active MLflow run
    loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    mlflow.log_metric("test_accuracy", accuracy)