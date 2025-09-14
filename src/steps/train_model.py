# src/steps/train_model.py

import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from zenml import step
import mlflow
import numpy as np
from typing_extensions import Tuple, Annotated

@step
def train_model(
    X_train_processed: np.ndarray, 
    y_train: np.ndarray
) -> Tuple[
    Annotated[str, "weights_path"], 
    Annotated[str, "run_id"]
]:
    """Trains the model and returns the path to the weights and the MLflow run ID."""
    mlflow.set_experiment("DP_Model_Training")
    with mlflow.start_run() as run:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=1.0, noise_multiplier=1.1, num_microbatches=1, learning_rate=0.001)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_processed, y_train, epochs=3, batch_size=32, verbose=1)
        
        weights_path = "model_weights.h5"
        model.save_weights(weights_path)
        
        # Log the full model to MLflow with the custom object for later deployment
        mlflow.tensorflow.log_model(
            model, 
            "model", 
            custom_objects={"DPOptimizerClass": DPKerasAdamOptimizer}
        )
        print("Model trained and logged to MLflow.")

        return weights_path, run.info.run_id