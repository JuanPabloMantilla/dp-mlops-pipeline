import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from zenml import step
import mlflow
import numpy as np

@step
def train_model(X_train_processed: np.ndarray, y_train: np.ndarray) -> str:
    """Trains a model and returns the path to the saved weights."""
    mlflow.set_experiment("DP_Model_Training")
    with mlflow.start_run():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        l2_norm_clip = 1.0
        noise_multiplier = 1.1
        learning_rate = 0.001
        
        mlflow.log_param("l2_norm_clip", l2_norm_clip)
        mlflow.log_param("noise_multiplier", noise_multiplier)
        mlflow.log_param("learning_rate", learning_rate)

        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=1,
            learning_rate=learning_rate)
        
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        batch_size = 32
        epochs = 3
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        model.fit(X_train_processed, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save only the weights to a file
        weights_path = "model_weights.h5"
        model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")

        # Return the path to the weights
        return weights_path