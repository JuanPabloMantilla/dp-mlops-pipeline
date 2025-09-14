# src/model_wrapper.py
import mlflow
import tensorflow as tf
import pandas as pd
import cloudpickle

class DPKerasWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """This method is called once when the model is loaded for serving."""
        # Load the preprocessor from the artifacts
        with open(context.artifacts["preprocessor"], 'rb') as f:
            self.preprocessor = cloudpickle.load(f)
        
        # Re-create the model's architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(108,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Load the trained weights from the artifacts
        self.model.load_weights(context.artifacts["model_weights"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """This method is called for every prediction request."""
        processed_input = self.preprocessor.transform(model_input)
        predictions = self.model.predict(processed_input)
        # Return predictions formatted as a DataFrame
        return pd.DataFrame((predictions > 0.5).astype(int), columns=["prediction"])