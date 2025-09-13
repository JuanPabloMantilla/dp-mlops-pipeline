from src.pipeline import dp_training_pipeline
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

if __name__ == "__main__":
    print("Initiating pipeline run...")

    # Define the custom object dictionary that Keras needs to load our model
    custom_objects = {"DPOptimizerClass": DPKerasAdamOptimizer}
    
    # Use the custom_object_scope to make Keras aware of our optimizer globally
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Run the pipeline within this scope, with cache disabled for one final clean run
        dp_training_pipeline.with_options(enable_cache=False)()
    
    print("Pipeline run finished.")