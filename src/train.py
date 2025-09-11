# 1. Imports - Bring in all the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

# --- Main Script Logic ---
def main():
    # 2. Load Data
    # The UCI Adult dataset is a classic choice.
    # Column names for the dataset (as they are not in the file)
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "income"]
    try:
        # You'll need to download 'adult.data' and place it in your 'data' folder
        df = pd.read_csv('data/adult.data', header=None, names=columns, na_values=' ?', skipinitialspace=True)
    except FileNotFoundError:
        print("Error: 'adult.data' not found. Please download it from the UCI repository and place it in the 'data' folder.")
        return

    # 3. Preprocessing
    # Drop rows with missing values for simplicity
    df.dropna(inplace=True)
    
    # Separate features (X) and target (y)
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 4. Define and Compile the Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 5. Set up DP Optimizer and Compile
    # These are your key DP parameters!
    l2_norm_clip = 1.0
    noise_multiplier = 1.1
    num_microbatches = 1 # Set to 1 for simplicity, can be batch_size for standard DP-SGD
    learning_rate = 0.001

    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)
    
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # 6. Train the Model
    batch_size = 32
    epochs = 3
    model.fit(X_train_processed, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_processed, y_test))

    # 7. Evaluate and Print Metrics
    loss, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # (Note: Calculating exact epsilon requires a dedicated library like 'dp_accounting'
    # which we can add later. For now, we know training was private).

    # 8. Save the Model Artifact
    model.save('baseline_model.h5')
    print("Model saved as baseline_model.h5")


# Make the script executable
if __name__ == '__main__':
    main()
