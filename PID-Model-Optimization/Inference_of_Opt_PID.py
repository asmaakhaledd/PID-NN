import numpy as np
import pandas as pd
import tensorflow as tf
import random

"""Load model"""

# Load the trained model
def load_model(model_path):
    pid_model = tf.keras.models.load_model(model_path, compile=False)
    pid_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return pid_model

"""Preprocess time features (cyclic encoding)"""

# Preprocess time features (Use timestep as timestamp)
def preprocess_time_features(timestep, glucose, weight):
    # Using cyclic encoding for time, but we will use timestep as time (1, 2, 3,...)
    hour = timestep % 24
    minute = 0  # Simulate with minute 0 for simplicity
    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)
    return [glucose - 110, glucose - random.uniform(70, 180), weight, time_sin, time_cos]

"""Prepare PID data for inference"""

# Prepare PID data for inference
def prepare_pid_data(timestep, glucose, weight):
    # Preprocess real-time data for the PID model
    return np.array([preprocess_time_features(timestep, glucose, weight)])

"""Adjust basal insulin dosage based on glucose and weight"""

def adjust_basal_insulin(glucose_level, weight, Kp, Ki, Kd):
    basal_rate_per_kg = 0.5
    TDI = basal_rate_per_kg * weight
    basal_insulin_dosage = 0.5 * TDI
    hourly_basal_rate = basal_insulin_dosage / 24

    target_glucose = 110

    if glucose_level > target_glucose:
        adjustment_factor = Kp * (glucose_level - target_glucose)  # Use Kp for proportional adjustment
        adjusted_basal_rate = hourly_basal_rate + adjustment_factor
    elif glucose_level < target_glucose:
        adjustment_factor = Ki * (target_glucose - glucose_level)  # Use Ki for integral adjustment
        adjusted_basal_rate = hourly_basal_rate - adjustment_factor
    else:
        adjusted_basal_rate = hourly_basal_rate

    # Apply limits to the basal rate
    min_basal_rate = 0.3
    max_basal_rate = 1.5
    adjusted_basal_rate = max(min_basal_rate, min(adjusted_basal_rate, max_basal_rate))

    return adjusted_basal_rate

"""Inference loop over real-time data"""

def predict_insulin_dosage(pid_model, glucose, weight, timestep):
    # Prepare input data for PID model
    X_pid_test = prepare_pid_data(timestep, glucose, weight)

    # Make prediction using the trained PID model (predicting Kp, Ki, Kd)
    predicted_pid_gains = pid_model.predict(X_pid_test)

    # Extract predicted PID gains (Kp, Ki, Kd)
    Kp, Ki, Kd = predicted_pid_gains[0]

    # Adjust basal insulin based on the predicted PID gains and glucose level
    adjusted_basal_rate = adjust_basal_insulin(glucose, weight, Kp, Ki, Kd)

    # Print the result
    print(f"\nSample {timestep}:")
    print(f"Timestep: {timestep}")
    print(f"Glucose: {glucose:.2f} mg/dL")
    print(f"Adjusted Hourly Basal Insulin: {adjusted_basal_rate:.2f} U per hour")
    print(f"Weight: {weight} kg")
    print("-" * 50)

"""Main"""

if __name__ == "__main__":
    # Path to your saved model
    model_path = '/content/drive/MyDrive/GP PID/opt_pid_tuning_model_2.h5'

    # Load the model
    pid_model = load_model(model_path)

    # Set weight constant (can be updated as needed)
    weight = 70  # Constant weight

    # Initialize timestep
    timestep = 1

    # Loop to simulate real-time data
    while True:
        # Get simulated real-time glucose data (replace with actual CGM data in real use)
        glucose = random.uniform(70, 180)  # Simulated glucose value between 70 and 180 mg/dL

        # Perform prediction and insulin adjustment
        predict_insulin_dosage(pid_model, glucose, weight, timestep)

        # Increment timestep for next reading
        timestep += 1

        # Limit the number of samples to avoid infinite loop for testing
        if timestep > 100:  # Example condition to stop after 100 iterations (remove for infinite loop)
            break
