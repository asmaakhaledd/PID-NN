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

def adjust_basal_insulin(glucose_level, weight, Kp, Ki, Kd, previous_glucose, cumulative_error):
    # Basal insulin rate calculation based on weight
    basal_rate_per_kg = 0.5
    TDI = 0.55 * weight  # Total daily insulin
    basal_insulin_dosage = basal_rate_per_kg * TDI  # Basal insulin dosage (50% of TDI)
    hourly_basal_rate = basal_insulin_dosage / 24  # Hourly basal rate

    # Target glucose level
    target_glucose = 100

    # Calculate the error (difference between current glucose and target)
    error = glucose_level - target_glucose

    # Proportional (Kp) term: Respond to immediate error
    adjustment_factor_proportional = Kp * error

    # Integral (Ki) term: Cumulative error over time
    cumulative_error += error  # Update cumulative error
    if cumulative_error > 10:
        cumulative_error = 10
    elif cumulative_error < -10:
        cumulative_error = -10
    adjustment_factor_integral = Ki * cumulative_error

    # Derivative (Kd) term: Rate of change in glucose
    glucose_change_rate = glucose_level - previous_glucose  # Difference between current and previous glucose levels
    adjustment_factor_derivative = Kd * glucose_change_rate

    # Total adjusted basal rate including all PID terms
    adjusted_basal_rate = hourly_basal_rate + adjustment_factor_proportional + adjustment_factor_integral + adjustment_factor_derivative

    # Apply limits to the basal rate to prevent it from becoming too low or too high
    min_basal_rate = 0.3
    max_basal_rate = 3
    adjusted_basal_rate = max(min_basal_rate, min(adjusted_basal_rate, max_basal_rate))


    # Return the adjusted basal rate and the updated cumulative error for next use
    return adjusted_basal_rate, cumulative_error, glucose_level

"""Predict insulin dosage"""

def predict_insulin_dosage(pid_model, glucose, weight, timestep, previous_glucose, cumulative_error):
    # Prepare input data for PID model
    X_pid_test = prepare_pid_data(timestep, glucose, weight)

    # Make prediction using the trained PID model (predicting Kp, Ki, Kd)
    predicted_pid_gains = pid_model.predict(X_pid_test)

    # Extract predicted PID gains (Kp, Ki, Kd)
    Kp, Ki, Kd = predicted_pid_gains[0]

    # Adjust basal insulin based on the predicted PID gains and glucose level
    adjusted_basal_rate, cumulative_error, previous_glucose = adjust_basal_insulin(
        glucose, weight, Kp, Ki, Kd, previous_glucose, cumulative_error
    )

    # Print the result
    print(f"\nSample {timestep}:")
    print(f"Timestep: {timestep}")
    print(f"Glucose: {glucose:.2f} mg/dL")
    print(f"Adjusted Hourly Basal Insulin: {adjusted_basal_rate:.2f} U per hour")
    print(f"Weight: {weight} kg")
    print("-" * 50)

    # Return updated values to be used in the next timestep
    return previous_glucose, cumulative_error, glucose, adjusted_basal_rate

"""Main"""

if __name__ == "__main__":
    # Path to your saved model
    model_path = 'PID-Model-Optimization/opt_pid_tuning_model_2.h5'

    # Load the model
    pid_model = load_model(model_path)

    # Initialize variables
    previous_glucose = 100
    cumulative_error = 0

    # Set weight constant (can be updated as needed)
    weight = 70  # Constant weight

    # Initialize timestep
    timestep = 1

    basal_rates = []


    while True:
      glucose = random.uniform(70, 180)

      if timestep == 1 or timestep % 15 == 0:
       previous_glucose, cumulative_error, glucose, adjusted_basal_rate = predict_insulin_dosage(pid_model, glucose, weight, timestep, previous_glucose, cumulative_error)
       basal_rates.append(adjusted_basal_rate)

      timestep+=1

      if timestep % 10 == 0:
        print(f"Processed Timestep {timestep}: Glucose: {glucose:.2f}, Basal Rate: {adjusted_basal_rate:.2f}")
      else:
        basal_rates.append(basal_rates[-1] if basal_rates else 1.0)  # Use last basal rate or default
      if timestep > 100:
            break
