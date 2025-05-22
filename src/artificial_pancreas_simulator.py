"""Imports & Dependencies"""

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math
from dataclasses import dataclass

"""InsulinSimulator Class"""

class InsulinSimulator:
    def __init__(self, model_path, test_case_path, totalSimulationTimeInNs):
        """
        Initializes the insulin simulation with the model and test case data.

        Args:
            model_path (str): Path to the saved model for PID prediction.
            test_case_path (str): Path to the test case data file.
        """
        # Load the trained neural network model for PID prediction
        self.model = self.load_model(model_path)

        # Parse the test case file for weight, simulation time, and meal schedules
        self.weight, self.meals = self.parse_test_case(test_case_path)

        # Convert simulation time from nanoseconds to seconds
        self.sim_time= int(totalSimulationTimeInNs/10**9)

        # Store the meal schedule in a dictionary for fast lookups based on time
        self.meal_schedule = {time: carbs for time, carbs in self.meals}

        # Initialize the simulation state (basal rates, bolus doses, etc.)
        self.state = self.initialize_state()



    def load_model(self, path):
        """
        Loads the trained model from the specified path.

        Args:
            path (str): Path to the saved model.

        Returns:
            tf.keras.Model: The loaded Keras model.
        """
        # Load the saved Keras model and compile it with the Adam optimizer
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model

    def parse_test_case(self, file_path):
        """
        Parse the test case file to extract weight, simulation time, and meal schedule.

        Args:
            file_path (str): Path to the test case file.

        Returns:
            tuple: weight (float), simulation time (int), meals (list of tuples)
        """
        # Open the test case file and read each line
        with open(file_path, 'r') as file:
            lines = file.readlines()

        weight = None
        sim_time = None
        meals = []

        # Parse each line in the test case file
        for line in lines:
            if "Body Weight" in line:
                # Extract body weight from the file
                weight = float(line.strip().split(':')[1].split()[0])
            elif "Meal" in line and "Time" in line:
                # Extract meal time and carbs from the file
                parts = line.strip().split(',')
                time = int(parts[0].split(':')[1].strip().split()[0])
                carbs = float(parts[1].split(':')[1].strip().split()[0])
                meals.append((time, carbs))

        # Return weight, and meals as a tuple
        return weight, meals

    def initialize_state(self):
        """
        Initializes the state variables for the simulation, including TDI, basal limit, and trackers.

        Returns:
            dict: State variables for the simulation.
        """
        # Calculate Total Daily Insulin (TDI) and basal insulin limit (50% of TDI)
        TDI = 0.45 * self.weight
        daily_basal_limit = 0.5 * TDI  # 50% of TDI as basal insulin per day

        # Return the initial state as a dictionary with key variables for the simulation
        return {
            "TDI": TDI,
            "daily_basal_limit": daily_basal_limit,
            "previous_glucose": 100,
            "cumulative_error": 0,
            "next_basal_timestep": 1,  # Initialize next basal time step
            "basal_tracker": {},  # key: day index, value: total basal delivered
            "basal_rates": [],
            "bolus_insulin": [],
            "carb_intake": [],
            "infusion_values": [],
            "bolusHoldUntil" : -1
        }


    def preprocess_time_features(self, timestep, glucose):
        """
        Preprocesses time features using cyclic encoding for the current hour of the day.

        Args:
            timestep (int): The current timestep (minute).
            glucose (float): The current glucose level.

        Returns:
            list: A list of preprocessed time features (glucose, weight, sin/cos encoding).
        """
        # Use cyclic encoding to represent hours of the day (sin and cos transformations for cyclic features)
        hour = timestep % 24
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)

        # Return the processed time features along with glucose and weight
        return [glucose - 120, glucose - self.state["previous_glucose"], self.weight, time_sin, time_cos]

    def prepare_pid_data(self, timestep, glucose):
        """
        Prepares the input data for the PID model by processing time features.

        Args:
            timestep (int): The current timestep (minute).
            glucose (float): The current glucose level.

        Returns:
            np.array: Processed input data for the PID model.
        """
        # Preprocess the features and prepare them for PID model inference
        return np.array([self.preprocess_time_features(timestep, glucose)])

    def predict_insulin_dosage(self, glucose, timestep):
        """
        Predicts the insulin dosage (Kp, Ki, Kd) using the trained PID model.

        Args:
            glucose (float): The current glucose level.
            timestep (int): The current timestep (minute).

        Returns:
            float: The adjusted basal insulin dosage.
        """
        # Prepare input data for PID prediction
        X_test = self.prepare_pid_data(timestep, glucose)

        # Predict PID gains (Kp, Ki, Kd) using the trained model
        Kp, Ki, Kd = self.model.predict(X_test)[0]

        # Adjust basal insulin dosage based on PID output
        return self.adjust_basal_insulin(glucose, Kp, Ki, Kd, timestep)

    def adjust_basal_insulin(self, glucose, Kp, Ki, Kd, timestep):
        """
        Adjusts the basal insulin dosage using a PID controller.

        Args:
            glucose (float): The current glucose level.
            Kp (float): Proportional gain from the model.
            Ki (float): Integral gain from the model.
            Kd (float): Derivative gain from the model.
            timestep (int): The current timestep (minute).

        Returns:
            float: The adjusted basal insulin rate.
        """
        target_glucose = 120  # Target glucose level
        error = glucose - target_glucose  # Calculate the error from the target glucose level
        self.state["cumulative_error"] = max(min(self.state["cumulative_error"] + error, 10), -10)  # Update cumulative error
        derivative = glucose - self.state["previous_glucose"]  # Rate of change of glucose level

        # Calculate basal dose based on weight and TDI
        basal_rate_per_kg = 0.5  # Basal insulin rate per kg of body weight
        basal_dose = basal_rate_per_kg * self.state["TDI"] / 24  # Basal insulin dose per hour
        adjusted_rate = basal_dose + (Kp * error) + (Ki * self.state["cumulative_error"]) + (Kd * derivative)

        # Apply limits to the adjusted basal rate (minimum of 0.3 and maximum of 1.5 U per hour)
        adjusted_rate = max(0.1, min(adjusted_rate, 1.5))

        # Store previous glucose for next timestep
        self.state["previous_glucose"] = glucose

        # Print the adjusted basal insulin for this timestep
        print(f"\nSample {timestep}:")
        print(f"Timestep: {timestep}")
        print(f"Glucose: {glucose:.2f} mg/dL")
        print(f"Adjusted Hourly Basal Insulin: {adjusted_rate:.2f} U per hour")
        print(f"Weight: {self.weight} kg")
        print("-" * 50)

        # Return the adjusted basal insulin rate
        return adjusted_rate

    def calculate_bolus(self, glucose, meal_carbs, timestep, target_glucose=120):
        """
        Calculates the bolus insulin based on meal carbs and glucose level.

        Args:
            glucose (float): The current glucose level.
            meal_carbs (float): The amount of carbs in the meal.
            timestep (int): The current timestep (minute).
            target_glucose (float): The target glucose level.

        Returns:
            float: The total bolus insulin required for the meal.
        """
        TDI = self.state["TDI"]
        correction_factor = 1800 / TDI  # Correction factor based on TDI
        correction_dose = max(0, (glucose - target_glucose) / correction_factor)  # Correction dose if glucose is high
        carb_ratio = 500 / TDI  # Carb-to-insulin ratio
        meal_dose = (meal_carbs / carb_ratio) * 60  # Calculate the bolus insulin for the meal
        total_bolus = meal_dose + correction_dose

        # Print bolus calculation details
        print(f"Timestep: {timestep}, Glucose: {glucose:.2f} mg/dL, Carbs:{meal_carbs}, Bolus for meal: {meal_dose:.2f} units/hr, Correction dose: {correction_dose:.2f} units, Total bolus: {total_bolus:.2f} units/hr")

        return total_bolus


    def run(self, glucose, currentTimeInNs):
        """Run the insulin simulation over the specified time."""
        timestep = int(currentTimeInNs/10**9)
        meal_carbs = 0

        # If the timestep is valid (within simulation time range)
        if timestep <= self.sim_time:

            if timestep < self.state["bolusHoldUntil"]:
                self.state["infusion_values"].append(0.4)
                self.state["basal_rates"].append(0.4)
                self.state["bolus_insulin"].append(0)
                self.state["carb_intake"].append(0)
                print(f"[Hold] Infusion Values at Timestep {timestep}: {self.state['infusion_values'][-1]}")
                return self.state["infusion_values"][-1]

            # Get the scheduled meal carbs 20 minutes after meal
            meal_carbs = self.meal_schedule.get(timestep - 20, 0)
            give_bolus = meal_carbs > 0
            give_basal = timestep == self.state["next_basal_timestep"] and not give_bolus
            print(f'Current Timestep: {timestep}, Meal Carbs: {meal_carbs}, Give Bolus: {give_bolus}, Give Basal: {give_basal}')
            
            # Handle bolus delivery if applicable
            if give_bolus:
                self.state["bolusHoldUntil"] = timestep + 120
                self.state["next_basal_timestep"] = timestep + 120
                total_bolus = self.calculate_bolus(glucose, meal_carbs, timestep)
                self.state["bolus_insulin"].append(float(total_bolus))
                self.state["carb_intake"].append(meal_carbs)
                self.state["infusion_values"].append(float(total_bolus))

                print(f"Infusion Values at Timestep {timestep}: {self.state['infusion_values'][-1]}")
                return self.state["infusion_values"][-1]

            # Handle basal delivery if applicable
            elif give_basal:
                rate = self.predict_insulin_dosage(glucose, timestep)
                self.state["basal_rates"].append(float(rate))
                self.state["infusion_values"].append(float(rate))
                self.state["next_basal_timestep"] = timestep + 30
                self.state["bolus_insulin"].append(0)
                self.state["carb_intake"].append(0)

                print(f"Infusion Values at Timestep {timestep}: {self.state['infusion_values'][-1]}")
                return self.state["infusion_values"][-1]

            else:
                self.maintain_previous()
                print(f'Maintaining previous basal rate: {self.state["infusion_values"][-1]}')
                return self.state["infusion_values"][-1]


    def maintain_previous(self):
        """Maintain the previous basal if no bolus or basal was delivered."""
        last = self.state["basal_rates"][-1] if self.state["basal_rates"] else 1.0
        self.state["infusion_values"].append(last)
        self.state["basal_rates"].append(last)
        self.state["bolus_insulin"].append(0)
        self.state["carb_intake"].append(0)

    def print_summary(self):
        """Print the summary of insulin delivery per day."""
        print("\nBasal insulin summary per day:")
        for day, total in self.state["basal_tracker"].items():
            print(f"  Day {day + 1}: {total:.2f} U / {self.state['daily_basal_limit']:.2f} U")

"""Main"""

if __name__ == "__main__":
    # Set the paths for the model and test case data
    model_path = 'models/nn_pid_tuning_model.h5'
    test_case_path = 'data/testcases/TestCases.txt'

    # Set the total simulation time in nanoseconds (20 seconds = 20 * 10^9 ns)
    totalSimulationTimeInNs = 3600000000000

    # Initialize the insulin simulation with the model and test case
    sim = InsulinSimulator(model_path, test_case_path, totalSimulationTimeInNs)

    # Initialize current time in nanoseconds (starting point)
    currentTimeInNs = 1000000000  # Example: 1 second (in nanoseconds)

    # Run the simulation until the total simulation time is reached
    while currentTimeInNs <= totalSimulationTimeInNs:
        glucose = random.uniform(70, 150)  # Generate random glucose values

        # Run the simulation for the current time and glucose value
        sim.run(glucose, currentTimeInNs)

        # Increment the time (simulate time passing, here we increment by 1 second in nanoseconds)
        currentTimeInNs += 1000000000
    sim.print_summary()