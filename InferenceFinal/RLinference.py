# --- START OF FILE inference_controller.py ---

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from collections import OrderedDict, deque # Import deque

class RL_PID_Controller:
    """
    Standalone RL-PID controller for artificial pancreas systems.
    Calculates insulin rate based on PID logic with gains tuned by an RL agent.
    Returns insulin rate in Units per Hour (U/h).
    Internal calculations (IOB) use Units per Minute (U/min).
    """

    def __init__(self, agent_path: str = None, target_glucose: float = 100):
        """
        Initializes the RL-PID controller.

        Args:
            agent_path (str, optional): Path to the saved RL agent model checkpoint (.pt file).
                                        Defaults to None (uses default PID gains).
            target_glucose (float, optional): The target blood glucose level in mg/dL.
                                              Defaults to 100.
        """
        self.target_glucose = target_glucose
        self.current_time = 0 # Tracks time steps since last reset

        # Initialize PID components (managed within self.pid object)
        self.pid = self._init_pid()

        # Initialize RL agent structure
        self.agent = self._init_agent()
        if agent_path:
            self.load_agent(agent_path)
        else:
            print("Warning: No agent path provided. Using default PID gains.")


    def _init_pid(self):
        """Initialize PID controller state and parameters."""
        pid = type('', (), {})() # Creates an empty object to hold PID state
        # Default Gains (will be overridden by RL agent if loaded)
        pid.Kp = 0.02
        pid.Ki = 0.0005
        pid.Kd = 0.0005
        # PID State Variables
        pid.prev_error = 0.0
        pid.integral = 0.0
        # Insulin Dose History (stores U/min values for IOB calculation)
        pid.insulin_doses = deque(maxlen=420) # Max duration considered for IOB
        # Insulin Action Model Parameters
        pid.insulin_peak = 90       # Time to peak effect (minutes)
        pid.insulin_duration = 420  # Duration of action (minutes)
        # Controller Output Constraints (in U/min)
        pid.min_insulin = 0.001 # Minimum deliverable dose rate (U/min)
        pid.max_insulin = 0.3   # Maximum deliverable dose rate (U/min)
        # Dosing Interval Logic
        pid.dosing_interval = 15    # Deliver non-zero dose at most every 15 minutes
        pid.last_dose_time = -pid.dosing_interval # Allow immediate first dose

        # --- Insulin Action Curve Setup ---
        # Create the non-normalized, raw action curve shape
        raw_curve = self._create_raw_action_curve(pid.insulin_peak, pid.insulin_duration)
        # Calculate the per-minute action fraction (normalized curve)
        curve_sum = raw_curve.sum()
        # action_curve_per_minute not strictly needed if using cumulative directly for IOB
        # pid.action_curve_per_minute = raw_curve / curve_sum if curve_sum > 0 else np.zeros_like(raw_curve)

        # Pre-calculate the CUMULATIVE action fraction curve
        # This represents the fraction of total effect that HAS occurred by time t
        # We normalize the raw curve before cumsum to get fraction
        normalized_raw_curve = raw_curve / curve_sum if curve_sum > 0 else np.zeros_like(raw_curve)
        pid.cumulative_action_curve = np.cumsum(normalized_raw_curve)
        # Ensure the curve reaches 1.0 at the end (handle potential float precision issues)
        if len(pid.cumulative_action_curve) > 0:
            pid.cumulative_action_curve /= pid.cumulative_action_curve[-1]


        return pid

    def _init_agent(self):
        """Initialize RL agent network structure and gain bounds."""
        agent = type('', (), {})() # Creates an empty object for the agent
        # Define the policy network architecture
        agent.policy_net = nn.Sequential(
            nn.Linear(11, 64),      # Input state dimension = 11
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),       # Output action dimension = 3 (Kp, Ki, Kd adjustments)
            nn.Sigmoid()            # Output between 0 and 1
        )
        # Default Gain Bounds (can be overridden by loaded checkpoint)
        # Use the less aggressive bounds discussed
        agent.gain_min = torch.tensor([0.01, 0.0001, 0.0001], dtype=torch.float32)
        agent.gain_max = torch.tensor([0.04, 0.0015, 0.0015], dtype=torch.float32) # Lowered max
        return agent

    def load_agent(self, path: str):
        """Load trained RL agent weights and parameters from a checkpoint file."""
        try:
            # Load checkpoint onto CPU to avoid device mismatches
            checkpoint = torch.load(path, map_location=torch.device('cpu'))

            # Load model state dictionary
            state_dict = checkpoint['model_state_dict']
            # Handle potential OrderedDict if saved directly from model.state_dict()
            if isinstance(state_dict, OrderedDict):
                 self.agent.policy_net.load_state_dict(state_dict)
            else:
                 # Attempt loading assuming it's a standard state dict object
                 self.agent.policy_net.load_state_dict(state_dict)
            self.agent.policy_net.eval() # Set model to evaluation mode

            print(f"Successfully loaded agent model state from {path}")

            # Load optimizer state if needed for further training (usually not needed for inference)
            # if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            #      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load exploration parameter epsilon if needed
            # if 'epsilon' in checkpoint:
            #      self.epsilon = checkpoint.get('epsilon', 0.01) # Default to low epsilon for inference

            # Load gain bounds if they were saved in the checkpoint
            if 'gain_bounds' in checkpoint:
                 loaded_min, loaded_max = checkpoint['gain_bounds']
                 # Ensure they are tensors
                 self.agent.gain_min = torch.as_tensor(loaded_min, dtype=torch.float32)
                 self.agent.gain_max = torch.as_tensor(loaded_max, dtype=torch.float32)
                 print(f"Loaded gain bounds: Min={self.agent.gain_min.numpy()}, Max={self.agent.gain_max.numpy()}")


        except FileNotFoundError:
            print(f"Error: Agent checkpoint file not found at {path}. Using default PID gains.")
        except Exception as e:
            print(f"Error loading agent from {path}: {e}. Using default PID gains.")
            # Consider raising the error depending on application needs


    def _create_raw_action_curve(self, peak, duration):
        """Generates the non-normalized insulin action curve shape."""
        t = np.arange(duration)
        # Clamp peak to avoid division by zero or invalid math if peak is near 0
        peak = max(peak, 1.0)
        duration = max(duration, 1.0) # Ensure duration is positive

        # Biphasic curve formula
        phase1 = 0.4 * (1.0 - np.exp(-0.2 * t / peak))
        phase2 = 0.6 * np.exp(-0.2 * (t / duration)**2)
        curve = phase1 + phase2
        return curve

    def _calculate_iob(self):
        """Calculate Insulin On Board (Units) based on REMAINING activity."""
        if not self.pid.insulin_doses:
            return 0.0

        active_insulin_units = 0.0
        num_doses = len(self.pid.insulin_doses)
        duration = self.pid.insulin_duration
        cumulative_curve = self.pid.cumulative_action_curve # Use pre-calculated cumulative curve

        # --- Debug print for cumulative curve (optional, uncomment to check shape) ---
        # if self.current_time < 2: print(f"DEBUG Cumulative Curve Start: {cumulative_curve[:5]}")
        # if self.current_time < 2: print(f"DEBUG Cumulative Curve End: {cumulative_curve[-5:]}")

        # --- Debug print for calculation start (optional) ---
        # print(f"--- DEBUG IOB @ t={self.current_time}: Calculating with {num_doses} doses (Remaining Activity Method) ---")

        for i in range(num_doses):
            dose_umin = self.pid.insulin_doses[i] # Doses are stored in U/min
            if dose_umin > 0:
                time_since_dose = num_doses - 1 - i # How many minutes ago this dose was given
                if 0 <= time_since_dose < duration:
                    # Fraction of total effect that HAS happened by time_since_dose
                    fraction_occurred = cumulative_curve[time_since_dose]
                    # Fraction of total effect REMAINING
                    fraction_remaining = 1.0 - fraction_occurred
                    # Ensure non-negative due to potential float precision near end of curve
                    fraction_remaining = max(0.0, fraction_remaining)

                    # Contribution to active insulin = Original Dose Amount * Fraction Remaining
                    # Note: Dose was rate (U/min), assuming 1-minute steps, dose amount = rate * 1 min
                    contribution = dose_umin * fraction_remaining # Units = (U/min) * (Unitless Fraction) -> Needs adjustment if dose represents bolus volume
                                                                  # Assuming dose_umin IS the amount delivered in that minute step for IOB calc
                    active_insulin_units += contribution

                    # --- Debug print for each contribution (optional) ---
                    # print(f"  t={self.current_time}: Dose[{i}]={dose_umin:.3f} U/min, time_since={time_since_dose}, "
                    #       f"frac_occurred={fraction_occurred:.4f}, frac_remain={fraction_remaining:.4f}, "
                    #       f"contrib={contribution:.4f} U, total_iob={active_insulin_units:.4f} U")

        # Apply a reasonable cap on calculated IOB (Units)
        # Maybe increase cap if high doses are common, e.g., 5.0 or higher
        iob_cap = 5.0
        final_iob_units = min(active_insulin_units, iob_cap)

        # --- Debug print for final result (optional) ---
        # print(f"--- DEBUG IOB @ t={self.current_time}: Final Calculated={active_insulin_units:.4f} U, Capped Return={final_iob_units:.4f} U ---")

        return final_iob_units


    def _create_state_vector(self, glucose, history, iob, meal):
        """Create normalized 11-dimensional state vector for the RL agent."""
        # Ensure history has at least 6 values, pad with current glucose if needed
        hist_list = list(history) # Ensure it's a mutable list
        while len(hist_list) < 6:
            hist_list.insert(0, glucose) # Prepend to keep order

        # Normalize Glucose (mg/dL -> 0-1 range based on 400 max)
        norm_glucose = np.clip(glucose / 400.0, 0, 1)
        # Normalize History
        norm_history = [np.clip(h / 400.0, 0, 1) for h in hist_list[-6:]]

        # Calculate Trends from normalized history
        trend_short = norm_history[-1] - norm_history[-2] if len(norm_history) >= 2 else 0.0
        trend_med = norm_history[-1] - norm_history[-4] if len(norm_history) >= 4 else 0.0

        # Normalize IOB (Units -> 0-1 range based on cap used in _calculate_iob)
        iob_cap_for_norm = 5.0 # Should match the cap in _calculate_iob
        norm_iob = np.clip(iob / iob_cap_for_norm, 0, 1) if iob is not None else 0.0
        # Normalize Meal Announcement (g -> 0-1 range based on expected max)
        meal_max_for_norm = 80.0
        norm_meal = np.clip(meal / meal_max_for_norm, 0, 1)

        # Assemble state vector in the order expected by the agent network
        # Expected order (based on typical use and training state):
        # [glucose, S1_proxy, S2_proxy, meal_impact, trend_short, trend_med, meal_flag, hist-4, hist-3, hist-2, hist-1]
        state = np.array([
            norm_glucose,           # 0: Current normalized glucose
            0.0,                    # 1: Proxy for S1 compartment (Subcutaneous insulin?) - Set to 0 if unavailable directly
            norm_iob,               # 2: Proxy for S2 compartment (Plasma insulin / Effect?) - Use normalized IOB
            norm_meal,              # 3: Normalized meal impact/announcement
            trend_short,            # 4: Short-term trend
            trend_med,              # 5: Medium-term trend
            float(meal > 0),        # 6: Meal flag (binary)
            *norm_history[-4:]      # 7, 8, 9, 10: Last 4 normalized glucose values
        ], dtype=np.float32)

        # Final check for dimensionality (should always be 11 now)
        if len(state) != 11:
             # This indicates a bug in the assembly logic above
             raise ValueError(f"FATAL: State vector length is {len(state)}, expected 11.")

        return state


    def _get_gains(self, state):
        """Get PID gains (Kp, Ki, Kd) from RL agent based on current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): # Ensure no gradients are calculated during inference
            self.agent.policy_net.eval() # Set model to evaluation mode
            # Get raw action output from the network (values between 0 and 1)
            action_raw = self.agent.policy_net(state_tensor).squeeze(0).numpy()

        # --- Debug print for raw agent action (optional) ---
        # print(f"DEBUG GAINS @ t={self.current_time}: Raw Agent Action = {action_raw}")

        # Scale the raw action (0 to 1) to the defined gain range
        gain_min_np = self.agent.gain_min.numpy()
        gain_max_np = self.agent.gain_max.numpy()
        gain_range = gain_max_np - gain_min_np
        scaled_gains = gain_min_np + action_raw * gain_range

        # Final clip for safety, ensuring gains stay within bounds
        final_gains = np.clip(scaled_gains, gain_min_np, gain_max_np)

        # --- Debug print for final gains (optional) ---
        # print(f"DEBUG GAINS @ t={self.current_time}: Final Gains = {final_gains}")

        return final_gains


    def compute_insulin(self,
                        current_glucose: float,
                        glucose_history: list,
                        meal_announcement: float = 0) -> Tuple[float, Dict]:
            """
            Computes the insulin delivery rate in U/h based on current state.

            Args:
                current_glucose (float): Current blood glucose reading (mg/dL).
                glucose_history (list): Recent history of glucose readings (mg/dL), newest last.
                meal_announcement (float, optional): Grams of carbohydrates announced for a future meal.
                                                   Defaults to 0.

            Returns:
                Tuple[float, Dict]:
                    - float: The calculated insulin rate in Units per Hour (U/h).
                    - Dict: A status dictionary containing details of the calculation.
            """
            # --- Pre-calculation Safety Checks ---
            if current_glucose < 50:
                self._update_iob(0.0) # Record zero dose U/min
                status = self._create_status(current_glucose, 0.0, meal_announcement, self.pid.Kp, self.pid.Ki, self.pid.Kd) # Report 0 U/h
                status["alert"] = "CRITICAL_HYPO"
                self.pid.integral = 0 # Reset integral on severe hypo
                self.current_time += 1
                return 0.0, status
            elif current_glucose < 70:
                 self._update_iob(0.0) # Record zero dose U/min
                 status = self._create_status(current_glucose, 0.0, meal_announcement, self.pid.Kp, self.pid.Ki, self.pid.Kd) # Report 0 U/h
                 status["alert"] = "MILD_HYPO"
                 self.pid.integral = 0 # Reset integral on any hypo
                 self.current_time += 1
                 return 0.0, status
            # Note: High glucose check removed here, handled by PID logic + clipping

            # 1. Calculate current IOB (Units) and Create State Vector
            current_iob_units = self._calculate_iob()
            state_vector = self._create_state_vector(current_glucose, glucose_history,
                                                     current_iob_units, meal_announcement)

            # 2. Get RL-adjusted PID gains
            kp, ki, kd = self._get_gains(state_vector)
            self.pid.Kp, self.pid.Ki, self.pid.Kd = kp, ki, kd # Update controller gains

            # 3. Calculate PID error and terms
            error = current_glucose - self.target_glucose # Positive error when high
            # Integral term update with anti-windup
            integral_change = error # Assumes 1-minute steps
            potential_integral = self.pid.integral + integral_change
            integral_min, integral_max = -300, 300 # Define bounds clearly
            self.pid.integral = np.clip(potential_integral, integral_min, integral_max)
            # Derivative term calculation
            derivative = error - self.pid.prev_error # Assumes called every minute

            # 4. Calculate raw PID output (represents U/min rate tendency)
            output_tendency = (kp * error +
                               ki * self.pid.integral + # Use clipped integral
                               kd * derivative)

            # 5. Apply Dosing Interval Logic and Constraints
            final_dose_umin = 0.0 # Default to zero dose (rate in U/min)
            eligible_for_dose = (self.current_time - self.pid.last_dose_time >= self.pid.dosing_interval)

            if eligible_for_dose:
                # Apply IOB constraint (using IOB in Units)
                # Use the less aggressive IOB reduction discussed
                iob_reduction_factor = max(0.0, 1.0 - current_iob_units / 2.0) # Reduced impact point
                potential_dose_umin = output_tendency * iob_reduction_factor

                # Clip potential dose to absolute PID controller min/max (U/min)
                potential_dose_umin = np.clip(potential_dose_umin, 0, self.pid.max_insulin)

                # Apply Glucose-based Safety Constraints (using the stricter limits)
                if current_glucose < 90:
                    potential_dose_umin = 0.0
                elif current_glucose < 110:
                    potential_dose_umin = np.clip(potential_dose_umin, 0, self.pid.max_insulin * 0.05)

                # Final check: only deliver if above minimum threshold (U/min)
                if potential_dose_umin >= self.pid.min_insulin:
                    final_dose_umin = potential_dose_umin
                    self.pid.last_dose_time = self.current_time # Reset timer ONLY if dose delivered
                # else: final_dose_umin remains 0.0

                # Anti-Windup Adjustment: If output wanted insulin but safety/IOB prevented it
                if output_tendency > 0 and final_dose_umin < output_tendency * iob_reduction_factor * 0.9: # If dose significantly reduced
                    # Gently reduce integral if prevented from delivering desired dose
                    self.pid.integral = max(integral_min, self.pid.integral - abs(integral_change) * 0.5)

            # 6. Store Previous Error for next derivative calculation
            self.pid.prev_error = error

            # 7. Update IOB tracking deque with the final U/min dose rate
            self._update_iob(final_dose_umin)

            # 8. Convert final dose rate to U/h for reporting and return
            final_dose_uh = final_dose_umin * 60.0

            # 9. Create status dictionary
            status = self._create_status(current_glucose, final_dose_uh, meal_announcement, kp, ki, kd)
            if current_glucose > 400: status["alert"] = "CRITICAL_HYPER" # Add hyper alert if applicable

            self.current_time += 1
            # Return the final rate in U/h
            return float(final_dose_uh), status


    def _update_iob(self, dose_umin):
        """Track insulin dose rates (U/min) using the deque for IOB calculation."""
        self.pid.insulin_doses.append(float(dose_umin)) # Add current U/min dose rate


    def _create_status(self, glucose, insulin_uh, meal, kp, ki, kd):
         """Helper to create the status dictionary for output."""
         # Calculate IOB again here to ensure it reflects the latest dose added in _update_iob
         current_iob = self._calculate_iob()
         status = {
            "time": self.current_time,
            "glucose": float(glucose),
            "target_glucose": float(self.target_glucose),
            "insulin_delivered_uh": float(insulin_uh), # Rate in U/h
            "iob_calculated_units": float(current_iob), # IOB in Units
            "pid_integral": float(self.pid.integral),
            "pid_prev_error": float(self.pid.prev_error),
            "meal_active": float(meal) > 0,
            "gains": {"Kp": float(kp), "Ki": float(ki), "Kd": float(kd)},
            "alert": None # Default, can be overridden
         }
         return status

    def reset(self):
        """Reset controller state for a new simulation or period."""
        # Re-initialize PID state variables and timers
        self.pid = self._init_pid()
        # Reset internal time counter
        self.current_time = 0
        print("RL_PID_Controller state reset.")


# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    # Ensure the agent model file exists or handle the error
    # Use the standard agent checkpoint file saved during training
    agent_file = "rl_pid_controller.pt" # Make sure this matches the saved file name
    print(f"Attempting to load agent from: {agent_file}")
    try:
        # Initialize controller, attempt to load the agent
        controller = RL_PID_Controller(agent_path=agent_file, target_glucose=100)
    except Exception as e: # Catch broader exceptions during init/loading
        print(f"\n---!!! An error occurred during controller initialization: {e} !!!---")
        print("--- Running test with default PID gains only. ---")
        controller = RL_PID_Controller(agent_path=None, target_glucose=100)

    # --- Simulation Setup ---
    print("\n--- Controller Test Simulation ---")
    test_duration_minutes = 720 # Simulate for 12 hours
    initial_glucose = 150.0
    test_glucose = initial_glucose
    # Initialize history correctly (oldest first, newest last)
    test_history = [initial_glucose] * 6 # Start with stable history

    # Simple Meal Schedule for Test (time_minute, carbs_grams)
    test_meal_schedule = [(120, 30), (360, 35), (600, 25)]
    meal_idx = 0

    # Data Logging for Test Plotting (Optional)
    log_time = []
    log_glucose = []
    log_insulin_uh = []
    log_iob = []

    # --- Simulation Loop ---
    for t in range(test_duration_minutes):
        # Simple Glucose Dynamics Simulation (replace with actual patient model if available)
        if t > 0:
            # Get last delivered dose rate (U/min) from internal deque for effect calculation
            last_delivered_dose_umin = controller.pid.insulin_doses[-1] if controller.pid.insulin_doses else 0.0
            # Basal glucose drop effect
            glucose_change = -0.5 # Slower basal drop for realism
            # Insulin effect (scaled based on U/min rate)
            glucose_change -= last_delivered_dose_umin * 15 # Adjusted insulin sensitivity factor
            test_glucose += glucose_change
            test_glucose = max(40.0, test_glucose) # Glucose floor

        # Meal Effect Simulation
        if meal_idx < len(test_meal_schedule) and t == test_meal_schedule[meal_idx][0]:
            carbs = test_meal_schedule[meal_idx][1]
            print(f"\n*** Meal Announced: {carbs}g at t={t} ***")
            # Simple immediate glucose increase from meal
            test_glucose += carbs * 0.5 # Adjust factor as needed
            meal_announcement = float(carbs)
            meal_idx += 1
        else:
            meal_announcement = 0.0

        # Update glucose history (newest value at end)
        test_history.append(test_glucose)
        if len(test_history) > 20: # Keep history reasonably sized
            test_history = test_history[-20:]

        # --- Call the Controller ---
        # compute_insulin returns rate in U/h
        try:
            insulin_uh, status = controller.compute_insulin(
                current_glucose=test_glucose,
                glucose_history=list(test_history), # Pass a copy
                meal_announcement=meal_announcement
            )
        except Exception as e:
            print(f"!!! Error during compute_insulin at t={t}: {e} !!!")
            # Optionally break or continue with default values
            insulin_uh = 0.0
            status = {"error": str(e)} # Basic error status
            # break

        # --- Log Data ---
        log_time.append(t)
        log_glucose.append(status.get("glucose", test_glucose)) # Use glucose from status if available
        log_insulin_uh.append(status.get("insulin_delivered_uh", insulin_uh))
        log_iob.append(status.get("iob_calculated_units", 0.0))

        # --- Print Status ---
        # Print status less frequently for longer simulations
        if t % 30 == 0 or meal_announcement > 0 or status.get("alert") is not None:
             print(f"t={status.get('time', t):<3}: Glc={status.get('glucose', test_glucose):<6.1f} | "
                   f"Insulin={status.get('insulin_delivered_uh', insulin_uh):.3f} U/h | "
                   f"IOB={status.get('iob_calculated_units', 0.0):.3f} U | "
                   f"Kp={status.get('gains', {}).get('Kp', -1):.4f} | "
                   f"Ki={status.get('gains', {}).get('Ki', -1):.5f} | "
                   f"Kd={status.get('gains', {}).get('Kd', -1):.5f} | "
                   f"Alert={status.get('alert', None)}")


    print("\n--- End Test Simulation ---")

    # --- Basic Plotting for Test Results ---
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Glucose Plot
        axs[0].plot(log_time, log_glucose, label='Glucose (mg/dL)')
        axs[0].axhline(70, color='r', linestyle='--', linewidth=0.8, label='Hypo Threshold')
        axs[0].axhline(180, color='orange', linestyle='--', linewidth=0.8, label='Hyper Threshold')
        axs[0].set_ylabel('Glucose (mg/dL)')
        axs[0].set_title('Inference Test Simulation Results')
        axs[0].legend()
        axs[0].grid(True, linestyle=':')

        # Insulin Plot
        axs[1].plot(log_time, log_insulin_uh, label='Insulin Rate (U/h)', color='blue')
        axs[1].set_ylabel('Insulin (U/h)')
        axs[1].legend()
        axs[1].grid(True, linestyle=':')

        # IOB Plot
        axs[2].plot(log_time, log_iob, label='Calculated IOB (Units)', color='green')
        axs[2].set_ylabel('IOB (Units)')
        axs[2].set_xlabel('Time (minutes)')
        axs[2].legend()
        axs[2].grid(True, linestyle=':')

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping test plot generation.")
    except Exception as plot_err:
        print(f"\nError during plotting: {plot_err}")

# --- END OF FILE inference_controller.py ---