# --- START OF FILE inference_controller.py ---

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from collections import OrderedDict, deque # Import deque

class RL_PID_Controller:
    """Standalone RL-PID controller for artificial pancreas systems."""

    def __init__(self, agent_path: str = None, target_glucose: float = 100):
        self.target_glucose = target_glucose
        self.current_time = 0 # Tracks time steps since last reset

        # Initialize PID components (managed within self.pid object)
        self.pid = self._init_pid()

        # Initialize RL agent
        self.agent = self._init_agent()
        if agent_path:
            self.load_agent(agent_path)

    def _init_pid(self):
        """Initialize PID controller state"""
        pid = type('', (), {})()
        pid.Kp = 0.02
        pid.Ki = 0.0005
        pid.Kd = 0.0005
        pid.prev_error = 0
        pid.integral = 0
        pid.insulin_doses = deque(maxlen=420) # Use deque for efficient history storage
        pid.insulin_peak = 90
        pid.insulin_duration = 420
        pid.action_curve = self._create_action_curve(pid.insulin_peak, pid.insulin_duration)
        pid.min_insulin = 0.001 # Minimum deliverable dose
        pid.max_insulin = 0.3   # Maximum deliverable dose
        # --- NEW: Dosing interval logic ---
        pid.dosing_interval = 15
        pid.last_dose_time = -pid.dosing_interval # Allow immediate first dose
        # --- END NEW ---
        # Create the per-minute action curve
        raw_curve = self._create_raw_action_curve(pid.insulin_peak, pid.insulin_duration)
        # Normalize it (fraction active per minute)
        curve_sum = raw_curve.sum()
        pid.action_curve_per_minute = raw_curve / curve_sum if curve_sum > 0 else np.zeros_like(raw_curve)
        # --- NEW: Pre-calculate CUMULATIVE action curve ---
        # This represents the fraction of total effect that HAS occurred by time t
        pid.cumulative_action_curve = np.cumsum(pid.action_curve_per_minute)
        # --- END NEW ---

        return pid

    def _init_agent(self):
        """Initialize RL agent network structure"""
        agent = type('', (), {})()
        agent.policy_net = nn.Sequential(
            nn.Linear(11, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        # Ensure tensors are created correctly
        agent.gain_min = torch.tensor([0.01, 0.0001, 0.0001], dtype=torch.float32)
        agent.gain_max = torch.tensor([0.05, 0.002, 0.002], dtype=torch.float32)
        return agent

    def load_agent(self, path: str):
        """Load trained RL agent weights"""
        try:
            # Load checkpoint onto CPU, works regardless of saved device
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            # Handle potential OrderedDict if saved directly from model.state_dict()
            state_dict = checkpoint['model_state_dict']
            if isinstance(state_dict, OrderedDict):
                 self.agent.policy_net.load_state_dict(state_dict)
            else:
                 # Attempt loading assuming it's a standard state dict object
                 self.agent.policy_net.load_state_dict(state_dict)

            print(f"Successfully loaded agent model state from {path}")

            # Optionally load optimizer state and epsilon if needed for further training/evaluation
            if 'optimizer_state_dict' in checkpoint:
                 # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Need optimizer defined
                 pass
            if 'epsilon' in checkpoint:
                 # self.epsilon = checkpoint['epsilon'] # Need epsilon defined
                 pass
            if 'gain_bounds' in checkpoint:
                 self.agent.gain_min, self.agent.gain_max = checkpoint['gain_bounds']


        except Exception as e:
            print(f"Error loading agent from {path}: {e}")
            # Consider raising the error or handling it based on application needs
            # raise e


    def compute_insulin(self,
                        current_glucose: float,
                        glucose_history: list,
                        meal_announcement: float = 0) -> Tuple[float, Dict]:
            # ... (Safety checks for < 70 and > 400 remain similar) ...

            # 1. Get RL-adjusted PID gains & IOB
            current_iob = self._calculate_iob()
            state = self._create_state_vector(current_glucose, glucose_history,
                                            current_iob, meal_announcement)
            gains = self._get_gains(state)
            self.pid.Kp, self.pid.Ki, self.pid.Kd = gains

            # 2. Calculate PID error (Corrected Sign)
            # Positive error when glucose is high
            error = current_glucose - self.target_glucose

            # --- Integral Update & Anti-Windup ---
            integral_change = error # Assuming 1-minute steps
            potential_integral = self.pid.integral + integral_change
            # Clip integral within bounds
            self.pid.integral = np.clip(potential_integral, -300, 300) # Use bounds consistent with training

            # --- Derivative Calculation ---
            # Uses error from the previous step stored in self.pid.prev_error
            derivative = error - self.pid.prev_error # Assumes called every minute
            # Store the current error for the next step's derivative calculation later


            # --- Raw PID Output (Positive means insulin needed) ---
            output = (gains[0] * error +
                    gains[1] * self.pid.integral + # Use clipped integral
                    gains[2] * derivative)

            # --- Dosing Interval Logic ---
            final_dose = 0.0 # Default to zero dose
            eligible_for_dose = (self.current_time - self.pid.last_dose_time >= self.pid.dosing_interval)

            if eligible_for_dose:
                # Stronger IOB reduction factor
                iob_reduction_factor = max(0.0, 1.0 - current_iob / 2.0) # Was / 3.0
                # OR even non-linear:
                # iob_reduction_factor = max(0.0, (1.0 - current_iob / 2.5)**2)
                potential_dose = output * iob_reduction_factor

                # Clip potential dose to valid range (0 to max_insulin)
                potential_dose = np.clip(potential_dose, 0, self.pid.max_insulin)
                # --- Glucose Safety Constraints ---
                if current_glucose < 90: # More cautious threshold
                    potential_dose = 0.0
                elif current_glucose < 110: # More cautious range
                    potential_dose = np.clip(potential_dose, 0, self.pid.max_insulin * 0.05) # Heavy reduction
                # else: potential_dose remains as calculated after IOB reduction & clipping

                # Final check: only deliver if above minimum threshold
                if potential_dose >= self.pid.min_insulin:
                    final_dose = potential_dose
                    self.pid.last_dose_time = self.current_time # Reset timer
                else:
                    final_dose = 0.0

                # --- Anti-Windup Adjustment (similar to training) ---
                if output > 0 and final_dose < output * iob_reduction_factor * 0.9:
                    self.pid.integral = max(-300, self.pid.integral - abs(integral_change) * 0.5)


            # --- Store Previous Error ---
            # Store the error calculated in *this* step for the *next* step's derivative
            self.pid.prev_error = error


            # 4. Update IOB tracking with the actual dose decided
            self._update_iob(final_dose)

            # 5. Create status dictionary
            status = self._create_status(current_glucose, final_dose, meal_announcement, gains)
            # Add alert if hypo occurred before dose calculation
            if current_glucose < 70: status["alert"] = "MILD_HYPO"
            if current_glucose < 50: status["alert"] = "CRITICAL_HYPO" # Already handled return? Check logic flow.

            self.current_time += 1 # Increment internal time step
            return float(final_dose), status
    def _get_gains(self, state):
        """Get PID gains from RL agent"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            self.agent.policy_net.eval()
            action = self.agent.policy_net(state_tensor).squeeze(0).numpy()

        # ADD THIS PRINT to see the raw network output
        print(f"DEBUG GAINS @ t={self.current_time}: Raw Agent Action = {action}")

        gain_range = self.agent.gain_max.numpy() - self.agent.gain_min.numpy()
        gains = self.agent.gain_min.numpy() + action * gain_range

        # Optional: print scaled gains before clipping
        # print(f"DEBUG GAINS @ t={self.current_time}: Scaled Gains = {gains}")

        final_gains = np.clip(gains, self.agent.gain_min.numpy(), self.agent.gain_max.numpy())
        return final_gains
    
        # --- NEW HELPER METHOD ---
    def _create_raw_action_curve(self, peak, duration):
        """Generates the non-normalized insulin action curve shape."""
        t = np.arange(duration)
        # Clamp peak to avoid division by zero or tiny numbers if peak is near 0
        peak = max(peak, 1)
        # Original Biphasic curve calculation
        phase1 = 0.4 * (1 - np.exp(-0.2 * t / peak))
        phase2 = 0.6 * np.exp(-0.2 * (t / duration)**2)
        curve = phase1 + phase2
        return curve
    # --- END NEW HELPER ---

    def _create_state_vector(self, glucose, history, iob, meal):
        """Create normalized state vector for the RL agent."""
        # Ensure history has at least 6 values, pad with current glucose if needed
        while len(history) < 6:
            history.insert(0, glucose) # Prepend to keep order

        norm_glucose = np.clip(glucose / 400, 0, 1) # Normalize and clip
        # Use last 6 values for history normalization
        norm_history = [np.clip(h / 400, 0, 1) for h in history[-6:]]

        # Calculate trends from normalized history
        trend_short = norm_history[-1] - norm_history[-2] if len(norm_history) >= 2 else 0
        trend_med = norm_history[-1] - norm_history[-4] if len(norm_history) >= 4 else 0 # Corrected index

        # Normalize IOB and meal announcement
        norm_iob = np.clip(iob / 3.0, 0, 1) if iob is not None else 0 # Max IOB assumed 3 for normalization
        norm_meal = np.clip(meal / 80, 0, 1)   # Max meal size assumed 80g for normalization

        # --- CORRECTED STATE VECTOR ASSEMBLY ---
        # Match the order expected by the agent (based on hovorka_model.get_state likely structure)
        # state_dim=11: [norm_glucose, S1_proxy, S2_proxy, norm_meal, trend_short, trend_med, meal_flag, hist-4, hist-3, hist-2, hist-1]
        state = np.array([
            norm_glucose,           # 0: Current glucose
            # Using IOB as proxy for S1/S2 state if simulator state unavailable.
            # If agent didn't use S1/S2 much, this might be ok. If it did, this is an approximation.
            # Let's use 0 for S1 proxy and norm_iob for S2 proxy as a convention.
            0.0,                    # 1: Proxy for S1 (e.g., subcutaneous insulin) - Set to 0 if unavailable
            norm_iob,               # 2: Proxy for S2 (e.g., plasma insulin / effect) - Use IOB
            norm_meal,              # 3: Meal impact/announcement
            trend_short,            # 4: Short trend
            trend_med,              # 5: Medium trend
            float(meal > 0),        # 6: Meal flag
            *norm_history[-4:]      # 7, 8, 9, 10: Last 4 normalized glucose values (hist[-4] to hist[-1])
        ], dtype=np.float32)
        # --- END CORRECTION ---


        # Ensure state vector has the correct dimension (11) - This check should now pass
        if len(state) != 11:
             # This should ideally not happen now, but keep as a safeguard
             print(f"FATAL ERROR: State vector length is {len(state)}, expected 11. Check assembly logic.")
             # Fallback to a zero vector or raise error, as truncation is bad
             # return np.zeros(11, dtype=np.float32)
             raise ValueError(f"State vector creation failed, length {len(state)} != 11")

        return state

    def _create_action_curve(self, peak, duration):
        """Insulin action curve"""
        t = np.arange(duration)
        phase1 = 0.4 * (1 - np.exp(-0.2 * t/peak))
        phase2 = 0.6 * np.exp(-0.2 * (t/duration)**2)
        curve = phase1 + phase2
        curve_sum = curve.sum()
        return curve / curve_sum if curve_sum > 0 else np.zeros_like(curve)

    def _update_iob(self, dose):
        """Track insulin doses using the deque"""
        self.pid.insulin_doses.append(float(dose)) # Add current dose to the right end

    def _calculate_iob(self):
        """Calculate Insulin On Board based on REMAINING activity."""
        if not self.pid.insulin_doses:
            return 0.0

        active_insulin = 0.0
        num_doses = len(self.pid.insulin_doses)
        duration = self.pid.insulin_duration
        cumulative_curve = self.pid.cumulative_action_curve # Use pre-calculated cumulative curve

        # Optional: Debug print to check cumulative curve values near start and end
        # if self.current_time < 5: print(f"DEBUG Cumulative Curve Start: {cumulative_curve[:5]}")
        # if self.current_time < 5: print(f"DEBUG Cumulative Curve End: {cumulative_curve[-5:]}")


        # print(f"--- DEBUG IOB @ t={self.current_time}: Calculating with {num_doses} doses (Remaining Activity Method) ---")

        for i in range(num_doses):
            dose = self.pid.insulin_doses[i]
            if dose > 0:
                time_since_dose = num_doses - 1 - i
                if 0 <= time_since_dose < duration:
                    # Fraction of effect that HAS happened by time_since_dose
                    fraction_occurred = cumulative_curve[time_since_dose]
                    # Fraction of effect REMAINING
                    fraction_remaining = 1.0 - fraction_occurred
                    # Ensure fraction remaining is not negative due to potential float precision near end of curve
                    fraction_remaining = max(0.0, fraction_remaining)

                    contribution = dose * fraction_remaining
                    active_insulin += contribution
                    # Optional debug print for this method
                    # print(f"  t={self.current_time}: Dose[{i}]={dose:.3f}, time_since={time_since_dose}, "
                    #       f"frac_occurred={fraction_occurred:.4f}, frac_remain={fraction_remaining:.4f}, "
                    #       f"contrib={contribution:.4f}, total_iob={active_insulin:.4f}")
                # else: # Dose is older than duration, contributes 0
                    # pass

        final_iob = min(active_insulin, 3.0) # Apply cap (Maybe increase cap? 3.0 might be too low if doses are 0.3)
        # Optional debug print
        # print(f"--- DEBUG IOB @ t={self.current_time}: Final Calculated={active_insulin:.4f}, Capped Return={final_iob:.4f} ---")

        return final_iob

    def _create_status(self, glucose, insulin, meal, gains=None):
         """Helper to create the status dictionary."""
         status = {
            "time": self.current_time,
            "glucose": float(glucose),
            "target_glucose": float(self.target_glucose),
            "insulin_delivered": float(insulin),
            "iob_calculated": float(self._calculate_iob()),
            "pid_integral": float(self.pid.integral),
            "pid_prev_error": float(self.pid.prev_error),
            "meal_active": float(meal) > 0,
            "alert": None # Default to None, can be overridden
         }
         if gains is not None:
             status["gains"] = {"Kp": float(gains[0]), "Ki": float(gains[1]), "Kd": float(gains[2])}
         else:
             status["gains"] = {"Kp": float(self.pid.Kp), "Ki": float(self.pid.Ki), "Kd": float(self.pid.Kd)} # Report current gains
         return status

    def reset(self):
        """Reset controller state"""
        self.pid = self._init_pid() # Re-initialize PID state including timers
        self.pid.prev_error = 0     # Explicitly reset prev_error
        self.current_time = 0      # Reset internal time
        print("RL_PID_Controller state reset.")


if __name__ == "__main__":
    # Test with sample data
    # Ensure the agent model file exists or handle the error
    agent_file = "agent_episode_500.pt" # Or the latest saved agent
    try:
        controller = RL_PID_Controller(agent_path=agent_file, target_glucose=100)
    except FileNotFoundError:
        print(f"Agent file {agent_file} not found. Running with default PID gains.")
        controller = RL_PID_Controller(agent_path=None, target_glucose=100) # Init without loading agent

    test_glucose = 150.0
    # History should be recent first for _create_state_vector logic
    test_history = [145.0, 147.0, 149.0, 151.0, 153.0, 155.0] # Oldest to newest

    print("\n--- Controller Test ---")
    for t in range(30): # Test for 30 minutes
        # Simulate glucose change
        if t > 0:
             # Simple glucose drift + effect of last insulin - VERY basic simulation
             last_insulin = controller.pid.insulin_doses[-1] if controller.pid.insulin_doses else 0
             test_glucose -= 1.0 # Baseline drop
             test_glucose -= last_insulin * 10 # Crude insulin effect
             test_glucose = max(60, test_glucose) # Floor glucose

        # Update history (newest value at end)
        test_history.append(test_glucose)
        if len(test_history) > 20: # Keep history reasonably sized for test
            test_history.pop(0)

        # Announce meal at t=5 for 40g
        meal = 40.0 if t == 5 else 0.0
        if meal > 0:
             print(f"*** Meal Announced: {meal}g at t={t} ***")
             test_glucose += 20 # Simulate immediate meal effect for testing

        insulin, status = controller.compute_insulin(
            current_glucose=test_glucose,
            glucose_history=list(test_history), # Pass a copy
            # insulin_on_board=controller._calculate_iob(), # Removed, calculated internally
            meal_announcement=meal
        )

        print(f"t={status['time']:<3}: Glc={status['glucose']:<6.1f} | "
              f"Insulin={status['insulin_delivered']:.3f} | "
              f"IOB={status['iob_calculated']:.3f} | "
              f"Kp={status['gains']['Kp']:.4f} | "
              f"Ki={status['gains']['Ki']:.5f} | "
              f"Kd={status['gains']['Kd']:.5f} | "
              f"Alert={status['alert']}")

        # Add a small delay for readability if needed
        # import time
        # time.sleep(0.1)
    print("--- End Test ---")

# --- END OF FILE inference_controller.py ---