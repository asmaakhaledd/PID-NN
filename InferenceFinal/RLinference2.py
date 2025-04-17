# --- START OF FILE inference_controller.py ---

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from collections import OrderedDict, deque
import os
import sys # For path modification if needed

# --- Robust Import ---
try:
    # Assumes running from project root (PID-NN) or path is setup
    from artificial_pancreas.controller.rl_agent import RLAgent
    print("DEBUG: Imported RLAgent via package path.")
except ImportError:
    print("DEBUG: Package import failed, attempting path modification...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assumes InferenceFinal is direct child of PID-NN which contains artificial_pancreas
        project_root = os.path.dirname(script_dir)
        if 'artificial_pancreas' not in os.listdir(project_root):
             # Maybe script is inside artificial_pancreas/inference? Go up one more.
             project_root = os.path.dirname(project_root)

        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"DEBUG: Added '{project_root}' to sys.path from inference_controller.py")
        from artificial_pancreas.controller.rl_agent import RLAgent
        print("DEBUG: Imported RLAgent via sys.path modification.")
    except ImportError as e_inner:
        print(f"FATAL: Could not import RLAgent. Check structure and path. Error: {e_inner}")
        sys.exit(1) # Exit if import fails


class RL_PID_Controller:
    """
    Standalone RL-PID controller for artificial pancreas systems.
    Returns insulin rate in Units per Hour (U/h).
    Safety logic is applied within compute_insulin.
    """

    def __init__(self, agent_path: str = None, target_glucose: float = 100):
        """Initializes the RL-PID controller for inference."""
        self.target_glucose = target_glucose
        self.current_time = 0
        self.pid = self._init_pid()
        self.agent = self._init_agent() # Initialize structure first
        if agent_path and os.path.exists(agent_path): # Check if path exists
            try:
                # Use the robust load method defined in RLAgent
                self.agent.load(agent_path) # Loads weights, gain bounds into self.agent
            except Exception as e:
                print(f"\n---!!! Failed to load agent {agent_path}: {e} !!!---")
                print("--- Continuing with default PID gains. ---")
                self._use_default_gains() # Set default gains if load failed
        elif agent_path:
             print(f"Warning: Agent path provided but not found: {agent_path}. Using default PID gains.")
             self._use_default_gains()
        else:
            print("Warning: No agent path provided. Using default PID gains.")
            self._use_default_gains()

    def _use_default_gains(self):
        """Helper to set default low gains when agent isn't loaded."""
        # Use the minimum bounds defined in the agent structure
        self.pid.Kp, self.pid.Ki, self.pid.Kd = self.agent.gain_min.numpy()


    def _init_pid(self):
        """Initialize PID controller state and parameters for inference."""
        pid = type('', (), {})()
        pid.Kp, pid.Ki, pid.Kd = 0.01, 0.0001, 0.0001 # Init with min gains
        pid.prev_error = 0.0; pid.integral = 0.0
        pid.insulin_doses = deque(maxlen=420) # U/min
        pid.insulin_peak = 90; pid.insulin_duration = 420
        pid.min_insulin = 0.001 # U/min
        pid.max_insulin = 0.25  # ** Use Lowered Max Insulin Rate (U/min) **
        pid.dosing_interval = 15; pid.last_dose_time = -pid.dosing_interval

        # Corrected Insulin Action Curve Setup
        raw_curve = self._create_raw_action_curve(pid.insulin_peak, pid.insulin_duration)
        curve_sum = raw_curve.sum()
        if curve_sum > 0:
            normalized_raw_curve = raw_curve / curve_sum
            pid.cumulative_action_curve = np.cumsum(normalized_raw_curve)
            if len(pid.cumulative_action_curve) > 0:
                 pid.cumulative_action_curve /= pid.cumulative_action_curve[-1]
        else: pid.cumulative_action_curve = np.zeros(pid.insulin_duration)
        return pid

    def _init_agent(self):
        """Initialize RL agent structure and attach robust load method."""
        agent_struct = type('', (), {})()
        agent_struct.policy_net = nn.Sequential(
            nn.Linear(11, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Sigmoid()
        )
        # Initialize with agent's default bounds (will be overwritten on load)
        temp_agent = RLAgent() # Create temporary full agent to get defaults
        agent_struct.gain_min = temp_agent.gain_min
        agent_struct.gain_max = temp_agent.gain_max
        agent_struct.epsilon = temp_agent.epsilon_min # Add epsilon attribute
        # Bind the load method from RLAgent class to this instance
        agent_struct.load = RLAgent.load.__get__(agent_struct, type(agent_struct))
        return agent_struct

    def _create_raw_action_curve(self, peak, duration):
        t = np.arange(duration); peak = max(peak, 1.0); duration = max(duration, 1.0)
        p1 = 0.4 * (1.0 - np.exp(-0.2 * t / peak)); p2 = 0.6 * np.exp(-0.2 * (t / duration)**2)
        return p1 + p2

    def _calculate_iob(self):
        if not self.pid.insulin_doses: return 0.0
        active_units = 0.0; n = len(self.pid.insulin_doses); dur = self.pid.insulin_duration
        curve = self.pid.cumulative_action_curve
        for i in range(n):
            dose_umin = self.pid.insulin_doses[i]
            if dose_umin > 0:
                t_since = n - 1 - i
                if 0 <= t_since < dur:
                    f_remain = max(0.0, 1.0 - curve[t_since])
                    active_units += dose_umin * f_remain
        return min(active_units, 5.0) # Use 5.0 U cap

    def _create_state_vector(self, glucose, history, iob, meal):
        """Create normalized 11-dimensional state vector for the RL agent."""
        hist_list = list(history)
        while len(hist_list) < 6:
            hist_list.insert(0, glucose)
        norm_glucose = np.clip(glucose / 400.0, 0, 1)
        norm_history = [np.clip(h / 400.0, 0, 1) for h in hist_list[-6:]]
        trend_short = norm_history[-1] - norm_history[-2] if len(norm_history) >= 2 else 0.0
        trend_med = norm_history[-1] - norm_history[-4] if len(norm_history) >= 4 else 0.0
        iob_cap_for_norm = 5.0
        norm_iob = np.clip(iob / iob_cap_for_norm, 0, 1) if iob is not None else 0.0
        meal_max_for_norm = 80.0
        norm_meal = np.clip(meal / meal_max_for_norm, 0, 1)

        state = np.array([
            norm_glucose,           # 0
            0.0,                    # 1: Proxy for S1
            norm_iob,               # 2: Proxy for S2 <- Original value
            norm_meal,              # 3
            trend_short,            # 4
            trend_med,              # 5
            float(meal > 0),        # 6
            *norm_history[-4:]      # 7, 8, 9, 10
        ], dtype=np.float32)

        # ========================================================
        # <<< TEMPORARY DIAGNOSTIC CHANGE IS HERE >>>
        # --- !! TEMPORARY DIAGNOSTIC CHANGE !! ---
        # Force state[2] to a fixed value to test agent sensitivity
        state[2] = 0.0  # Forcing it low for this test run
        # state[2] = 0.5 # Alternative: Force it mid-range

        # Add a print statement to confirm the change is active
        print(f"DEBUG: Forcing state[2] (IOB proxy) to: {state[2]:.2f}")
        # --- !! END TEMPORARY DIAGNOSTIC CHANGE !! ---
        # ========================================================

        if len(state) != 11: raise ValueError(f"State len {len(state)} != 11.")
        return state

    def _get_gains(self, state):
        """Gets gains from agent - NO internal safety logic here."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_raw = self.agent.policy_net(state_tensor).squeeze(0).numpy()

        g_min, g_max = self.agent.gain_min.numpy(), self.agent.gain_max.numpy()
        scaled = g_min + action_raw * (g_max - g_min)
        final_gains = np.clip(scaled, g_min, g_max)
        # print(f"DEBUG GAINS @ t={self.current_time}: Raw={action_raw} -> Final={final_gains}") # Optional
        return final_gains

    def compute_insulin(self, current_glucose: float, glucose_history: list,
                        meal_announcement: float = 0) -> Tuple[float, Dict]:
        """Computes insulin rate (U/h), applying safety logic mirroring training."""

        # Pre-calculation Hypo Checks
        if current_glucose < 70:
            alert = "CRITICAL_HYPO" if current_glucose < 50 else "MILD_HYPO"
            self._update_iob(0.0); self.pid.integral = 0
            gains_to_report = (self.pid.Kp, self.pid.Ki, self.pid.Kd)
            status = self._create_status(current_glucose, 0.0, meal_announcement, *gains_to_report)
            status["alert"] = alert; self.current_time += 1
            return 0.0, status

        # 1. State & IOB
        current_iob_units = self._calculate_iob()
        state_vector = self._create_state_vector(current_glucose, glucose_history,
                                                 current_iob_units, meal_announcement)

        # 2. Get RL-adjusted PID gains
        kp, ki, kd = self._get_gains(state_vector)
        self.pid.Kp, self.pid.Ki, self.pid.Kd = kp, ki, kd

        # 3. PID Calculation
        error = current_glucose - self.target_glucose
        integral_change = error
        potential_integral = self.pid.integral + integral_change
        integral_min, integral_max = -300, 300
        self.pid.integral = np.clip(potential_integral, integral_min, integral_max)
        derivative = error - self.pid.prev_error

        # Raw PID output tendency (U/min)
        output_tendency = (kp * error + ki * self.pid.integral + kd * derivative)

        # 4. Dose Eligibility & Safety Logic
        final_dose_umin = 0.0
        eligible_for_dose = (self.current_time - self.pid.last_dose_time >= self.pid.dosing_interval)

        if eligible_for_dose:
            # Apply **stricter** IOB reduction (mirroring training)
            iob_reduction_factor = max(0.0, 1.0 - current_iob_units / 1.5) # Denominator reduced
            potential_dose_umin = output_tendency * iob_reduction_factor

            # Clip to absolute max insulin rate (now lower: 0.25 U/min)
            potential_dose_umin = np.clip(potential_dose_umin, 0, self.pid.max_insulin)

            # Apply Glucose Safety Constraints (mirroring training - stricter)
            if current_glucose < 95:   # Increased threshold
                potential_dose_umin = 0.0
            elif current_glucose < 115: # Increased threshold
                potential_dose_umin = np.clip(potential_dose_umin, 0, self.pid.max_insulin * 0.05) # Kept heavy reduction

            # Predictive/Trend Safety (Optional - Add if used in training)
            # ...

            if potential_dose_umin >= self.pid.min_insulin:
                final_dose_umin = potential_dose_umin
                self.pid.last_dose_time = self.current_time

            if output_tendency > 0 and final_dose_umin < output_tendency * iob_reduction_factor * 0.9:
                 self.pid.integral = max(integral_min, self.pid.integral - abs(integral_change) * 0.5)

        # 5. Store Previous Error
        self.pid.prev_error = error

        # 6. Update IOB tracking deque (U/min)
        self._update_iob(final_dose_umin)

        # 7. Convert final dose rate to U/h for return
        final_dose_uh = final_dose_umin * 60.0

        # 8. Create status dictionary
        status = self._create_status(current_glucose, final_dose_uh, meal_announcement, kp, ki, kd)
        if current_glucose > 400: status["alert"] = "CRITICAL_HYPER"

        self.current_time += 1
        return float(final_dose_uh), status

    def _update_iob(self, dose_umin):
        self.pid.insulin_doses.append(float(dose_umin))

    def _create_status(self, glucose, insulin_uh, meal, kp, ki, kd):
         current_iob = self._calculate_iob()
         status = {
            "time": self.current_time, "glucose": float(glucose),
            "target_glucose": float(self.target_glucose),
            "insulin_delivered_uh": float(insulin_uh),
            "iob_calculated_units": float(current_iob),
            "pid_integral": float(self.pid.integral),
            "pid_prev_error": float(self.pid.prev_error),
            "meal_active": float(meal) > 0,
            "gains": {"Kp": float(kp), "Ki": float(ki), "Kd": float(kd)},
            "alert": None
         }
         return status

    def reset(self):
        self.pid = self._init_pid()
        self.current_time = 0
        print("RL_PID_Controller state reset.")

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    AGENT_CHECKPOINT_FILE = "rl_pid_controller.pt" # Or "rl_pid_controller.pt"
    print(f"Attempting to load agent from: {AGENT_CHECKPOINT_FILE}")
    controller = RL_PID_Controller(agent_path=AGENT_CHECKPOINT_FILE, target_glucose=100)

    print("\n--- Controller Test Simulation ---")
    test_duration_minutes = 720; initial_glucose = 150.0
    test_glucose = initial_glucose; test_history = [initial_glucose] * 6
    test_meal_schedule = [(120, 30), (360, 35), (600, 25)]; meal_idx = 0
    log_time, log_glucose, log_insulin_uh, log_iob = [], [], [], []

    for t in range(test_duration_minutes):
        if t > 0:
            last_dose_umin = controller.pid.insulin_doses[-1] if controller.pid.insulin_doses else 0.0
            glucose_change = -0.5 - (last_dose_umin * 15)
            test_glucose = max(40.0, test_glucose + glucose_change)
        meal = 0.0
        if meal_idx < len(test_meal_schedule) and t == test_meal_schedule[meal_idx][0]:
            carbs = test_meal_schedule[meal_idx][1]
            print(f"\n*** Meal Announced: {carbs}g at t={t} ***")
            test_glucose += carbs * 0.5; meal = float(carbs); meal_idx += 1
        test_history.append(test_glucose); test_history = test_history[-20:]

        try:
            insulin_uh, status = controller.compute_insulin(test_glucose, list(test_history), meal)
        except Exception as e:
            print(f"!!! Error compute_insulin t={t}: {e} !!!"); insulin_uh, status = 0.0, {}

        log_time.append(t); log_glucose.append(status.get("glucose", test_glucose))
        log_insulin_uh.append(status.get("insulin_delivered_uh", insulin_uh))
        log_iob.append(status.get("iob_calculated_units", controller._calculate_iob()))

        if t % 60 == 0 or meal > 0 or status.get("alert") is not None or t == test_duration_minutes - 1:
             gains_dict = status.get('gains', {})
             print(f"t={status.get('time', t):<3}: Glc={status.get('glucose', -1):<6.1f} | "
                   f"Ins={status.get('insulin_delivered_uh', -1):.2f} U/h | "
                   f"IOB={status.get('iob_calculated_units', -1):.2f} U | "
                   f"Kp={gains_dict.get('Kp', -1):.3f} | Ki={gains_dict.get('Ki', -1):.4f} | Kd={gains_dict.get('Kd', -1):.4f} | "
                   f"Alert={status.get('alert', 'None')}")

    print("\n--- End Test Simulation ---")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axs[0].plot(log_time, log_glucose, label='Glucose (mg/dL)', color='tab:blue')
        axs[0].axhline(70, color='red', linestyle='--', linewidth=1, label='Hypo (70)')
        axs[0].axhline(180, color='orange', linestyle='--', linewidth=1, label='Hyper (180)')
        axs[0].set_ylabel('Glucose (mg/dL)'); axs[0].set_title('Inference Test Simulation Results')
        axs[0].legend(fontsize='small'); axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[1].plot(log_time, log_insulin_uh, label='Insulin Rate (U/h)', color='tab:green', drawstyle='steps-post')
        axs[1].set_ylabel('Insulin (U/h)'); axs[1].legend(fontsize='small')
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[2].plot(log_time, log_iob, label='Calculated IOB (Units)', color='tab:purple')
        axs[2].set_ylabel('IOB (Units)'); axs[2].set_xlabel('Time (minutes)')
        axs[2].legend(fontsize='small'); axs[2].grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(); plt.show()
    except ImportError: print("\nMatplotlib not found. Skipping plot.")
    except Exception as e: print(f"\nError during plotting: {e}")

# --- END OF FILE inference_controller.py ---