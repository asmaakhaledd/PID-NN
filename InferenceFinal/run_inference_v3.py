# --- START OF FILE run_inference_v3.py ---

import numpy as np
import torch
# import torch.nn as nn # Not needed directly here
from typing import Tuple, Dict # Keep this import
from collections import deque
import os
import sys
import argparse
import time # For simulation timing if needed

# --- Robust Import ---
try:
    from glucose_controller_rl.agent.rl_agent_v3 import RLAgentV3
    from glucose_controller_rl.controller.pid_controller_v3 import PIDControllerV3
    from glucose_controller_rl.utils import config_v3 as cfg
    from glucose_controller_rl.utils.state_utils import create_agent_state
    print("DEBUG: Imported V3 components via package path.")
except ImportError as e1:
    print(f"DEBUG: Package import failed ({e1}), attempting path modification...")
    try:
        # ... (Fallback import logic as before) ...
        from glucose_controller_rl.agent.rl_agent_v3 import RLAgentV3
        from glucose_controller_rl.controller.pid_controller_v3 import PIDControllerV3
        from glucose_controller_rl.utils import config_v3 as cfg
        from glucose_controller_rl.utils.state_utils import create_agent_state
        print("DEBUG: Imported V3 components after path modification.")
    except ImportError as e_inner:
         print(f"FATAL: Could not import required V3 modules. Check structure/path. Error: {e_inner}")
         sys.exit(1)

# ============================================================================
# === NEW STANDALONE CONTROLLER CLASS FOR INTEGRATION ===
# ============================================================================
class StandaloneControllerV3:
    """
    V3 RL-PID controller designed for integration with external simulations.
    Manages its own state (PID internal state, glucose history for trend).
    """
    def __init__(self, agent_path: str, target_glucose: float = cfg.TARGET_GLUCOSE):
        self.target_glucose = target_glucose
        self.current_step_count = 0 # Internal time counter

        print(f"--- Initializing Standalone Controller V3 ---")
        print(f"Using device: {cfg.DEVICE}")

        # Initialize internal PID logic controller
        self.pid = PIDControllerV3(target_glucose=target_glucose)

        # Initialize agent structure and load weights
        self.agent = RLAgentV3() # Uses config dims/bounds
        if agent_path and os.path.exists(agent_path):
            try:
                self.agent.load(agent_path)
                print(f"Agent loaded successfully into StandaloneControllerV3 from {agent_path}")
            except Exception as e:
                print(f"\n---!!! Failed to load agent {agent_path}: {e} !!!---")
                raise ValueError(f"Failed to load agent {agent_path}") from e
        elif not agent_path:
             raise ValueError("No agent path provided for StandaloneControllerV3.")
        else:
             raise FileNotFoundError(f"Agent file not found at {agent_path}")

        # State tracking needed BY THE CONTROLLER to create agent state
        self.glucose_history = deque(maxlen=cfg.STATE_GLUCOSE_HIST_LEN)
        self.time_since_last_meal = cfg.STATE_TIME_SINCE_MEAL_HORIZON # Initialize assuming no recent meal

    def reset(self, initial_glucose: float = 120):
        """Resets the controller's internal state."""
        self.pid.reset()
        self.glucose_history.clear()
        # Prime history buffer with initial glucose
        self.glucose_history.extend([initial_glucose] * cfg.STATE_GLUCOSE_HIST_LEN)
        self.time_since_last_meal = cfg.STATE_TIME_SINCE_MEAL_HORIZON
        self.current_step_count = 0
        print("Standalone Controller V3 state reset.")

    def compute_insulin(self, current_glucose: float, meal_grams: float = 0) -> Tuple[float, Dict]:
        """
        Computes the insulin delivery rate in U/h based on current state.
        Updates internal history and timers.

        Args:
            current_glucose (float): Current blood glucose reading (mg/dL).
            meal_grams (float, optional): Grams of carbohydrates for a meal occurring NOW.
                                           Defaults to 0.

        Returns:
            Tuple[float, Dict]:
                - float: The calculated insulin rate in Units per Hour (U/h).
                - Dict: A status dictionary containing details.
        """
        # 1. Update internal state trackers *before* calculations
        self.glucose_history.append(current_glucose) # Add current reading
        self.time_since_last_meal += 1
        if meal_grams > 0:
            self.time_since_last_meal = 0 # Reset meal timer

        # 2. --- Pre-calculation Hypo Check ---
        if current_glucose < cfg.REWARD_MILD_HYPO_THRESHOLD: # Use 80 from config
            alert = "CRITICAL_HYPO" if current_glucose < cfg.REWARD_CRITICAL_HYPO_THRESHOLD else "MILD_HYPO"
            # Call PID update to record zero dose and update PID state (like integral reset)
            _ = self.pid.update(current_glucose, self.current_step_count) # Returns 0.0
            insulin_uh = 0.0
            gains = (self.pid.Kp, self.pid.Ki, self.pid.Kd) # Report current PID gains
            status = self._create_status(current_glucose, insulin_uh, meal_grams > 0, *gains)
            status["alert"] = alert
            self.current_step_count += 1
            return insulin_uh, status

        # 3. Calculate IOB & Create Agent State
        current_iob_units = self.pid.calculate_iob()
        # Use internal trackers to create state
            # Inside StandaloneControllerV3.compute_insulin
        state = create_agent_state(
                current_glucose, self.glucose_history, current_iob_units,
                self.time_since_last_meal # <<< Corrected: 4 arguments passed
            )

        # 4. Get Gains from Agent
        kp, ki, kd = self.agent.get_gains(state, explore=False)

        # 5. Set PID gains (will be clipped inside PID)
        self.pid.set_gains(kp, ki, kd, self.agent.gain_min.numpy(), self.agent.gain_max.numpy())

        # 6. Update PID Controller (calculates final U/min, applies safety)
        final_dose_umin = self.pid.update(current_glucose, self.current_step_count)

        # 7. Convert final dose rate to U/h for return
        final_dose_uh = final_dose_umin * 60.0

        # 8. Create status dictionary
        status = self._create_status(current_glucose, final_dose_uh, meal_grams > 0, kp, ki, kd)
        if current_glucose > 400: status["alert"] = "CRITICAL_HYPER" # Check hyper > 400

        self.current_step_count += 1
        return float(final_dose_uh), status

    def _create_status(self, glucose, insulin_uh, meal_active_flag, kp, ki, kd):
         """Helper to create the status dictionary for output."""
         current_iob = self.pid.calculate_iob() # Get latest IOB
         status = {
            "time": self.current_step_count -1, # Time corresponds to the start of the step
            "glucose": float(glucose),
            "target_glucose": float(self.target_glucose),
            "insulin_delivered_uh": float(insulin_uh), # Rate in U/h
            "iob_calculated_units": float(current_iob), # IOB in Units
            "pid_integral": float(self.pid.integral),
            "pid_prev_error": float(self.pid.prev_error),
            "time_since_meal": int(self.time_since_last_meal),
            "meal_active": bool(meal_active_flag),
            "suggested_gains": {"Kp": float(kp), "Ki": float(ki), "Kd": float(kd)}, # Before PID clipping
            "active_gains": {"Kp": float(self.pid.Kp), "Ki": float(self.pid.Ki), "Kd": float(self.pid.Kd)}, # After PID clipping
            "alert": None
         }
         return status

# ============================================================================
# === END STANDALONE CONTROLLER CLASS ===
# ============================================================================


# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference with Trained RL-PID V3 Controller")
    parser.add_argument( "--agent_path", type=str, default=cfg.get_best_agent_path(), help=f"Path to agent checkpoint (default: {cfg.get_best_agent_path()})")
    parser.add_argument( "--duration", type=int, default=cfg.INFERENCE_TEST_DURATION, help=f"Simulation duration (min) (default: {cfg.INFERENCE_TEST_DURATION})")
    parser.add_argument( "--initial_glucose", type=float, default=cfg.INFERENCE_TEST_INITIAL_GLUCOSE, help=f"Starting glucose (default: {cfg.INFERENCE_TEST_INITIAL_GLUCOSE})")
    cli_args = parser.parse_args()

    # --- Initialize Controller ---
    try:
        # Use the new StandaloneControllerV3
        controller = StandaloneControllerV3(agent_path=cli_args.agent_path, target_glucose=cfg.TARGET_GLUCOSE)
    except Exception as e:
        print(f"---!!! Error initializing controller: {e} !!!---")
        sys.exit(1)

    # --- Simulation Setup ---
    print("\n--- V3 Controller Test Simulation ---")
    test_glucose = cli_args.initial_glucose
    controller.reset(initial_glucose=test_glucose) # Reset controller state

    test_meal_schedule = cfg.INFERENCE_TEST_MEALS; meal_idx = 0
    log_time, log_glucose, log_insulin_uh, log_iob = [], [], [], []
    log_kp, log_ki, log_kd = [], [], []

    # --- Simulation Loop ---
    for t in range(cli_args.duration):
        # Simple Glucose Dynamics (Replace with your target patient model call)
        current_sim_glucose = test_glucose # Use value from start of step for controller
        if t > 0:
            # Get last delivered dose rate (U/min) from *internal PID state* for dynamics
            # Note: controller.pid.insulin_doses stores the U/min rate
            last_dose_umin = controller.pid.insulin_doses[-1] if controller.pid.insulin_doses else 0.0
            glucose_change = cfg.INFERENCE_TEST_BASAL_DROP \
                             + (last_dose_umin * cfg.INFERENCE_TEST_INSULIN_SENSITIVITY)
            test_glucose = max(40.0, test_glucose + glucose_change) # Update glucose for *next* step

        # Meal Effect Simulation
        meal_grams = 0.0
        if meal_idx < len(test_meal_schedule) and t == test_meal_schedule[meal_idx][0]:
            carbs = test_meal_schedule[meal_idx][1]
            print(f"\n*** Meal Occurred: {carbs}g at t={t} ***")
            # Apply meal effect to the glucose that will be seen by the controller *next* step
            test_glucose += carbs * 0.7 # Stronger meal effect?
            meal_grams = float(carbs); meal_idx += 1

        # --- Call the Controller ---
        try:
            # Pass the glucose measured *at the start* of this step
            # Pass meal_grams occurring *at the start* of this step
            insulin_uh, status = controller.compute_insulin(current_sim_glucose, meal_grams)
        except Exception as e:
            print(f"!!! Error during compute_insulin at t={t}: {e} !!!")
            break # Stop simulation on error

        # --- Log Data ---
        log_time.append(t); log_glucose.append(status.get("glucose", current_sim_glucose))
        log_insulin_uh.append(status.get("insulin_delivered_uh", insulin_uh))
        log_iob.append(status.get("iob_calculated_units", -1.0))
        gains = status.get("active_gains", {"Kp": -1, "Ki": -1, "Kd": -1}) # Log active gains
        log_kp.append(gains.get("Kp")); log_ki.append(gains.get("Ki")); log_kd.append(gains.get("Kd"))

        # Print Status Periodically
        if t % 60 == 0 or meal_grams > 0 or status.get("alert") is not None or t == cli_args.duration - 1:
             print(f"t={status.get('time', t):<3}: Glc={status.get('glucose', -1):<6.1f} | "
                   f"Ins={status.get('insulin_delivered_uh', -1):.2f} U/h | "
                   f"IOB={status.get('iob_calculated_units', -1):.2f} U | "
                   f"Kp={gains.get('Kp', -1):.3f} | Ki={gains.get('Ki', -1):.4f} | Kd={gains.get('Kd', -1):.4f} | "
                   f"Alert={status.get('alert', 'None')}")


    print("\n--- End Test Simulation ---")

    # --- Plotting ---
    # (Plotting code remains the same as previous version)
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        axs[0].plot(log_time, log_glucose, label='Glucose (mg/dL)', color='tab:blue')
        axs[0].axhline(70, color='red', linestyle='--', lw=1, label='Hypo (70)')
        axs[0].axhline(180, color='orange', linestyle='--', lw=1, label='Hyper (180)')
        axs[0].set_ylabel('Glucose (mg/dL)'); axs[0].set_title(f'Inference Results ({os.path.basename(cli_args.agent_path)})')
        axs[0].legend(fontsize='small'); axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[1].plot(log_time, log_insulin_uh, label='Insulin Rate (U/h)', color='tab:green', drawstyle='steps-post')
        axs[1].set_ylabel('Insulin (U/h)'); axs[1].legend(fontsize='small')
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[2].plot(log_time, log_iob, label='Calculated IOB (Units)', color='tab:purple')
        axs[2].set_ylabel('IOB (Units)'); axs[2].legend(fontsize='small')
        axs[2].grid(True, linestyle=':', alpha=0.7)
        ax_gains2 = axs[3].twinx()
        p1, = axs[3].plot(log_time, log_kp, label='Kp', color='tab:red', alpha=0.8)
        p2, = ax_gains2.plot(log_time, log_ki, label='Ki', color='tab:cyan', alpha=0.7, linestyle=':')
        p3, = ax_gains2.plot(log_time, log_kd, label='Kd', color='tab:gray', alpha=0.7, linestyle='--')
        axs[3].set_ylabel('Kp Gain', color='tab:red'); ax_gains2.set_ylabel('Ki / Kd Gains', color='gray')
        axs[3].set_xlabel('Time (minutes)'); axs[3].legend(handles=[p1, p2, p3], fontsize='small', loc='upper left')
        axs[3].grid(True, linestyle=':', alpha=0.7, axis='x') # Grid only on primary x
        ax_gains2.grid(False)
        axs[3].tick_params(axis='y', labelcolor='tab:red')
        ax_gains2.tick_params(axis='y', labelcolor='gray')
        axs[3].set_ylim(bottom=0); ax_gains2.set_ylim(bottom=0)
        plt.tight_layout(); plt.show()
    except ImportError: print("\nMatplotlib not found. Skipping plot.")
    except Exception as e: print(f"\nError during plotting: {e}")

# --- END OF FILE run_inference_v3.py ---