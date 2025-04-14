# --- START OF FILE pid_controller.py ---

import numpy as np

class PIDController:
    def __init__(self, Kp=0.02, Ki=0.0005, Kd=0.0005):
        # Ultra-conservative initial gains
        self.Kp = np.clip(Kp, 0.01, 0.05)
        self.Ki = np.clip(Ki, 0.0001, 0.002)
        self.Kd = np.clip(Kd, 0.0001, 0.002)

        # Extended insulin pharmacokinetics
        self.insulin_peak = 90       # Slower absorption (was 60)
        self.insulin_duration = 420  # Longer action (was 300)
        self.action_curve = self._create_action_curve()

        # Controller state
        self.prev_error = 0
        self.integral = 0
        self.last_time = None
        self.insulin_doses = []
        self.min_insulin = 0.001     # Reduced minimum
        self.max_insulin = 0.3       # Reduced from 1.0

        # --- NEW: Dosing interval logic ---
        self.dosing_interval = 15   # Dose every 15 minutes
        self.last_dose_time = -self.dosing_interval # Allow immediate first dose if needed
        # --- END NEW ---

    def _create_action_curve(self):
        """More physiological insulin profile"""
        t = np.arange(self.insulin_duration)
        # Biphasic curve
        phase1 = 0.4 * (1 - np.exp(-0.2 * t/self.insulin_peak))
        phase2 = 0.6 * np.exp(-0.2 * (t/self.insulin_duration)**2)
        curve = phase1 + phase2
        # Ensure curve sums to 1 to represent fraction of total effect
        # Handle potential division by zero if sum is zero (though unlikely here)
        curve_sum = curve.sum()
        return curve / curve_sum if curve_sum > 0 else np.zeros_like(curve)


    def _calculate_iob(self):
        """Conservative IOB estimation with safety margin"""
        if not self.insulin_doses:
            return 0

        # Calculate IOB based on the *actual* doses delivered
        # The action curve represents the fraction of the total effect active at time t
        # Summing dose * action_curve[time_since_dose] gives active insulin units
        active_insulin = 0
        current_sim_step = len(self.insulin_doses) # Assuming each entry is one time step

        # Iterate through recorded doses from oldest potentially active to newest
        start_index = max(0, current_sim_step - self.insulin_duration)
        for i in range(start_index, current_sim_step):
            dose = self.insulin_doses[i]
            if dose > 0: # Only consider actual insulin doses
                time_since_dose = current_sim_step - 1 - i # Time steps passed since dose i
                if 0 <= time_since_dose < self.insulin_duration:
                     # action_curve index corresponds to time since dose
                    active_fraction = self.action_curve[time_since_dose]
                    active_insulin += dose * active_fraction

        # Apply safety factor and cap AFTER summing contributions
        # The original safety factor seemed to multiply the contribution, which might overestimate IOB.
        # Let's apply a simpler cap or factor if needed, but the sum itself *is* the IOB.
        # A safety factor might be better applied when *using* the IOB, not calculating it.
        # Reverting the safety factor here for a more standard IOB calculation.
        # The cap is still reasonable.
        return min(active_insulin, 3.0) # Absolute cap


    def update(self, current_glucose, target_glucose, current_time):
        # Emergency override - Keep this high up
        if current_glucose < 70:
            final_dose = 0.0
            self.insulin_doses.append(final_dose) # Record zero dose
            # Reset integral if stopping insulin due to hypo
            self.integral = 0
            return final_dose

        dt = 1 if self.last_time is None else current_time - self.last_time
        if dt <= 0: dt = 1 # Ensure dt is positive
        self.last_time = current_time

        # --- SIGN CORRECTION ---
        # Error should be positive when glucose is above target
        error = current_glucose - target_glucose
        # --- END SIGN CORRECTION ---

        # --- Predictive Target Adjustment (Optional Refinement) ---
        # This prediction was based on the OLD error definition.
        # If keeping prediction, it needs recalculating based on glucose levels directly.
        # Let's simplify for now and remove the effective_target adjustment,
        # relying directly on the corrected error.
        # If you want prediction, add it back carefully based on glucose trend.
        # Example simplified prediction (can be more complex):
        # last_glucose = self.prev_glucose if hasattr(self, 'prev_glucose') else current_glucose # Need to store previous glucose
        # glucose_change = current_glucose - last_glucose
        # predicted_glucose = current_glucose + glucose_change * 2
        # self.prev_glucose = current_glucose # Store for next step
        # (Safety checks using predicted_glucose below should use this)
        # For now, we use current_glucose in safety checks.

        # --- Refined Integral Update & Anti-Windup ---
        # Store potential integral change
        integral_change = error * dt
        potential_integral = self.integral + integral_change

        # Apply anti-windup bounds *before* using the integral value
        self.integral = np.clip(potential_integral, -300, 300) # Adjust bounds as needed

        # --- Derivative Calculation (Uses previous ERROR) ---
        # Ensure prev_error is stored *after* potential modification
        # We store the error from the *previous* step for the derivative calculation
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        # Store the *current* error for the *next* step's derivative calculation at the end


        # Calculate raw PID output (positive output now means insulin needed)
        output = (self.Kp * error +
                  self.Ki * self.integral +  # Use the clipped integral
                  self.Kd * derivative)

        # --- Dosing Interval Logic ---
        final_dose = 0.0 # Default to zero dose
        eligible_for_dose = (current_time - self.last_dose_time >= self.dosing_interval)

        if eligible_for_dose:
            current_iob = self._calculate_iob()
            # Stronger IOB reduction factor
            iob_reduction_factor = max(0.0, 1.0 - current_iob / 2.0) # Was / 3.0
            # OR even non-linear:
            # iob_reduction_factor = max(0.0, (1.0 - current_iob / 2.5)**2)
            potential_dose = output * iob_reduction_factor

            # Clip potential dose to valid range (0 to max_insulin)
            potential_dose = np.clip(potential_dose, 0, self.max_insulin)

            # --- Glucose Safety Constraints ---
            if current_glucose < 90: # More cautious threshold
                potential_dose = 0.0
            elif current_glucose < 110: # More cautious range
                potential_dose = np.clip(potential_dose, 0, self.max_insulin * 0.05) # Heavy reduction
            # else: potential_dose remains as calculated after IOB reduction & clipping

            # Final check: only deliver if above minimum threshold
            if potential_dose >= self.min_insulin:
                final_dose = potential_dose
                self.last_dose_time = current_time # Reset timer only if dose is given
            else:
                final_dose = 0.0

            # --- Anti-Windup Adjustment ---
            # If PID output suggested insulin (output > 0) but safety/interval/min_dose
            # prevented it (final_dose == 0), prevent integral windup.
            # Reduce the integral slightly or clamp it. Let's slightly reduce.
            if output > 0 and final_dose < output * iob_reduction_factor * 0.9: # Check if dose was significantly reduced/zeroed
                 # Reduce integral gently if we wanted to dose but couldn't/didn't significantly
                 self.integral = max(-300, self.integral - abs(integral_change) * 0.5) # Pull back integral


        # --- Store Previous Error ---
        # Store the error calculated in *this* step for the *next* step's derivative
        self.prev_error = error

        # Record the *actual* dose administered
        self.insulin_doses.append(final_dose)

        # Return the actual dose
        return float(final_dose)

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = np.clip(float(Kp), 0.01, 0.05)
        self.Ki = np.clip(float(Ki), 0.0001, 0.002)
        self.Kd = np.clip(float(Kd), 0.0001, 0.002)

    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.last_time = None
        self.insulin_doses = []
        # --- NEW: Reset dosing timer ---
        self.last_dose_time = -self.dosing_interval
        # --- END NEW ---

# --- END OF FILE pid_controller.py ---