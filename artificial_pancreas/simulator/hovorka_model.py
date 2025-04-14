import numpy as np
from collections import deque

class HovorkaPatientSimulator:
    def __init__(self):
        # Safer initial state
        init_glucose = np.random.uniform(100, 140)
        self.Q1 = init_glucose / 18
        self.Q2 = init_glucose / 18
        self.S1 = 0
        self.S2 = 0
        self.Ra = 0
        self.glucose_history = deque([init_glucose/400]*6, maxlen=6)
        
        # Insulin sensitivity with variability
        self.insulin_effect = 0
        self.insulin_effect_half_life = 90  # minutes
        
        # Model parameters with ±10% variability
        self.F01 = 0.0097 * np.random.uniform(0.9, 1.1)
        self.EGP0 = 0.0161 * np.random.uniform(0.9, 1.1)
        self.VG = 1.88 * np.random.uniform(0.95, 1.05)
        self.VI = 0.05 * np.random.uniform(0.9, 1.1)
        self.k12 = 0.066 * np.random.uniform(0.9, 1.1)
        self.ke = 0.138 * np.random.uniform(0.9, 1.1)
        self.ka1 = 0.006 * np.random.uniform(0.8, 1.2)
        self.ka2 = 0.06 * np.random.uniform(0.8, 1.2)
        self.ka3 = 0.03 * np.random.uniform(0.8, 1.2)
        self.SIT = 51.2e-4 * np.random.uniform(0.8, 1.2)
        self.SID = 8.2e-4 * np.random.uniform(0.8, 1.2)
        self.SIE = 520e-4 * np.random.uniform(0.8, 1.2)
        
        # Meal tracking
        self.active_meals = []
        self.time_since_last_meal = 360

    def update(self, insulin_rate, time_step=1):
        # Hard safety limit
        insulin_rate = np.clip(insulin_rate, 0, 0.2)
        
        # Emergency glucose correction
        if self.Q1 * 18 < 60:
            self.Q1 = 110 / 18
            self.Q2 = 110 / 18
            return 110
        
        # Update insulin effect
        self.insulin_effect *= 0.5**(time_step/self.insulin_effect_half_life)
        self.insulin_effect += insulin_rate * time_step
        
        # Dynamic sensitivity
        sensitivity_factor = 1 / (1 + 0.01 * self.insulin_effect)
        
        insulin_input = insulin_rate / self.VI
        
        # Update insulin compartments
        dS1 = insulin_input - self.ka1 * self.S1
        dS2 = self.ka1 * self.S1 - self.ke * self.S2
        self.S1 = np.clip(self.S1 + dS1 * time_step, 0, 1000)
        self.S2 = np.clip(self.S2 + dS2 * time_step, 0, 1000)
        
        # Update meal absorption
        self.Ra = 0
        remaining_meals = []
        
        for amount, time_passed in self.active_meals:
            time_passed += time_step
            if time_passed < 360:  # 6-hour absorption
                # Dual-phase absorption
                phase1 = 0.6 * amount * np.exp(-0.03 * time_passed)
                phase2 = 0.4 * amount * np.exp(-0.01 * time_passed)
                self.Ra += (phase1 + phase2) / 180.16
                remaining_meals.append((amount, time_passed))
        
        self.active_meals = remaining_meals
        self.time_since_last_meal += time_step
        
        # Glucose fluxes
        EGP = max(self.EGP0 * (1 - self.SID * sensitivity_factor * self.S2), 0)
        U = max(self.F01 + (self.SIT * sensitivity_factor * self.S2) * self.Q1, 0)
        
        # Update glucose
        dQ1 = -(self.k12 + self.SIE * sensitivity_factor * self.S2) * self.Q1 + self.k12 * self.Q2 - U + EGP + self.Ra
        dQ2 = self.k12 * self.Q1 - self.k12 * self.Q2
        self.Q1 = np.clip(self.Q1 + dQ1 * time_step, 0, 1000/18)
        self.Q2 = np.clip(self.Q2 + dQ2 * time_step, 0, 1000/18)
        
        # Update history
        glucose = self.Q1 * 18
        self.glucose_history.append(glucose/400)
        return glucose

    def administer_meal(self, carbs):
        actual_carbs = np.clip(carbs, 0, 80)  # Max 80g carbs
        # Split into immediate and delayed components
        self.active_meals.append((actual_carbs * 0.7, 0))  # Fast carbs (70%)
        self.active_meals.append((actual_carbs * 0.3, 0))  # Slow carbs (30%)
        self.time_since_last_meal = 0

    def get_state(self):
        history = list(self.glucose_history)
        
        # Calculate trends
        trend_short = history[-1] - history[-2] if len(history) >= 2 else 0
        trend_medium = history[-1] - history[-4] if len(history) >= 4 else 0
        
        # Meal impact
        meal_impact = sum(amount * (1 - min(time_passed/360, 1)) 
                        for amount, time_passed in self.active_meals) / 80
        
        return np.concatenate([
            np.array([
                self.Q1 * 18 / 400,       # Current glucose
                self.S1 / 100,            # Insulin compartment 1
                self.S2 / 100,            # Insulin compartment 2
                min(meal_impact, 1),      # Meal impact
                trend_short,              # Short-term trend
                trend_medium,             # Medium-term trend
                self.time_since_last_meal < 360  # Recent meal flag
            ]),
            np.array(history[-4:])        # Last 4 glucose values
        ], dtype=np.float32)