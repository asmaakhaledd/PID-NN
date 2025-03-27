# inference_controller.py
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from collections import OrderedDict

class RL_PID_Controller:
    """Standalone RL-PID controller for artificial pancreas systems."""
    
    def __init__(self, agent_path: str = None, target_glucose: float = 100):
        self.pid = self._init_pid()
        self.target_glucose = target_glucose
        self.current_time = 0
        self.insulin_history = []
        
        # Initialize RL agent
        self.agent = self._init_agent()
        if agent_path:
            self.load_agent(agent_path)

    def _init_pid(self):
        """Initialize PID controller with safe defaults"""
        pid = type('', (), {})()
        pid.Kp = 0.02
        pid.Ki = 0.0005
        pid.Kd = 0.0005
        pid.prev_error = 0
        pid.integral = 0
        pid.insulin_doses = []
        pid.insulin_peak = 90
        pid.insulin_duration = 420
        pid.action_curve = self._create_action_curve(pid.insulin_peak, pid.insulin_duration)
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
        agent.gain_min = torch.tensor([0.01, 0.0001, 0.0001])
        agent.gain_max = torch.tensor([0.05, 0.002, 0.002])
        return agent

    def load_agent(self, path: str):
        """Load trained RL agent weights"""
        checkpoint = torch.load(path)
        self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded agent from {path}")

    def compute_insulin(self,
                       current_glucose: float,
                       glucose_history: list,
                       insulin_on_board: float = None,
                       meal_announcement: float = 0) -> Tuple[float, Dict]:
        """Main control function."""
        # Safety checks
        if current_glucose < 50:
            return 0.0, {"alert": "CRITICAL_HYPO"}
        elif current_glucose > 400:
            return 0.3, {"alert": "CRITICAL_HYPER"}
        
        # Get RL-adjusted PID gains
        state = self._create_state_vector(current_glucose, glucose_history,
                                        insulin_on_board, meal_announcement)
        gains = self._get_gains(state)
        self.pid.Kp, self.pid.Ki, self.pid.Kd = gains
        
        # Calculate PID output
        error = self.target_glucose - current_glucose
        self.pid.integral += error
        derivative = error - self.pid.prev_error
        self.pid.prev_error = error
        
        output = (gains[0] * error + 
                 gains[1] * self.pid.integral + 
                 gains[2] * derivative)
        
        # Apply safety constraints
        insulin_rate = np.clip(output, 0, 0.3)
        self._update_iob(insulin_rate)
        
        status = {
            "time": self.current_time,
            "glucose": current_glucose,
            "insulin": insulin_rate,
            "iob": self._calculate_iob(),
            "gains": {"Kp": gains[0], "Ki": gains[1], "Kd": gains[2]},
            "meal_active": meal_announcement > 0
        }
        
        self.current_time += 1
        return float(insulin_rate), status

    def _get_gains(self, state):
        """Get PID gains from RL agent"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            self.agent.policy_net.eval()
            action = self.agent.policy_net(state_tensor).squeeze(0).numpy()
        
        gains = self.agent.gain_min.numpy() + action * (self.agent.gain_max.numpy() - self.agent.gain_min.numpy())
        return np.clip(gains, self.agent.gain_min.numpy(), self.agent.gain_max.numpy())

    def _create_state_vector(self, glucose, history, iob, meal):
        """Create normalized state vector"""
        if len(history) < 6:
            history = [glucose] * (6 - len(history)) + history
            
        norm_glucose = glucose / 400
        norm_history = [h/400 for h in history[-6:]]
        trend_short = norm_history[-1] - norm_history[-2]
        trend_med = norm_history[-1] - norm_history[-4]
        
        return np.array([
            norm_glucose,
            0,  # Placeholder
            min(iob/3.0, 1) if iob else 0,
            min(meal/80, 1),
            trend_short,
            trend_med,
            float(meal > 0),
            *norm_history[-4:]
        ], dtype=np.float32)

    def _create_action_curve(self, peak, duration):
        """Insulin action curve"""
        t = np.arange(duration)
        phase1 = 0.4 * (1 - np.exp(-0.2 * t/peak))
        phase2 = 0.6 * np.exp(-0.2 * (t/duration)**2)
        curve = phase1 + phase2
        return curve / curve.sum()

    def _update_iob(self, dose):
        """Track insulin doses"""
        self.pid.insulin_doses.append(dose)
        if len(self.pid.insulin_doses) > self.pid.insulin_duration:
            self.pid.insulin_doses.pop(0)

    def _calculate_iob(self):
        """Calculate insulin-on-board"""
        if not self.pid.insulin_doses:
            return 0
            
        total = 0
        for t, dose in enumerate(self.pid.insulin_doses):
            if t < len(self.pid.action_curve):
                total += dose * self.pid.action_curve[t]
        return min(total * 1.3, 3.0)

    def reset(self):
        """Reset controller state"""
        self.pid = self._init_pid()
        self.current_time = 0


if __name__ == "__main__":
    # Test with sample data
    controller = RL_PID_Controller(agent_path="rl_pid_controller.pt")
    
    test_glucose = 150
    test_history = [155, 153, 151, 149, 147, 145]
    
    for t in range(10):
        test_glucose -= 2
        insulin, status = controller.compute_insulin(
            current_glucose=test_glucose,
            glucose_history=test_history,
            insulin_on_board=controller._calculate_iob(),
            meal_announcement=40 if t == 0 else 0
        )
        test_history.append(test_glucose)
        print(f"t={t}: Glucose={test_glucose} | Insulin={insulin:.3f} | Kp={status['gains']['Kp']:.4f}")