# hardware_interface.py
import serial
from inference_controller import RL_PID_Controller

class HardwareController:
    def __init__(self):
        self.controller = RL_PID_Controller(agent_path="rl_pid_controller.pt")
        self.cgm = serial.Serial('/dev/ttyACM0', 9600)  # CGM
        self.pump = serial.Serial('/dev/ttyUSB0', 9600) # Insulin pump
        
    def run(self):
        while True:
            glucose = float(self.cgm.readline().decode().strip())
            insulin_rate, _ = self.controller.compute_insulin(
                current_glucose=glucose,
                glucose_history=self.controller.get_glucose_history()
            )
            self.pump.write(f"{insulin_rate}\n".encode())

if __name__ == "__main__":
    hw = HardwareController()
    hw.run()