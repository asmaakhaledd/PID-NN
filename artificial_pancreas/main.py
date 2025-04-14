import numpy as np
from training.train_controller import train_controller
import matplotlib.pyplot as plt
import torch
# Add to main.py after training
def export_model(agent, filename="rl_pid_controller.pt"):
    # Save full agent state (recommended)
    agent.save(filename)
    
    # Also export as TorchScript for production
    example_state = torch.zeros(11)  # Match your state dimension
    traced_model = torch.jit.trace(agent.policy_net, example_state)
    traced_model.save("rl_pid_controller_traced.pt")
    



def main():
    print("Training RL-PID controller with insulin delay compensation...")
    agent, stats = train_controller(
        target_glucose=100,
        episodes=500,
        steps_per_episode=720  # 12-hour episodes
    )
    export_model(agent)    
    # Analysis
    glucose = np.array(stats['glucose_history'])
    insulin = np.array(stats['insulin_history'])
    
    print(f"\nFinal Performance:")
    print(f"Time in Range (70-180 mg/dL): {np.mean((glucose >= 70) & (glucose <= 180)) * 100:.1f}%")
    print(f"Hypoglycemia Events: {stats['hypo_count']}")
    print(f"Hyperglycemia Events: {stats['hyper_count']}")
    print(f"Average Insulin: {np.mean(insulin):.2f} ± {np.std(insulin):.2f} mU/min")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Glucose trace
    plt.subplot(2, 2, 1)
    plt.plot(glucose[-1440:])  # Last 24 hours
    plt.axhline(70, color='r', linestyle='--', label='Hypo Threshold')
    plt.axhline(180, color='r', linestyle='--', label='Hyper Threshold')
    plt.title('Final 24-Hour Glucose Profile')
    plt.ylabel('Glucose (mg/dL)')
    plt.xlabel('Time (minutes)')
    plt.legend()
    
    # Insulin delivery
    plt.subplot(2, 2, 2)
    plt.plot(insulin[-1440:])
    plt.title('Insulin Delivery')
    plt.ylabel('mU/min)')
    plt.xlabel('Time (minutes)')
    
    # Training progress
    plt.subplot(2, 2, 3)
    plt.plot(stats['time_in_range'])
    plt.title('Time in Range During Training')
    plt.ylabel('TIR (%)')
    plt.xlabel('Episode')
    
    # Hypo/Hyper events
    plt.subplot(2, 2, 4)
    plt.bar(['Hypoglycemia', 'Hyperglycemia'], 
            [stats['hypo_count'], stats['hyper_count']])
    plt.title('Total Adverse Events')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()