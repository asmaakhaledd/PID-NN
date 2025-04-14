import numpy as np
from controller.pid_controller import PIDController
from controller.rl_agent import RLAgent
from simulator.hovorka_model import HovorkaPatientSimulator
import torch

def train_controller(target_glucose=100, episodes=1000, steps_per_episode=720):
    pid = PIDController()
    agent = RLAgent(state_dim=11, action_dim=3)
    
    stats = {
        'glucose_history': [],
        'insulin_history': [],
        'hypo_count': 0,
        'hyper_count': 0,
        'episode_rewards': [],
        'time_in_range': [],
        'meal_responses': []
    }

    def get_meal_schedule(episode):
        """More varied meal schedule with random sizes and times"""
        if episode < 50:  # Initial training with small meals
            return [(180, 20), (480, 25)]
        elif episode < 200:  # Medium meals
            return [(120, 30), (360, 35), (600, 25)]
        else:  # Full variability
            num_meals = np.random.randint(2, 4)
            times = sorted(np.random.randint(60, steps_per_episode-120, num_meals))
            sizes = np.random.uniform(20, 50, num_meals)
            return list(zip(times, sizes))

    def calculate_reward(glucose, iob, target):
        """Enhanced reward function with better balance"""
        if glucose < 70:
            severity = max(70 - glucose, 0)
            # SIGNIFICANTLY increased penalty, non-linear severity scaling
            penalty = 30 + severity**1.5 # Was 10 + severity * 1.2
            # Increase IOB factor too
            return -penalty * (1 + 1.0 * iob) # Was 0.6 * iob
        elif glucose > 180:
            # Hyper penalty can also be adjusted if needed, but hypo is primary issue
            severity = max(glucose - 180, 0)
            return -8 * (1 + severity/30) # Keep hyper penalty for now
        else:
            # Positive reward shape - maybe make it narrower?
            glucose_reward = 8 * np.exp(-0.5*((glucose-target)/20)**2) # Was /25
            # IOB penalty for time in range
            iob_penalty = 2 * np.log(1 + 0.5 * iob) # Keep this relatively small
            return glucose_reward - iob_penalty

    for episode in range(episodes):
        patient = HovorkaPatientSimulator()
        pid.reset()
        
        # Wider initial glucose range
        initial_glucose = np.random.uniform(85, 150)
        patient.Q1 = initial_glucose / 18
        patient.Q2 = initial_glucose / 18
        
        # Get meal schedule
        meal_schedule = get_meal_schedule(episode)
        next_meal_idx = 0
        
        episode_reward = 0
        hypo_count = 0
        hyper_count = 0
        meal_response_data = []
        
        for step in range(steps_per_episode):
            # Administer scheduled meals
            if next_meal_idx < len(meal_schedule) and step >= meal_schedule[next_meal_idx][0]:
                carbs = meal_schedule[next_meal_idx][1]
                patient.administer_meal(carbs)
                next_meal_idx += 1
                meal_response_data.append((step, carbs))
            
            current_glucose = patient.Q1 * 18
            state = patient.get_state()
            
            # Get action with exploration
            if episode < 30:  # Initial conservative phase
                gains = [0.04, 0.001, 0.001]  # More conservative
            else:
                gains = agent.get_action(state, explore=True)
            
            pid.set_gains(*gains)
            
            # Calculate insulin with IOB consideration
            insulin_rate = pid.update(current_glucose, target_glucose, step)
            next_glucose = patient.update(insulin_rate)
            
            # Track stats
            stats['glucose_history'].append(next_glucose)
            stats['insulin_history'].append(insulin_rate)
            
            # Calculate reward with enhanced function
            current_iob = pid._calculate_iob()
            reward = calculate_reward(next_glucose, current_iob, target_glucose)
            
            # Count events
            if next_glucose < 70:
                hypo_count += 1
                if next_glucose < 60:  # Emergency correction
                    patient.Q1 = 110/18  # Correct to 110 mg/dL
                    patient.Q2 = 110/18
            elif next_glucose > 180:
                hyper_count += 1
            
            # Store experience
            done = step == steps_per_episode - 1
            agent.remember(state, gains, reward, patient.get_state(), done)
            
            # Train after initial phase
            if episode >= 30 and len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            episode_reward += reward
            
            # Early termination for extreme states
            if next_glucose < 50 or next_glucose > 400:
                break
        
        # Update statistics
        glucose_window = np.array(stats['glucose_history'][-steps_per_episode:])
        time_in_range = np.mean((glucose_window >= 70) & (glucose_window <= 180)) * 100
        
        stats['hypo_count'] += hypo_count
        stats['hyper_count'] += hyper_count
        stats['episode_rewards'].append(episode_reward)
        stats['time_in_range'].append(time_in_range)
        stats['meal_responses'].extend(meal_response_data)
        
        print(f"Episode {episode+1}/{episodes}, "
              f"Glucose: {patient.Q1*18:.1f} mg/dL, "
              f"Insulin: {np.mean(stats['insulin_history'][-steps_per_episode:]):.3f} U/min, "
              f"Hypo: {hypo_count}, Hyper: {hyper_count}, "
              f"TIR: {time_in_range:.1f}%")
        
        # Save model periodically
        if (episode + 1) % 50 == 0:
            agent.save(f"agent_episode_{episode+1}.pt")
        
        # Early stopping if performance is good
        if episode > 200 and np.mean(stats['time_in_range'][-50:]) > 85 and hypo_count < 2:
            print("Early stopping - good performance achieved")
            break
    
    return agent, stats