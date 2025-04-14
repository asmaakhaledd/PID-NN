import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class RLAgent:
    def __init__(self, state_dim=11, action_dim=3, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # More conservative network architecture
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # Smoother than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        
        # Initialize with smaller weights
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.constant_(layer.bias, 0.01)
        
# Tighter gain bounds
        self.gain_min = torch.tensor([0.01, 0.0001, 0.0001], dtype=torch.float32)
        self.gain_max = torch.tensor([0.04, 0.0015, 0.0015], dtype=torch.float32) # Lowered max
        
        # Training setup
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5, weight_decay=1e-6)
        self.memory = deque(maxlen=30000)
        self.batch_size = 512
        self.gamma = 0.97
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.max_grad_norm = 0.5

    def get_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.policy_net(state_tensor).squeeze(0).numpy()
        
        gains = self.gain_min.numpy() + action * (self.gain_max.numpy() - self.gain_min.numpy())
        
        # Enhanced safety constraints
        current_glucose = state[0] * 400
        trend = state[5] * 400  # mg/dL per step
        predicted_glucose = current_glucose + trend * 3  # 3-step prediction
        
        # Emergency override
        if current_glucose < 70:
            return self.gain_min.numpy() * 0.1  # Minimal gains
        
        # Predictive safety
        if predicted_glucose < 90:
            reduction = 0.05 + (90 - predicted_glucose)/90 * 0.85
            gains = np.maximum(gains * reduction, [0.005, 0.00005, 0.00005])
        elif predicted_glucose < 100:
            gains = gains * 0.3
            
        # Trend-based adjustment
        if trend < -2:  # Rapid drop
            gains = gains * max(0.1, 1 + trend/10)
            
        # IOB consideration
        if state[2] > 0.3:  # High insulin compartment
            gains = gains * 0.5
            
        if explore and random.random() < self.epsilon:
            noise = np.random.normal(0, 0.001, size=self.action_dim)
            gains = np.clip(gains + noise, self.gain_min.numpy(), self.gain_max.numpy())
            
        return gains
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        current_q = self.policy_net(states)
        with torch.no_grad():
            next_q = self.policy_net(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gain_bounds': (self.gain_min, self.gain_max)
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        if 'gain_bounds' in checkpoint:
            self.gain_min, self.gain_max = checkpoint['gain_bounds']