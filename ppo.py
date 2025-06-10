import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.module):
    def __init__(self) -> None:
        super(Actor).__init__()
        self.optim = optim.Adam(self.parameters())
        self.lstm = nn.LSTM(input_size=0, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Tanh(32, 2)
        
        
    def forward(self, 
                state: torch.Tensor, 
                sample_action: bool = False) -> torch.Tensor:
        output, _ = self.lstm(state)[:, -1, :]
        output = self.fc(output)
        # For inference, directly sample stochasically on a normal distribution using mean and std
        if sample_action:
            return torch.normal(output[0], output[1])
        # For training, return parameters directly
        return output
    
    
    def surrogate_objective(self, 
                            critic: nn,
                            rewards: torch.Tensor, 
                            states: torch.Tensor, 
                            next_states: torch.Tensor, 
                            past_policy: nn,
                            discount_factor: float = 0.99,
                            clipping_parameter: float = 0.2,
        ) -> torch.Tensor:
        
        # For non-discrete action space; Compute the ratio between the current and past policies
        # Continuous action outputs can get small, thus exp and ln workaround employed
        policy_ratio = torch.exp(torch.log(self(states)) - torch.log(past_policy(states)))
        advantage_estimation = rewards + critic(next_states) * discount_factor
        
        surr_obj = torch.minimum(policy_ratio * advantage_estimation, 
                             torch.clip(policy_ratio, 
                                        1 - clipping_parameter, 
                                        1 + clipping_parameter) * advantage_estimation)
        return -torch.mean(surr_obj)
    
    
    def train(self, 
              critic, 
              rewards: torch.Tensor, 
              states: torch.Tensor, 
              next_states: torch.Tensor, 
              past_policy, 
              discount_factor: float = 0.99, 
              clipping_parameter: float = 0.2,
        ) -> None:
        
        self.optim.zero_grad()
        # MSE loss between critic and bootstrapped value
        loss = self.surrogate_objective(critic=critic, 
                                        rewards=rewards, 
                                        states=states, 
                                        next_states=next_states, 
                                        past_policy=past_policy, 
                                        discount_factor=discount_factor, 
                                        clipping_parameter=clipping_parameter,)
        loss.backward()
        self.optim.step()
    
    
class Critic():
    def __init__(self) -> None:
        super(Actor).__init__()
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.parameters())
        self.lstm = nn.LSTM(input_size=0, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Tanh(32, 1)
        
        
    def forward(self, 
                state: torch.Tensor,
        ) -> torch.Tensor:
        value, _ = self.lstm(state)[:, -1, :]
        value = self.fc(value)
        return value
    
    
    def train(self, 
              rewards: torch.Tensor, 
              states: torch.Tensor, 
              next_states: torch.Tensor, 
              discount_factor: float = 0.99,
        ) -> None:
        self.optim.zero_grad()
        # Compute values of t and t+1 states
        action_score, next_action_score = self(states), self(next_states)
        # MSE loss between critic and bootstrapped value
        loss = self.criterion(action_score, rewards + discount_factor * next_action_score)
        loss.backward()
        self.optim.step()