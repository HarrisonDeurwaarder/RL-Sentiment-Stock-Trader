import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class Actor(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(32, 2),
                                nn.Tanh(),)
        
        self.optim = optim.Adam(self.parameters())
        
    # Overridden __call__ for IDE autocomplete
    def __call__(self,
                 state: torch.Tensor,) -> Tuple[torch.Tensor, float]:
        return super().__call__(state)
        
        
    def forward(self, 
                state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        _, (hidden, _) = self.lstm(state)
        output = self.fc(hidden[-1])
        # Return a sampled action and the distribution params
        return torch.normal(output[0], torch.exp(output[1])), output
    
    
    def surrogate_objective(self, 
                            critic: nn,
                            rewards: torch.Tensor, 
                            states: torch.Tensor,
                            old_actions: torch.Tensor,
                            discount_factor: float = 0.99,
                            clipping_parameter: float = 0.2,) -> torch.Tensor:
        
        # For non-discrete action space; Compute the ratio between the current and past policies
        # Continuous action outputs can get small, thus exp and ln workaround employed
        policy_ratio = torch.exp(torch.log(self(states)) - torch.log(old_actions))
        advantage_estimation = rewards + critic(states) * discount_factor
        
        surr_obj = torch.minimum(policy_ratio * advantage_estimation, 
                             torch.clip(policy_ratio, 
                                        1 - clipping_parameter, 
                                        1 + clipping_parameter) * advantage_estimation)
        return -torch.mean(surr_obj)
    
    
    def train(self, 
              critic, 
              rewards: torch.Tensor, 
              states: torch.Tensor,
              old_actions: torch.Tensor,
              discount_factor: float = 0.99, 
              clipping_parameter: float = 0.2,
        ) -> None:
        
        self.optim.zero_grad()
        # MSE loss between critic and bootstrapped value
        loss = self.surrogate_objective(critic=critic, 
                                        rewards=rewards, 
                                        states=states,
                                        old_actions=old_actions,
                                        discount_factor=discount_factor, 
                                        clipping_parameter=clipping_parameter,)
        loss.backward()
        self.optim.step()
    
    
class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(32, 2),
                                nn.Tanh(),)
        
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.parameters())
        
        
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