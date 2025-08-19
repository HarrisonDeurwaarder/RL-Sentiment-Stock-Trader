import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class Actor(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Linear(32, 2)
        
        self.optim = optim.Adam(self.parameters())
        
    # Overridden __call__ for IDE autocomplete
    def __call__(self,
                 state: torch.Tensor,) -> Tuple[torch.Tensor, float]:
        return super().__call__(state)
        
        
    def forward(self, 
                state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        _, (hidden, _) = self.lstm(state)
        out = self.fc(hidden[-1])
        # Return a sampled action and the distribution
        mean, log_std = torch.chunk(out, chunks=2, dim=-1)
        dist = torch.distributions.Normal(mean.squeeze(-1), torch.exp(log_std.squeeze(-1)))
        return dist.sample(), dist
    
    
    def surrogate_objective(self, 
                            critic: nn,
                            rewards: torch.Tensor, 
                            states: torch.Tensor,
                            old_actions: torch.Tensor,
                            old_log_probs: torch.Tensor,
                            discount_factor: float = 0.99,
                            clipping_parameter: float = 0.2,) -> torch.Tensor:
        
        # For non-discrete action space; Compute the ratio between the current and past policies
        # Continuous action outputs can get small, thus exp and ln workaround employed
        policy_ratio = torch.exp(self(states)[1].log_prob(old_actions) - old_log_probs)
        advantage_estimation = rewards + critic(states) * discount_factor
        
        surr_obj = torch.minimum(advantage_estimation * policy_ratio, 
                             torch.clip(policy_ratio, 
                                        1 - clipping_parameter, 
                                        1 + clipping_parameter) * advantage_estimation)
        return -torch.mean(surr_obj)
    
    
    def train(self, 
              critic, 
              rewards: torch.Tensor, 
              states: torch.Tensor,
              old_actions: torch.Tensor,
              old_log_probs: torch.Tensor,
              discount_factor: float = 0.99, 
              clipping_parameter: float = 0.2,
        ) -> None:
        
        self.optim.zero_grad()
        # MSE loss between critic and bootstrapped value
        loss = self.surrogate_objective(critic=critic, 
                                        rewards=rewards, 
                                        states=states,
                                        old_actions=old_actions,
                                        old_log_probs=old_log_probs,
                                        discount_factor=discount_factor, 
                                        clipping_parameter=clipping_parameter,)
        loss.backward()
        self.optim.step()
    
    
class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=32, num_layers=2, dropout=0.1)
        self.fc = nn.Linear(32, 1)
        
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.parameters())
        
        
    def forward(self, 
                state: torch.Tensor,) -> torch.Tensor:
        _, (hidden, _) = self.lstm(state)
        out = self.fc(hidden[-1])
        return out.squeeze(-1)
    
    
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