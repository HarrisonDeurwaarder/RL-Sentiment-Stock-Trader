import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.module):
    def __init__(self) -> None:
        self.optim = optim.Adam(self.parameters())
        self.net = nn.Sequential(
            nn.Linear(5),
            nn.Linear(10),
            nn.Linear(15),
            nn.Linear(8),
            nn.Linear(2),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor, sample_action: bool = False) -> torch.Tensor:
        output = self.net(state)
        # For inference, directly sample stochasically on a normal distribution using mean and stddev
        if sample_action:
            return torch.normal(output[0], output[1])
        # For training, return parameters directly
        return output
    
    def surrogate_objective(self, states: torch.Tensor, past_policy) -> torch.Tensor:
        # For non-discrete action space; Compute the ratio between the current and past policies
        # Continuous action outputs can get small, thus exp and ln workaround employed
        ratio_mean = torch.exp(torch.log(self(states)) - torch.log(past_policy(states)))
    
    def train(self, states: torch.Tensor, next_states: torch.Tensor, discount_factor: float) -> None:
        self.optim.zero_grad()
        # Compute values of t and t+1 states
        outs, next_outs = self(states), self(next_states)
        # MSE loss between critic and bootstrapped value
        loss = self.criterion(outs, value + discount_factor * next_outs)
        loss.backward()
        self.optim.step()
    
    
class Critic():
    def __init__(self) -> None:
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.parameters())
        self.net = nn.Sequential(
            nn.Linear(5),
            nn.Linear(10),
            nn.Linear(15),
            nn.Linear(8),
            nn.Linear(2)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
    
    def train(self, rewards: torch.Tensor, states: torch.Tensor, next_states: torch.Tensor, discount_factor: float = 0.99) -> None:
        self.optim.zero_grad()
        # Compute values of t and t+1 states
        action_score, next_action_score = self(states), self(next_states)
        # MSE loss between critic and bootstrapped value
        loss = self.criterion(action_score, rewards + discount_factor * next_action_score)
        loss.backward()
        self.optim.step()
        