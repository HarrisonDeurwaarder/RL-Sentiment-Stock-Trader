from environment import *
from model import *
from utils import *
from data import *
from torch.utils.data.dataloader import DataLoader
import torch

# Ensures the agent is adaptable to the start amount
dist = torch.distributions.Uniform(low=100, high=1000)
DISCOUNT_FACTOR = 0.99
CLIPPING_PARAM = 0.2
EPOCHS = 4
HORIZON = 2048
BATCH = 64
MEMORY = 20
assert HORIZON % BATCH == 0


def main() -> None:
    actor = Actor()
    critic = Critic()
        
    # No explicit iteration count; every ticker will be exhausted
    for ticker in tickers:

        # Assemble the environment and define the rollout indices
        env = Environment(ticker=ticker, total_capital=dist.sample().item())
        # Until episode is over (no more rollouts can be extracted)
        ep_over = False
        while not ep_over:
            
            # To be converted to a torch Dataset for training
            states = []
            actions = []
            rewards = []
            
            states.append(torch.from_numpy(env.reset().to_numpy()).float())
            
            # Until rollout queue is exhausted
            rollout_over = False
            while not rollout_over:
                # Compute rollout
                
                action, dist_params = actor(states[-1])
                next_state, reward, rollout_over = env.step(action=action.item(),
                                              max_memory=MEMORY,)
                states.append(torch.from_numpy(next_state.values).float())
                actions.append(dist_params.values)
                rewards.append(reward)
                
            data = Rollout(states=states,
                           actions=actions,
                           rewards=rewards,)
            data = DataLoader(data, batch_size=BATCH, shuffle=True)
            
            # Training loop
            for _ in range(EPOCHS):
                for batch in data:
                    actor.train(critic=critic,
                                rewards=batch['rewards'],
                                states=batch['states'],
                                old_actions=batch['actions'],
                                discount_factor=DISCOUNT_FACTOR,
                                clipping_parameter=CLIPPING_PARAM,)
                    critic.train(rewards=batch['rewards'],
                                states=batch['states'],
                                next_states=batch['next_states'],
                                discount_factor=DISCOUNT_FACTOR)
            
                
            # Verifies that more rollouts can be extracted
            ep_over = env.next_rollout()
                


if __name__ == '__main__':
    main()