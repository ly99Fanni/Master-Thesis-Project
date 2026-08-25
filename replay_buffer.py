import random
from collections import deque
from config import Config

import torch

class ReplayBuffer:

    def __init__(self, config: Config):
        self.config = config
        self.replay_batch_size = self.config.replay_batch_size
        self.device = self.config.device
        self.buffer = deque(maxlen=self.config.replay_capacity)

    def push(self, state, action, reward, next_state, done):

        batch_size = state.shape[0]

        for i in range(batch_size):
            self.buffer.append((
                state[i].detach().cpu(),
                action[i].detach().cpu(),
                reward[i].detach().cpu(),
                next_state[i].detach().cpu(),
                done[i].detach().cpu()
            ))

    def sample(self):
        transitions = random.sample(self.buffer, self.replay_batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).long().to(self.device)
        rewards = torch.stack(rewards).float().to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.stack(dones).bool().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)