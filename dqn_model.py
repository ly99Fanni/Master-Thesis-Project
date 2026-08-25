import torch
import torch.nn as nn
from config import Config


class SimpleDQN(nn.Module):
    def __init__(self, state_size, num_actions, config: Config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(state_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, num_actions)
        )

    def forward(self, state):
        return self.network(state)



class DuelingDQN(nn.Module):

    def __init__(self, state_size, num_actions, config: Config):
        super().__init__()

        self.config = config
        self.feature = nn.Sequential(
            nn.Linear(state_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU()
        )

        # State value = V(s)
        self.value = nn.Linear(self.config.hidden_size, 1)

        # Advantage of each action = A(s,a)
        self.advantage = nn.Linear(
            self.config.hidden_size,
            num_actions
        )

    def forward(self, state):

        features = self.feature(state)

        value = self.value(features)

        advantage = self.advantage(features)

        q_values = (
            value
            + advantage
            - advantage.mean(
                dim=1,
                keepdim=True
            )
        )

        return q_values



class GDQN(nn.Module):

    def __init__(self,input_size,num_actions,hidden_size=64,num_gru_layers=1,):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        self.q_network = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state):

        # state:
        # (batch_size, window_size, input_size)

        gru_output, hidden = self.gru(state)

        # Take representation of the final timestep
        last_output = gru_output[:, -1, :]

        # Q-values
        q_values = self.q_network(last_output)

        return q_values