import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_to_pos(action):
    """Convert action [0=sell, 1=hold, 2=buy] to position [-1, 0, 1]"""
    pos = action - 1
    return pos.to(dtype=torch.float32, device=DEVICE)

def pos_to_action(pos):
    action = pos + 1
    return action.to(dtype=torch.float32, device=DEVICE)

class Config:

    def __init__(self,

        #### Basic settings ####
        num_actions: int = 3,
        num_steps: int = 200,
        batch_size: int = 300,
        lookback_window_size: int = 1,
        num_episodes = 300,
        num_evaluation_episodes=100,
        device = DEVICE,

        #### Price dynamics ####
        S_0: float = 120.0,
        S_mean: float = 100.0,
        sigma: float = 0.1,
        sigma_mean: float = 0.05,
        kappa: float = 2.0,
        mu_mean = 0.08,
        T: int = 12,

        #### Reward parameters ####
        gamma: float = 0.98, # discount factor
        half_bidask: float = 0.001,
        risk_aversion: float = 0.02,
        threshold:float =0.1,

        #### Agent learning parameters ####
        # epsilon-greedy policy
        learning_rate: float = 0.0003,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,

        # Softmax policy
        use_softmax: bool = True,
        temperature: float = 1.0,
        temperature_min: float = 0.05,
        temperature_decay: float = 0.995,

        #### Replay Buffer parameters ####
        buffer_is_on: bool = True,
        replay_capacity: int = 50000,
        replay_batch_size: int = 128,
        replay_start_size:int = 1000,

        ### Model ###
        DuelingDQN: bool = False,
        GDQN: bool = True,
        hidden_size: int = 64,


    ):
        # Basic settings
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.window_size = lookback_window_size
        self.num_episodes = num_episodes
        self.num_evaluation_episodes = num_evaluation_episodes
        self.device = device

        # Price dynamics
        self.S_0 = S_0
        self.S_mean = S_mean
        self.sigma = sigma
        self.sigma_mean = sigma_mean
        self.kappa = kappa
        self.mu_mean = mu_mean
        self.T = T

        # Reward
        self.gamma = gamma
        self.half_bidask= half_bidask
        self.risk_aversion = risk_aversion
        self.threshold = threshold

        # Agent - epsilon-greedy exploration
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Agent - softmax policy
        self.use_softmax = use_softmax
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

        # Replay Buffer
        self.buffer_is_on = buffer_is_on
        self.replay_capacity = replay_capacity
        self.replay_batch_size = replay_batch_size
        self.replay_start_size = replay_start_size

        # Model
        self.use_DuelingDQN = DuelingDQN
        self.use_GDQN = GDQN
        self.hidden_size = hidden_size
