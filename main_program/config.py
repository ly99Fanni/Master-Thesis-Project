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
        num_steps: int = 500,
        batch_size: int = 200,
        lookback_window_size: int = 30,
        agent4_window_size: int = 60,
        num_episodes = 1000,
        num_evaluation_episodes=10,
        device = DEVICE,
        agent_type: int = 1, # agent types: 1,2,3,4


        #### Price dynamics ####
        S_0: float = 120.0,
        S_mean: float = 100.0,
        sigma: float = 0.15,
        sigma_mean: float = 0.05,
        kappa: float = 1.0,
        # kappa_mean: float = 0.10,
        mu_mean = 0.03,
        T: int = 52,


        #### Reward parameters ####
        gamma: float = 0.999, # discount factor
        half_bidask: float = 0.0003,
        risk_aversion: float = 0.2,
        threshold:float =0.1,

        #### Agent learning parameters ####
        # epsilon-greedy policy
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.005,
        epsilon_decay: float = 0.98,

        # Softmax policy
        use_softmax: bool = True,
        temperature: float = 1.0,
        temperature_min: float = 0.005,
        temperature_decay: float = 0.99,

        #### Replay Buffer parameters ####
        replay_capacity: int = 50000,
        replay_batch_size: int = 64,
        replay_start_size:int = 1000,

        ### Model ###
        DuelingDQN: bool = True,
        GDQN: bool = False,
        hidden_size: int = 64,

        ### Plot agent action strategy
        plot_action_strategy = False,

        ### Prioritized Replay Buffer
        per_alpha = 0.6,
        per_beta_start=0.4,
        per_beta_end = 1.0,
        per_epsilon = 1e-6,
        replay_type = "prioritized", # "prioritized"
        train_frequency= 30,
        target_update_frequency = 500,
        max_grad_norm = 10.0



    ):
        # Basic settings
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.num_evaluation_episodes = num_evaluation_episodes
        self.device = device
        self.agent_type = agent_type

        if self.agent_type == 4:
            self.window_size = agent4_window_size
        else:
            self.window_size = lookback_window_size

        # Price dynamics
        self.S_0 = S_0
        self.S_mean = S_mean
        self.sigma = sigma
        self.sigma_mean = sigma_mean
        self.kappa = kappa
        self.mu_mean = mu_mean
        self.T = T
        # self.kappa_mean = kappa_mean

        # Reward
        self.gamma = gamma
        self.half_bidask= half_bidask
        self.risk_aversion = risk_aversion
        # self.threshold = threshold

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
        self.replay_capacity = replay_capacity
        self.replay_batch_size = replay_batch_size
        self.replay_start_size = replay_start_size

        # Model
        self.use_DuelingDQN = DuelingDQN
        self.use_GDQN = GDQN
        self.hidden_size = hidden_size

        self.plot_action_strategy = plot_action_strategy

        # Prioritized Replay Buffer
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_epsilon = per_epsilon

        self.replay_type = replay_type
        # options:
        # "none"
        # "uniform"
        # "prioritized"

        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.max_grad_norm = max_grad_norm
        self.per_beta_end = per_beta_end