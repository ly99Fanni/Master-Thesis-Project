from dataclasses import dataclass
from enum import IntEnum
from torch import Tensor
from typing import Dict, List, Tuple, Optional



class Action(IntEnum):
    SHORT = 0
    OUT = 1
    BUY = 2


@dataclass
class MarketConfig:
    S0: float = 120.0
    S_mean: float = 100.0
    Kappa: int = 2
    sigma: float = 0.3
    T: int = 12
    mean_mu: float = 0.08
    mean_sigma: float = 0.05

    batch_size: int = 200
    num_episode: int = 3

    gamma: float = 0.99
    half_ba: float = 1.0
    risk_av: float = 0.02




