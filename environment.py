import numpy as np
import torch
from torch import Tensor
from config import Config, action_to_pos, pos_to_action
import matplotlib.pyplot as plt
import pandas as pd

class Environment:
    def __init__(self, config: Config):
        ## Setting parameters
        self.config = config
        self.batch_size = self.config.batch_size
        self.num_steps = self.config.num_steps
        self.window_size = self.config.window_size
        self.device = self.config.device

        # Price
        self.S_0 = self.config.S_0
        self.S_mean = self.config.S_mean
        self.sigma = self.config.sigma
        self.sigma_mean = self.config.sigma_mean
        self.kappa = self.config.kappa
        self.mu_mean = self.config.mu_mean
        self.dt = 1 / self.config.T

        # log price
        self.X = self._convert_to_logarithm(self.S_0)
        self.X_mean = self._convert_to_logarithm(self.S_mean)

        # reward
        self.total_reward = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.reward_seq = torch.zeros((self.batch_size, self.num_steps - 1), dtype=torch.float32, device=self.device)
        self.gamma = self.config.gamma
        self.half_bidask = self.config.half_bidask
        self.risk_aversion = self.config.risk_aversion
        self.last_profit = torch.zeros(self.batch_size, device=self.device)
        self.last_transaction_cost = torch.zeros(self.batch_size, device=self.device)
        self.last_risk_penalty = torch.zeros(self.batch_size, device=self.device)
        self.last_turnover = torch.zeros(self.batch_size, device=self.device)


        # Initialize
        self.price_seq, self.mean_price_seq = self._simulate_price_sequence()
        self.pos = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)




    def _convert_to_logarithm(self, S):
        return np.log(S)

    def _simulate_next_price(self, price, price_mean, volatility, mean = False):
        if mean:
            drift = self.mu_mean * self.dt
        else:
            drift = self.kappa * (price_mean - price) * self.dt


        dW = torch.randn(self.batch_size, device=self.device) * (self.dt **0.5)
        noise = volatility * dW

        next_price = price + drift + noise

        return next_price

    def _simulate_price_sequence(self,):
        log_price_seq = torch.ones((self.batch_size, self.num_steps), dtype=torch.float32, device=self.device) * self.X
        log_mean_price_seq = torch.ones((self.batch_size, self.num_steps), dtype=torch.float32, device=self.device) * self.X_mean

        for t in range(self.num_steps - 1):
            log_mean_price_seq[:, t + 1] = self._simulate_next_price(log_mean_price_seq[:, t], self.mu_mean, self.sigma_mean, mean=True)
            log_price_seq[:, t + 1] = self._simulate_next_price(log_price_seq[:, t], log_mean_price_seq[:,t], self.sigma)

        return torch.exp(log_price_seq), torch.exp(log_mean_price_seq)

    def get_state(self):
        """Return current price"""
        return(
            self.price_seq.detach().clone(),
        )

    def reset(self):
        self.price_seq, self.mean_price_seq = self._simulate_price_sequence()
        self.total_reward = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.reward_seq = torch.zeros((self.batch_size, self.num_steps - 1), dtype=torch.float32, device=self.device)
        self.last_profit = torch.zeros(self.batch_size, device=self.device)
        self.last_transaction_cost = torch.zeros(self.batch_size, device=self.device)
        self.last_risk_penalty = torch.zeros(self.batch_size, device=self.device)
        self.last_turnover = torch.zeros(self.batch_size, device=self.device)

    def step(self, action, t):
        old_pos = self.pos.clone()
        new_pos = action_to_pos(action)
        turnover = torch.abs(new_pos - old_pos)

        current_price = self.price_seq[:, t]
        next_price = self.price_seq[:, t + 1]

        profit = new_pos *  (next_price   - current_price)
        transaction_cost = self.half_bidask * current_price * torch.abs(new_pos - self.pos)
        risk_penalty = self.risk_aversion * (self.sigma * new_pos) ** 2
        reward = profit - transaction_cost - risk_penalty

        self.last_profit = profit.detach()
        self.last_transaction_cost = transaction_cost.detach()
        self.last_risk_penalty = risk_penalty.detach()
        self.last_turnover = turnover.detach()

        self.reward_seq[:, t] = reward.detach()
        self.total_reward += reward
        self.pos.copy_(new_pos)
        new_state = (next_price.detach().clone(), new_pos.detach().clone())

        done = torch.full(
            (self.batch_size,),
            t == self.num_steps - 2,
            dtype=torch.bool,
            device=self.device
        )

        return new_state, reward.detach().clone(), done


    # def build_state(self,price,mean_price,pos):
    #     batch_size = price.shape[0]
    #     price_window = price[:, -self.window_size:]
    #     mean_window = mean_price[:, -self.window_size:]
    #
    #     relative_diff = (price_window - mean_window) / mean_window
    #
    #     price_momentum = (price_window[:, 1:] - price_window[:, :-1]) / price_window[:, :-1]
    #
    #     mean_trend = (mean_window[:, 1:] - mean_window[:, :-1]) / mean_window[:, :-1]
    #
    #     position = pos.reshape(batch_size, 1)
    #
    #     return torch.cat((relative_diff,mean_trend,price_momentum,position),dim=1)

    def build_state(self,price,mean_price,position):
        """
        price:      (batch_size, window_size)
        mean_price: (batch_size, window_size)
        position:   (batch_size,)

        recurrent=False:
            Simple DQN
            returns (batch_size, window_size * num_features)

        recurrent=True:
            GDQN
            returns (batch_size, window_size, num_features)
        """

        # -----------------------------------------
        # 1. Relative difference
        # -----------------------------------------

        relative_diff = (price - mean_price) / mean_price

        # -----------------------------------------
        # 2. Price momentum
        # -----------------------------------------

        price_momentum = torch.zeros_like(price)

        price_momentum[:, 1:] = (price[:, 1:] - price[:, :-1]) / price[:, :-1]


        # -----------------------------------------
        # 3. Mean trend
        # -----------------------------------------

        mean_trend = torch.zeros_like(mean_price)

        mean_trend[:, 1:] = (mean_price[:, 1:] - mean_price[:, :-1]) / mean_price[:, :-1]

        # shape:
        # (batch_size, window_size)

        # -----------------------------------------
        # 4. Position
        # -----------------------------------------

        position_feature = position.reshape(-1, 1).expand(-1,price.shape[1])


        # -----------------------------------------
        # 5. Create temporal state
        # -----------------------------------------

        state = torch.stack((relative_diff,mean_trend,price_momentum,position_feature),dim=2)


        # -----------------------------------------
        # GDQN
        # -----------------------------------------

        if self.config.use_GDQN:
            return state

        # -----------------------------------------
        # Simple DQN
        # -----------------------------------------

        return state.flatten(start_dim=1)


    def print_data(self, num_batch_size=3, num_time_steps=10):
        num_batches = max(0, min(num_batch_size, self.batch_size))
        num_steps = max(0, min(num_time_steps, self.num_steps))

        action_names = {
            0: "SELL",
            1: "HOLD",
            2: "BUY"
        }

        state_names = {
            0: "Below Mean",
            1: "Near Mean",
            2: "Above Mean"
        }

        batch_rows = [[] for _ in range(num_batches)]
        total_reward = torch.zeros(num_batches, dtype=torch.float32, device=self.device)

        for batch in range(num_batches):
            batch_rows[batch].append({
                "Step": 0,
                "Price": self.price_seq[batch, 0].item(),
                "Mean": self.mean_price_seq[batch, 0].item(),
                "Action": "-",
                "Position": int(self.pos[batch].item()),
                "Reward": 0.0,
                "Cumulative Reward": 0.0,
                "Done": False
            })

        for t in range(num_steps):

            actions = torch.randint(
                low=0,
                high=self.config.num_actions,
                size=(self.batch_size,),
                device=self.device
            )

            state, reward, done = self.step(actions, t)

            prices = state[0]
            positions = state[1]

            total_reward += reward[:num_batches]

            for batch in range(num_batches):
                mean = self.mean_price_seq[batch, t].item()


                batch_rows[batch].append({
                    "Step": t +1 ,
                    "Price": prices[batch].item(),
                    "Mean": mean,
                    "Action": action_names[int(actions[batch].item())],
                    "Position": int(positions[batch].item()),
                    "Reward": reward[batch].item(),
                    "Cumulative Reward": total_reward[batch].item(),
                    "Done": done[batch].item()
                })

            if done.all():
                break

        for batch in range(num_batches):
            df = pd.DataFrame(batch_rows[batch])

            print(f"\nBatch {batch + 1}\n")

            print(
                df.to_string(
                    index=False,
                    formatters={
                        "Price": "{:.2f}".format,
                        "Mean": "{:.2f}".format,
                        "Reward": "{:.2f}".format,
                        "Cumulative Reward": "{:.2f}".format
                    }
                )
            )


    def print_env(self):
        self.reset()
        parameters = {
            "Batch size": self.batch_size,
            "Number of steps": self.num_steps,
            "S0": self.S_0,
            "S_mean": self.S_mean,
            "T": self.config.T,
            "Kappa": self.kappa,
            "Mean mu": self.mu_mean,
            "Volatility": self.sigma,
            "Mean Volatility": self.sigma_mean,
            "Device": self.device,
        }

        print("\nEnvironment Setup - Parameters")
        print("-" * 40)
        print(f"{'Parameter':<20} {'Value':>15}")
        print("-" * 40)

        for name, value in parameters.items():
            if isinstance(value, float):
                value = f"{value:.4f}"

            print(f"{name:<20} {str(value):>15}")

        # print("-" * 40)
        # print('')
        # print("Simulated price sequence")
        # print(env.get_state())
        print("-" * 70)

    def plot_price(self, num_batch=1, show_mean=True):
        price_seq = self.price_seq
        mean_price_seq = self.mean_price_seq

        m = max(0, min(num_batch, self.batch_size))

        x = np.arange(price_seq.shape[1]) * self.dt

        plt.figure(figsize=(10, 6))

        for i in range(m):
            plt.plot(
                x,
                price_seq[i].detach().cpu().numpy(),
                label=f"Price {i + 1}"
            )

            if show_mean:
                plt.plot(
                    x,
                    mean_price_seq[i].detach().cpu().numpy(),
                    linestyle="--",
                    label=f"Mean {i + 1}"
                )

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Simulated Price Paths")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_reward(self, num_batch=5):
        num_batches = max(0, min(num_batch, self.batch_size))
        cumulative_reward = torch.cumsum(self.reward_seq[:num_batches],dim=1)
        x = np.arange(cumulative_reward.shape[1])

        plt.figure(figsize=(10, 6))

        for batch in range(num_batches):
            plt.plot(x, cumulative_reward[batch].detach().cpu().numpy(), label=f"Batch {batch + 1}")

        plt.axhline(0, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.title("Reward Over Time")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


#### SANITY CHECK ####
if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    # env.print_env()
    env.print_data(num_time_steps=10, num_batch_size=2)
    # env.plot_price()
    # env.plot_reward()
    # print('Total rewards for each batch size: ', env.total_reward)
    state = env.build_state(env.price_seq, env.mean_price_seq, env.pos)
    print(state.shape)