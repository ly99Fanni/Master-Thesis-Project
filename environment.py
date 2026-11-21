import matplotlib.pyplot as plt
import torch
from torch import Tensor
from config import MarketConfig
from matplotlib.animation import FuncAnimation

torch.manual_seed(42)

class MarketEnv():
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.S0 = torch.tensor(self.cfg.S0, dtype=torch.float32)
        self.X0 = torch.log(self.S0)
        self.S_mean = torch.tensor(self.cfg.S_mean, dtype=torch.float32)
        self.X_mean = torch.log(self.S_mean)
        self.dt = 1/self.cfg.T
        self.cum_reward = 0
        self.B = self.cfg.batch_size
        self.half_ba = self.cfg.half_ba
        self.risk_av = self.cfg.risk_av
        self.sigma = self.cfg.sigma
        self.kappa = self.cfg.Kappa
        self.mean_mu = self.cfg.mean_mu
        self.mean_sigma = self.cfg.mean_sigma


        # === Initial state ===
        self.simulated_price_seq, self.simulated_price_mean_seq = self._simulate_price_seq()
        self.pos = torch.zeros(self.B, dtype=torch.float32)

        self.print_env()

    def _simulate_next_price(self, current_price_mean: Tensor, current_price: Tensor) -> Tensor:
        '''
        Calculate the next price based on Ornestein-Uhlenbeck ( or Vasicek) model
        dS = Kappa * (S_mean - S_{t-1}) * dt + sigma * dWt  --> drift term + random term
        next price = S_t = S_{t-1} + dS

        :return: next price
        '''
        drift = self.kappa * (current_price_mean - current_price) * self.dt
        random = self.sigma * torch.randn(1) * torch.sqrt(torch.tensor(self.dt, dtype=torch.float32))
        next_price = current_price + drift + random
        return next_price

    def _next_price_mean(self, current_price_mean):
        '''
        The price mean is not constant over time.
        Calculate the next price mean based on Arithmetic Brownian Motion:
        dS_mean = Mu * dt + sigma_mean * dW_mean
        next S_mean = S_mean_{t-1} + dS_mean

        :return: next S_mean
        '''
        dW = torch.randn(1) * torch.sqrt(torch.tensor(self.dt, dtype=torch.float32))
        next_mean = current_price_mean + self.mean_sigma * dW + self.mean_mu * self.dt

        return next_mean


    def _simulate_price_seq(self) -> tuple[Tensor, Tensor]:
        '''
        Simulate the prices and the price means for B times

        :return: A sequence of B price values and a corresponding sequence of B mean price values.
        '''

        price_seq = torch.ones((self.B,), dtype=torch.float32) * self.X0
        price_mean_seq = torch.ones((self.B,), dtype=torch.float32) * self.X_mean
        for t in range(self.B - 1):
            next_price = self._simulate_next_price(price_mean_seq[t],price_seq[t])
            price_seq[t+1] = next_price
            price_mean_seq[t + 1] = self._next_price_mean(price_mean_seq[t])

        return price_seq, price_mean_seq




    def step(self, current_price: Tensor, next_price: Tensor, prev_pos: Tensor, action: Tensor) -> tuple[float, float, float]:
        '''
        Calculate the reward based on the action taken in the previous state.
        Reward is calculated as following:
        R_t = Profit - Cost - Risk term

        :return: Reward
        '''
        pos = (action - 1).item()
        next_price = torch.exp(next_price)
        diff_price = next_price - current_price
        reward = pos * diff_price
        reward -= self.half_ba * torch.abs(pos - prev_pos)
        reward -= self.risk_av * ((self.sigma*pos)**2)

        return next_price, pos, reward


    def simulate(self, action: Tensor, current_price: Tensor, simulated_price_seq: Tensor, n=3, print_data: bool = False):
        '''
        Total reward accumulated over B time steps

        :return: total cumulative reward
        '''

        for t in range(self.B - 1):

            if print_data and t < n:
                print('[Step {:02d}]'.format(t))
                print('Current price: {:.2f}'.format(current_price))
                print('Action to take: {}'.format(action[t].item()))

            next_price, new_pos, reward = self.step(
                current_price,
                simulated_price_seq[t + 1],
                self.pos[t],
                action[t]
            )

            # update state
            current_price = next_price
            self.cum_reward += reward
            self.pos[t + 1] = new_pos

            # only print first n steps
            if print_data and t < n:
                print('Reward: {:.2f}'.format(reward))
                print('Total reward: {:.2f}'.format(self.cum_reward))
                print('-' * 60)


        if print_data:
            print('\nFinal cumulative reward: {:.2f}'.format(self.cum_reward))

        return self.cum_reward

    def reset(self, restart=True):
        self.pos = torch.zeros(self.B, dtype=torch.float32)
        self.cum_reward = 0

        if restart:
            self.simulated_price_seq, self.simulated_price_mean_seq = self._simulate_price_seq()


    def simulate_episode(self, action: Tensor, num_episodes = 5, print_data: bool=False):
        cum_reward_seq = torch.zeros((num_episodes,), dtype=torch.float32)

        for ep in range(num_episodes):
            self.reset()
            cum_reward = self.simulate(action, self.S0, self.simulated_price_seq)
            cum_reward_seq[ep] = cum_reward

        if print_data:
            print(f'Cumulative returns for {num_episodes} episodes')
            print(cum_reward_seq)

        return cum_reward_seq



    def print_env(self):
        print('=' * 60)
        print('Environment Setup')
        print(f'vol = {self.sigma}, kappa = {self.kappa}, mean = {self.S_mean}')
        print('=' * 60)

        print('Print the first 10 simulated prices and means')
        print(torch.exp(self.simulated_price_seq)[:10])
        print(torch.exp(self.simulated_price_mean_seq)[:10])

        print('=' * 60)
        print('')

    def plot_price_mean(self, price_seq: Tensor, price_mean: Tensor,
                        label1: str = 'Market value',
                        label2: str = 'Fundamental value',
                        title: str = 'Growing mean-reverting price dynamic'):

        fig = plt.figure(figsize=(10, 6))
        plt.plot(torch.exp(price_seq).squeeze().numpy(), label=label1)
        plt.plot(torch.exp(price_mean).squeeze().numpy(), label=label2)
        plt.xlabel('t')
        plt.ylabel('St')
        plt.title(title)
        plt.legend()
        # plt.ylim(bottom=4.0)
        plt.show()

    def plot_animation_price_mean(self, price_seq: Tensor, price_mean: Tensor,
                                  label1: str = 'Market value',
                                  label2: str = 'Fundamental value',
                                  title: str = 'Growing mean-reverting price dynamic'):

        y1 = torch.exp(price_seq)
        y2 = torch.exp(price_mean)
        fig, ax = plt.subplots()

        line1, = ax.plot([], [], label=label1)
        line2, = ax.plot([], [], label=label2)

        ax.set_xlim(0, self.B)
        ax.set_ylim(min(y1.min(), y2.min()) - 1, max(y1.max(), y2.max()) + 1)
        ax.axhline(self.S_mean, linestyle = '--', color='red')
        ax.set_title(title)

        def update(frame):
            x = range(frame)

            line1.set_data(x, y1[:frame])
            line2.set_data(x, y2[:frame])

            return line1, line2

        ani = FuncAnimation(fig, update, frames=self.B, interval=100, blit=True)

        plt.show()







if __name__ == "__main__":
    cfg = MarketConfig()
    env = MarketEnv(cfg)

    ## plot price dynamic ##
    # env.plot_price_mean(env.simulated_price_seq, env.simulated_price_mean_seq)
    # env.plot_animation_price_mean(env.simulated_price_seq, env.simulated_price_mean_seq)

    ## Actions ##
    actions1 = torch.randint(0, 3, (cfg.batch_size,))
    # actions1 = 0 * torch.ones(cfg.batch_size, dtype=torch.long) # only short
    # actions2 = torch.randint(0, 3, (cfg.batch_size,))
    # actions2 = 2 * torch.ones(cfg.batch_size, dtype=torch.long) # only long

    ## Comparing agents decision-making when the fundamental knowledge is known or not ##
    # env.simulate(actions2, env.S0, env.simulated_price_seq, n= 5, print_data=True)
    # print('='*60)
    # env.reset(restart=False)
    # env.simulate(actions2, env.S_mean, env.simulated_price_mean_seq, n= 5, print_data=True)

    ## Simulate the cumulative rewards for n episodes ##
    env.simulate_episode(actions1, print_data=True, num_episodes=10)









