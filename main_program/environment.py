import numpy as np
import torch
import matplotlib.pyplot as plt
from main_program.config import Config, action_to_pos


class Environment:

    def __init__(self, config: Config):

        self.config = config
        self.observe_mean = True

        # ==================================================
        # 0. Set the type of Agent
        # ==================================================
        self.agent_type = config.agent_type

        # ==================================================
        # 1. Basic dimensions
        # ==================================================

        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.window_size = config.window_size
        self.device = config.device

        self.dt = 1 / config.T

        # ==================================================
        # 2. Market parameters
        # ==================================================
        self.S_start = self.config.S_0
        self.S_mean_start = self.config.S_mean
        self.S_0 = config.S_0
        self.S_mean = config.S_mean

        self.sigma = config.sigma
        self.sigma_mean = config.sigma_mean

        self.kappa = config.kappa
        self.mu_mean = config.mu_mean

        # ==================================================
        # 3. Reward parameters
        # ==================================================

        self.gamma = config.gamma
        self.half_bidask = config.half_bidask
        self.risk_aversion = config.risk_aversion

        # ==================================================
        # 4. Persistent market state
        #
        # These values continue between episodes.
        # ==================================================

        self.log_price = torch.full(
            (self.batch_size,),
            float(np.log(self.S_0)),
            dtype=torch.float32,
            device=self.device
        )

        self.log_mean = torch.full(
            (self.batch_size,),
            float(np.log(self.S_mean)),
            dtype=torch.float32,
            device=self.device
        )


        # ==================================================
        # 5. Trading position
        # ==================================================

        self.pos = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        # ==================================================
        # 6. Episode data
        # ==================================================

        self.price_seq = None
        self.mean_price_seq = None

        self.total_reward = None
        self.reward_seq = None

        self.last_profit = None
        self.last_transaction_cost = None
        self.last_risk_penalty = None
        self.last_turnover = None

        # ==================================================
        # 7. Generate first episode
        # ==================================================

        self.reset()


    # ======================================================
    # MARKET SIMULATION
    # ======================================================

    def _simulate_price_sequence(self):

        log_price_seq = torch.zeros(
            (self.batch_size, self.num_steps),
            dtype=torch.float32,
            device=self.device
        )

        log_mean_seq = torch.zeros(
            (self.batch_size, self.num_steps),
            dtype=torch.float32,
            device=self.device
        )

        # Start from previous episode's final values
        log_price_seq[:, 0] = self.log_price
        log_mean_seq[:, 0] = self.log_mean

        for t in range(self.num_steps - 1):
            current_log_price = log_price_seq[:, t]
            current_log_mean = log_mean_seq[:, t]

            # ==================================================
            # Mean process
            # ==================================================

            dW_mean = (
                    torch.randn(
                        self.batch_size,
                        device=self.device
                    )
                    * np.sqrt(self.dt)
            )


            next_log_mean = (
                    current_log_mean
                    + self.mu_mean * self.dt
                    + self.sigma_mean * dW_mean
            )

            # ==================================================
            # Price process
            # ==================================================

            dW_price = (
                    torch.randn(
                        self.batch_size,
                        device=self.device
                    )
                    * np.sqrt(self.dt)
            )

            price_drift = (
                    self.kappa
                    * (
                            current_log_mean
                            - current_log_price
                    )
                    * self.dt
            )

            next_log_price = (
                    current_log_price
                    + price_drift
                    + self.sigma * dW_price
            )

            # ==================================================
            # Store
            # ==================================================

            log_price_seq[:, t + 1] = next_log_price
            log_mean_seq[:, t + 1] = next_log_mean


        # ======================================================
        # Save final values for next episode
        # ======================================================

        self.log_price = (
            log_price_seq[:, -1]
            .detach()
            .clone()
        )

        self.log_mean = (
            log_mean_seq[:, -1]
            .detach()
            .clone()
        )

        self.log_price_seq = (
            log_price_seq
            .detach()
            .clone()
        )

        self.log_mean_seq = (
            log_mean_seq
            .detach()
            .clone()
        )

        # ======================================================
        # Convert back to actual prices
        # ======================================================


        self.price_seq = torch.exp(
            self.log_price_seq.to(
                dtype=torch.float64
            )
        )

        self.mean_price_seq = torch.exp(
            self.log_mean_seq.to(
                dtype=torch.float64
            )
        )


        return (
            self.log_price_seq,
            self.log_mean_seq
        )


    # ======================================================
    # RESET EPISODE
    # ======================================================

    def reset(self):

        (
            self.log_price_seq,
            self.log_mean_seq
        ) = self._simulate_price_sequence()

        # Position continues

        self._reset_statistics()

        return (
            self.log_price_seq,
            self.log_mean_seq
        )
    # ======================================================
    # ENVIRONMENT STEP
    # ======================================================

    def step(
            self,
            action,
            t
    ):

        old_pos = self.pos.clone()

        new_pos = action_to_pos(
            action
        ).to(
            dtype=torch.float32,
            device=self.device
        )

        current_log_price = (
            self.log_price_seq[:, t]
        )

        next_log_price = (
            self.log_price_seq[:, t + 1]
        )

        log_return = (
                next_log_price
                - current_log_price
        )

        price_return = torch.expm1(
            log_return
        )

        profit = (
                new_pos
                * price_return
        )

        # ==================================================
        # Turnover
        # ==================================================

        turnover = torch.abs(
            new_pos
            - old_pos
        )

        # ==================================================
        # Transaction cost
        # ==================================================

        transaction_cost = (
                self.half_bidask
                * turnover
        )

        # ==================================================
        # Risk penalty
        # ==================================================

        risk_penalty = (
                self.risk_aversion
                * self.sigma ** 2
                * self.dt
                * new_pos ** 2
        )

        # ==================================================
        # Reward
        # ==================================================

        reward = (
                profit
                - transaction_cost
                - risk_penalty
        )

        # ==================================================
        # Store statistics
        # ==================================================

        self.last_profit = (
            profit.detach()
        )

        self.last_transaction_cost = (
            transaction_cost.detach()
        )

        self.last_risk_penalty = (
            risk_penalty.detach()
        )

        self.last_turnover = (
            turnover.detach()
        )

        self.reward_seq[:, t] = (
            reward.detach()
        )

        self.total_reward += (
            reward.detach()
        )

        # ==================================================
        # Update position
        # ==================================================

        self.pos.copy_(
            new_pos
        )
        # ==================================================
        # Continuous market:
        # end of chunk is NOT terminal
        # ==================================================

        done = torch.zeros(
            self.batch_size,
            dtype=torch.bool,
            device=self.device
        )

        return reward.detach(), done


    def build_state(
            self,
            price_window,
            pos,
            mean_window=None

    ):


        if self.agent_type == 1:

            return self.agent1_observation(
                price_window,
                pos,
            )

        elif self.agent_type == 2:

            return self.agent2_observation(
                price_window,
                pos,
                mean_window
            )

        elif self.agent_type == 3:

            return self.agent3_observation(
                price_window,
                pos,
                mean_window
            )

        elif self.agent_type == 4:
            return self.agent4_observation(price_window, pos)

        else:

            raise ValueError(
                f"Unknown agent_state: {self.agent_type}"
            )


    # ======================================================
    # BUILD STATE - Agent 1 - only observe the market prices
    # ======================================================

    def agent1_observation(
            self,
            log_price_window,
            pos
    ):

        # ==================================================
        # Current position
        # ==================================================

        position = (
            pos.to(
                dtype=log_price_window.dtype,
                device=log_price_window.device
            )
            .unsqueeze(1)
        )

        # ==================================================
        # window_size = 0
        #
        # Only current market price is observable.
        # A return cannot be calculated from one price.
        # ==================================================

        if self.window_size == 0:

            current_log_price = (
                log_price_window[:, -1:]
            )

            log_price_relative_to_start = (
                    current_log_price
                    - np.log(self.S_0)
            )

            state = torch.cat(
                [
                    log_price_relative_to_start,
                    position
                ],
                dim=1
            )

        # ==================================================
        # window_size > 0
        #
        # Use log returns over the lookback window.
        # ==================================================

        else:

            log_returns = (
                    log_price_window[:, 1:]
                    - log_price_window[:, :-1]
            )

            state = torch.cat(
                [
                    log_returns,
                    position
                ],
                dim=1
            )

        # ==================================================
        # Safety
        # ==================================================

        if not torch.isfinite(state).all():
            raise RuntimeError(
                "Agent 1 state contains NaN or Inf."
            )

        return state

    # ======================================================
    # BUILD STATE - Agent 2 - observe the market prices and the current fundamental price
    # ======================================================

    def agent2_observation(
            self,
            log_price_window,
            pos,
            log_mean_window
    ):

        # ==================================================
        # Price returns
        # ==================================================

        log_returns = (
                log_price_window[:, 1:]
                - log_price_window[:, :-1]
        )

        # ==================================================
        # Current price
        # ==================================================

        current_log_price = (
            log_price_window[:, -1:]
        )

        current_log_mean = log_mean_window.to(log_price_window).unsqueeze(1)


        # ==================================================
        # Market price relative to current mean
        # ==================================================

        log_deviation = (
                current_log_price
                - current_log_mean
        )

        # ==================================================
        # Current position
        # ==================================================

        position = (
            pos.to(
                dtype=log_price_window.dtype,
                device=log_price_window.device
            )
            .unsqueeze(1)
        )

        # ==================================================
        # Build state
        # ==================================================

        state = torch.cat(
            [
                log_returns,
                log_deviation,
                position
            ],
            dim=1
        )

        # print("log_mean_window:", log_mean_window.shape)
        # print("price_window:", current_log_price.shape)
        # print("pos:", pos.shape if torch.is_tensor(pos) else type(pos))
        return state

    # ======================================================
    # BUILD STATE – Agent 3 – Observe market prices and fundamental prices over the lookback window
    # ======================================================

    def agent3_observation(
            self,
            log_price_window,
            pos,
            log_mean_window
    ):

        # ==================================================
        # Log returns
        # ==================================================

        log_returns = (
                log_price_window[:, 1:]
                - log_price_window[:, :-1]
        )

        # ==================================================
        # Log price relative to fundamental
        # ==================================================

        log_deviation = (
                log_price_window
                - log_mean_window
        )

        # ==================================================
        # Position
        # ==================================================

        position = (
            pos.to(
                dtype=log_price_window.dtype,
                device=log_price_window.device
            )
            .unsqueeze(1)
        )

        # ==================================================
        # State
        # ==================================================

        state = torch.cat(
            [
                log_returns,
                log_deviation,
                position
            ],
            dim=1
        )

        return state

    def agent4_observation(
            self,
            log_price_window,
            pos
    ):

        # ==================================================
        # Log returns
        # ==================================================

        log_returns10 = (
                log_price_window[:, -10:]
                - log_price_window[:, -11:-1]
        )

        log_returns = (
                log_price_window[:, 1:]
                - log_price_window[:, :-1]
        )

        # Convert log prices to relative prices
        # Normalization improves numerical stability
        prices = torch.exp(
            log_price_window - log_price_window[:, -1:]
        )

        current_price = prices[:, -1:]

        # ==================================================
        # Simple Moving Averages (SMA)
        # ==================================================

        sma10 = prices[:, -10:].mean(
            dim=1,
            keepdim=True
        )

        sma30 = prices[:, -30:].mean(
            dim=1,
            keepdim=True
        )

        # Relative difference between moving averages
        sma_crossover = sma10 / sma30 - 1.0

        # ==================================================
        # MACD (12, 26, 9)
        # ==================================================

        alpha12 = 2.0 / 13.0
        alpha26 = 2.0 / 27.0
        alpha9 = 2.0 / 10.0

        ema_fast = prices[:, :1]
        ema_slow = prices[:, :1]

        macd_signal = torch.zeros_like(current_price)

        for t in range(1, prices.shape[1]):
            price = prices[:, t:t + 1]

            ema_fast = (
                    alpha12 * price
                    + (1.0 - alpha12) * ema_fast
            )

            ema_slow = (
                    alpha26 * price
                    + (1.0 - alpha26) * ema_slow
            )

            macd_value = ema_fast - ema_slow

            macd_signal = (
                    alpha9 * macd_value
                    + (1.0 - alpha9) * macd_signal
            )

        macd = macd_value / current_price
        signal = macd_signal / current_price

        # # ==================================================
        # # Relative Strength Index (RSI)
        # # ==================================================
        #
        # changes = prices[:, -14:] - prices[:, -15:-1]
        #
        # gains = torch.clamp(changes, min=0.0)
        # losses = torch.clamp(-changes, min=0.0)
        #
        # average_gain = gains.mean(dim=1, keepdim=True)
        # average_loss = losses.mean(dim=1, keepdim=True)
        #
        # rsi = 100.0 * average_gain / (
        #         average_gain + average_loss + 1e-8
        # )
        #
        # no_movement = (
        #         (average_gain == 0)
        #         & (average_loss == 0)
        # )
        #
        # rsi = torch.where(
        #     no_movement,
        #     torch.full_like(rsi, 50.0),
        #     rsi
        # )
        #
        # rsi_normalized = (rsi - 50.0) / 50.0



        # ==================================================
        # Momentum
        # ==================================================

        # momentum10 = (
        #         current_price / prices[:, -11:-10] - 1.0
        # )

        momentum20 = (
                current_price / prices[:, -21:-20] - 1.0
        )

        # ==================================================
        # Historical Volatility
        # ==================================================

        volatility20 = log_returns[:, -20:].std(
            dim=1,
            keepdim=True,
            correction=0
        )

        # volatility50 = log_returns[:, -50:].std(
        #     dim=1,
        #     keepdim=True,
        #     correction=0
        # )

        # ==================================================
        # Position
        # ==================================================

        position = (
            pos.to(
                dtype=log_price_window.dtype,
                device=log_price_window.device
            )
            .reshape(-1, 1)
        )

        # ==================================================
        # Build state
        # ==================================================

        state = torch.cat(
            [

                log_returns10,
                sma_crossover,
                macd,
                signal,
                momentum20,
                volatility20,
                position
            ],
            dim=1
        )

        return state


    def print_env(self):
        """
        Print the main environment configuration.
        """

        print("=" * 60)
        print("MARKET ENVIRONMENT")
        print("=" * 60)

        print("\nDimensions")
        print("-" * 60)
        print(f"Batch size:              {self.batch_size}")
        print(f"Number of steps:         {self.num_steps}")
        print(f"Window size:             {self.window_size}")
        print(f"dt:                      {self.dt:.6f}")
        print(f"Device:                  {self.device}")

        print("\nMarket parameters")
        print("-" * 60)
        print(f"Initial price S0:        {self.S_0}")
        print(f"Initial mean:            {self.S_mean}")
        print(f"Price volatility sigma:  {self.sigma}")
        print(f"Mean volatility:         {self.sigma_mean}")
        print(f"Mean reversion kappa:    {self.kappa}")
        print(f"Mean drift mu:           {self.mu_mean}")

        print("\nReward parameters")
        print("-" * 60)
        print(f"Half bid-ask:            {self.half_bidask}")
        print(f"Risk aversion:           {self.risk_aversion}")
        print(f"Gamma:                   {self.gamma}")

        print("\nCurrent tensors")
        print("-" * 60)

        if self.price_seq is not None:
            print(
                f"price_seq shape:         "
                f"{tuple(self.price_seq.shape)}"
            )

        if self.mean_price_seq is not None:
            print(
                f"mean_price_seq shape:    "
                f"{tuple(self.mean_price_seq.shape)}"
            )

        if self.pos is not None:
            print(
                f"position shape:          "
                f"{tuple(self.pos.shape)}"
            )

        print("=" * 60)


    def plot_price(self):
        """
        Plot all simulated price paths.

        Gray:
            individual simulated prices

        Blue:
            average simulated price across the batch

        Red dashed:
            average stochastic mean across the batch
        """

        if self.price_seq is None:
            raise ValueError(
                "No price sequence available."
            )

        # ==================================================
        # Convert tensors to NumPy
        # ==================================================

        prices = (
            self.price_seq
            .detach()
            .cpu()
            .numpy()
        )

        mean_prices = (
            self.mean_price_seq
            .detach()
            .cpu()
            .numpy()
        )

        # ==================================================
        # Time axis
        # ==================================================

        time = (
                np.arange(self.num_steps)
                * self.dt
        )

        # ==================================================
        # Batch averages
        # ==================================================

        average_price = prices.mean(axis=0)

        average_mean = mean_prices.mean(axis=0)

        # ==================================================
        # Plot
        # ==================================================

        plt.figure(figsize=(12, 7))

        # Individual price paths
        for i in range(self.batch_size):
            plt.plot(
                time,
                prices[i],
                color="black",
                alpha=0.15,
                linewidth=0.8
            )

        # Average simulated price
        plt.plot(
            time,
            average_price,
            color="blue",
            linewidth=3,
            label="Average Price"
        )

        # Mean price seq
        plt.plot(
            time,
            average_mean,
            color="red",
            linewidth=3,
            linestyle="--",
            label="Average Mean"
        )

        # ==================================================
        # Labels
        # ==================================================

        plt.xlabel("Time")
        plt.ylabel("Price")

        plt.title(
            f"Mean-Reverting Price Simulation "
            f"(Batch Size = {self.batch_size})"
        )

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def full_reset(self):

        # ==================================================
        # Reset underlying log state
        # ==================================================

        self.log_price = torch.full(
            (self.batch_size,),
            float(np.log(self.S_0)),
            dtype=torch.float32,
            device=self.device
        )

        self.log_mean = torch.full(
            (self.batch_size,),
            float(np.log(self.S_mean)),
            dtype=torch.float32,
            device=self.device
        )

        # ==================================================
        # Reset position
        # ==================================================

        self.pos.zero_()

        # ==================================================
        # Generate new market
        # ==================================================

        (
            self.log_price_seq,
            self.log_mean_seq
        ) = self._simulate_price_sequence()

        self._reset_statistics()

        return (
            self.log_price_seq,
            self.log_mean_seq
        )


    def _reset_statistics(self):

        self.total_reward = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        self.reward_seq = torch.zeros(
            (self.batch_size, self.num_steps - 1),
            dtype=torch.float32,
            device=self.device
        )

        self.last_profit = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        self.last_transaction_cost = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        self.last_risk_penalty = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

        self.last_turnover = torch.zeros(
            self.batch_size,
            dtype=torch.float32,
            device=self.device
        )

    def plot_single_path(self, batch_number=0):
        """
        Plot one simulated price path together with its
        corresponding stochastic mean-price path.

        Parameters
        ----------
        batch_number : int
            Batch index to plot.
            Must be between 0 and batch_size - 1.
        """

        # --------------------------------------------------
        # Check batch number
        # --------------------------------------------------
        user = max(0, min(batch_number, self.batch_size - 1))

        # --------------------------------------------------
        # Get selected batch
        # --------------------------------------------------

        price_path = (
            self.price_seq[user]
            .detach()
            .cpu()
            .numpy()
        )

        mean_path = (
            self.mean_price_seq[user]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------
        # Time axis
        # --------------------------------------------------

        time = np.arange(self.num_steps) * self.dt

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plt.figure(figsize=(12, 6))

        plt.plot(
            time,
            price_path,
            linewidth=2,
            label="Price"
        )

        plt.plot(
            time,
            mean_path,
            linewidth=2,
            linestyle="--",
            label="Mean Price"
        )

        plt.xlabel("Time")
        plt.ylabel("Price")

        plt.title(
            f"Simulated Price Path - Batch {user}"
        )

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def calculate_ema(self, values, period):

        # ==================================================
        # Step 1. Calculate smoothing factor
        # ==================================================

        alpha = 2.0 / (period + 1)

        # ==================================================
        # Step 2. Initialize EMA using SMA
        # ==================================================

        ema = values[:, :period].mean(
            dim=1,
            keepdim=True
        )

        # ==================================================
        # Step 3. Calculate EMA recursively
        # ==================================================

        for t in range(period, values.shape[1]):
            current_value = values[:, t:t + 1]

            ema = (
                    alpha * current_value
                    + (1.0 - alpha) * ema
            )

        return ema
