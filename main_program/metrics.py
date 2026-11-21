from main_program.config import Config
import torch
import matplotlib.pyplot as plt


class TradingMetrics:

    def __init__(self):
        self.reset()
        self.config = Config()

    def reset(self):
        self.rewards = []
        self.gross_returns = []
        self.net_returns = []
        self.transaction_costs = []
        self.risk_penalties = []
        self.turnovers = []
        self.position_changes = []
        self.positions = []

        # New: switching information
        self.previous_positions = []
        self.price_deviations = []

        self.switching_prices = []
        self.relative_prices = []

    # ==================================================
    # Record metrics at each time step
    # ==================================================

    @torch.no_grad()
    def update(
        self,
        market_return,
        previous_position,
        new_position,
        reward,
        transaction_cost,
        risk_penalty,
        market_price,
        fundamental_price
    ):
        r = torch.as_tensor(market_return).detach().reshape(-1)

        def as_batch(value):
            return torch.as_tensor(
                value, device=r.device
            ).detach().reshape(-1).expand_as(r)

        old = as_batch(previous_position)
        new = as_batch(new_position)
        reward = as_batch(reward)
        cost = as_batch(transaction_cost)
        risk = as_batch(risk_penalty)

        price = as_batch(market_price)
        fundamental = as_batch(fundamental_price)

        # Trading returns
        gross = new * r
        net = gross - cost

        # Trading activity
        turnover = (new - old).abs()
        changes = (new != old).float()

        switching_price = price

        # Price deviation at decision time
        relative_price = price / fundamental

        values = {
            "rewards": reward,
            "gross_returns": gross,
            "net_returns": net,
            "transaction_costs": cost,
            "risk_penalties": risk,
            "turnovers": turnover,
            "position_changes": changes,
            "positions": new,
            "previous_positions": old,
            "switching_prices": switching_price,
            "relative_prices": relative_price,
        }

        for name, value in values.items():
            getattr(self, name).append(value.cpu().clone())

    # ==================================================
    # Stack recorded trajectories
    # ==================================================

    def stack(self, name):
        return torch.stack(
            getattr(self, name),
            dim=1
        ).double()

    # ==================================================
    # Identify position transitions
    # ==================================================

    def get_transitions(self, relative=True):
        old = self.stack("previous_positions")
        new = self.stack("positions")

        if relative:
            prices = self.stack("relative_prices")
        else:
            prices = self.stack("switching_prices")

        masks = {
            "enter_long": (old == 0) & (new == 1),
            "enter_short": (old == 0) & (new == -1),
            "exit_long": (old == 1) & (new == 0),
            "exit_short": (old == -1) & (new == 0),

            # Direct position reversals
            "long_to_short": (old == 1) & (new == -1),
            "short_to_long": (old == -1) & (new == 1),
        }

        return {
            name: prices[mask]
            for name, mask in masks.items()
        }

    # ==================================================
    # Switching threshold statistics
    # ==================================================

    def switching_summary(self, relative=True):
        transitions = self.get_transitions(relative=relative)
        results = {}

        for name, values in transitions.items():

            if values.numel() == 0:
                results[name] = {
                    "count": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "p10": float("nan"),
                    "p90": float("nan"),
                }
                continue

            results[name] = {
                "count": values.numel(),
                "mean": values.mean().item(),
                "std": values.std(
                    correction=0
                ).item(),
                "p10": torch.quantile(
                    values, 0.10
                ).item(),
                "p90": torch.quantile(
                    values, 0.90
                ).item(),
            }

        return results

    # ==================================================
    # Plot switching price distributions
    # ==================================================

    def plot_switching_distribution(self, title=None):
        transitions = self.get_transitions()

        plt.figure(figsize=(10, 5))

        for name, values in transitions.items():

            if values.numel() == 0:
                continue

            plt.hist(
                values.numpy(),
                bins=30,
                alpha=0.4,
                density=True,
                label=name.replace("_", " ").title()
            )

        plt.axvline(
            1.0,
            color="black",
            linestyle="--",
            label="Fundamental value"
        )

        plt.xlabel("Price relative to fundamental value")
        plt.ylabel("Density")
        plt.title(title or "Agent Position Switching Levels")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==================================================
    # Calculate episode performance
    # ==================================================

    def summary(self):

        if not self.rewards:
            raise ValueError("No trading metrics recorded.")

        rewards = self.stack("rewards")
        gross = self.stack("gross_returns")
        net = self.stack("net_returns")
        costs = self.stack("transaction_costs")
        risks = self.stack("risk_penalties")
        turnovers = self.stack("turnovers")
        changes = self.stack("position_changes")
        positions = self.stack("positions")

        def mean_trajectory(x):
            return x.mean().item()

        # 1. Cumulative reward
        cumulative_reward = rewards.sum(dim=1)

        # 2. Cumulative returns
        cumulative_gross_return = (
            torch.prod(1 + gross, dim=1) - 1
        )

        cumulative_net_return = (
            torch.prod(1 + net, dim=1) - 1
        )

        # 3. Sharpe ratio
        mean_return = net.mean(dim=1)
        std_return = net.std(dim=1, correction=0)

        sharpe = (mean_return / std_return.clamp_min(1e-12))*(self.config.T **0.5)

        sharpe = torch.where(
            std_return > 1e-12,
            sharpe,
            torch.full_like(sharpe, float("nan"))
        )

        finite_sharpe = sharpe[torch.isfinite(sharpe)]

        # 4. Maximum drawdown
        wealth = torch.cumprod(1 + net, dim=1)

        initial_wealth = torch.ones(
            net.shape[0],
            1,
            dtype=net.dtype,
            device=net.device
        )

        wealth = torch.cat(
            [initial_wealth, wealth], dim=1
        )

        running_peak = torch.cummax(
            wealth, dim=1
        ).values

        drawdown = 1 - wealth / running_peak
        max_drawdown = drawdown.max(dim=1).values

        # 5. Maximum single-step loss
        maximum_loss = (
            -net.min(dim=1).values
        ).clamp_min(0)

        # 6. Transaction costs and risk penalties
        total_costs = costs.sum(dim=1)
        total_risks = risks.sum(dim=1)

        # 7. Position changes and turnover
        total_changes = changes.sum(dim=1)
        change_frequency = changes.mean(dim=1)
        total_turnover = turnovers.sum(dim=1)

        # 8. Position ratios
        sell_ratio = (positions == -1).double().mean()
        hold_ratio = (positions == 0).double().mean()
        buy_ratio = (positions == 1).double().mean()

        # 9. Switching threshold statistics
        switching = self.switching_summary()

        return {
            "episode_reward": mean_trajectory(cumulative_reward),
            "gross_return": mean_trajectory(cumulative_gross_return),
            "net_return": mean_trajectory(cumulative_net_return),

            "sharpe_ratio": (
                finite_sharpe.mean().item()
                if finite_sharpe.numel() > 0
                else float("nan")
            ),

            "max_drawdown": mean_trajectory(max_drawdown),
            "maximum_loss": mean_trajectory(maximum_loss),
            "transaction_cost": mean_trajectory(total_costs),
            "risk_penalty": mean_trajectory(total_risks),

            "position_changes": mean_trajectory(total_changes),
            "change_frequency": mean_trajectory(change_frequency),
            "turnover": mean_trajectory(total_turnover),

            "sell_ratio": sell_ratio.item(),
            "hold_ratio": hold_ratio.item(),
            "buy_ratio": buy_ratio.item(),

            # Nested switching statistics
            "switching": switching,
        }
