import numpy as np
import pandas as pd
import torch

from main_program.agent import DQNAgent
from main_program.replay_buffer import ReplayBuffer
from main_program.prioritized_replay import PrioritizedReplayBuffer
from main_program.metrics import TradingMetrics

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import csv
from pathlib import Path
from datetime import datetime


# ==================================================
# Agent
# ==================================================

# agent_type = 1 --> this agent only can observe the market prices
# agent_type = 2 --> this agent can observe the market prices and the current fundamental price
# agent_type = 3 --> this agent can observe the market prices and the current and historical fundamental prices
# agent_type = 4 --> this agent can do technical analysis (calculate moving average, returns, stats, etc)

SAVE_OUTPUT_PATH = "save_model_weights"

class Trainer:

    def __init__(
        self,
        config,
        market,
        debug = False,
        save_weight = False,
        save_plot = False
    ):

        self.config = config
        self.market = market
        self.metrics = TradingMetrics()
        self.agent_type = config.agent_type
        self.plot_action_strategy = config.plot_action_strategy
        self.debug = debug
        self.save_weight = save_weight
        self.save_plot = save_plot
        self.checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.reset()

        # ==================================================
        # Initial market
        # ==================================================

        initial_state_fn = self._get_initial_state_function(agent_type=self.agent_type)
        initial_state = initial_state_fn(episode=0)

        self.agent = DQNAgent(
            config,
            initial_state
        )

        # ==========================================================
        # Replay buffer
        # ==========================================================

        if self.config.replay_type == "none":

            self.replay_buffer = None

        elif self.config.replay_type == "uniform":

            self.replay_buffer = ReplayBuffer(
                self.config
            )

        elif self.config.replay_type == "prioritized":

            self.replay_buffer = (
                PrioritizedReplayBuffer(
                    self.config
                )
            )

        else:

            raise ValueError(
                f"Unknown replay_type: "
                f"{self.config.replay_type}"
            )


    def reset(self):

        self.market.full_reset()

        # Episode statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_stats = []

        # Position statistics
        self.sell_ratios = []
        self.hold_ratios = []
        self.buy_ratios = []

        # Historical market data
        self.prev_price_history = None
        self.prev_mean_history = None

        self.global_step = 0
        self.optimizer_updates = 0

        self.switching_history = []

        self.best_reward = float("-inf")
        self.best_episode = None


    # ======================================================
    # AGENT 1
    # ======================================================
    def _initial_state1(self, episode):
        price = self._price_window(0, episode=episode)

        return self.market.build_state(price, self.market.pos)

    def _current_state1(self,t, episode):
        price = self._price_window(t, episode)

        return self.market.build_state(price,self.market.pos)

    def _next_state1(self,t, episode):
        next_t = t + 1
        price = self._price_window(next_t, episode)

        return self.market.build_state(price,self.market.pos)

    # ======================================================
    # AGENT 2
    # ======================================================
    def _initial_state2(self, episode):

        # Price history ending at logical t = 0
        price = self._price_window(
            t=0,
            episode=episode
        )

        # Mean history ending at logical t = 0
        mean_window = self._mean_window(
            t=0,
            episode=episode
        )

        # Agent 2 observes only the current mean
        mean_price = mean_window[:, -1]

        return self.market.build_state(
            price_window=price,
            pos=self.market.pos,
            mean_window = mean_price,
        )

    def _current_state2(self, t, episode):

        # Price history ending at t
        price = self._price_window(
            t=t,
            episode=episode
        )

        # Mean history ending at t
        mean_window = self._mean_window(
            t=t,
            episode=episode
        )

        # Only current mean
        mean_price = mean_window[:, -1]

        return self.market.build_state(
            price_window=price,
            pos=self.market.pos,
            mean_window=mean_price
        )

    def _next_state2(self, t, episode):

        next_t = t + 1

        # Price history ending at t+1
        price = self._price_window(
            t=next_t,
            episode=episode
        )

        # Mean history ending at t+1
        mean_window = self._mean_window(
            t=next_t,
            episode=episode
        )

        # Only current mean at t+1
        mean_price = mean_window[:, -1]

        return self.market.build_state(
            price_window=price,
            pos=self.market.pos,
            mean_window=mean_price
        )
    # ======================================================
    # AGENT 3
    # ======================================================
    def _initial_state3(self, episode):

        price = self._price_window(
            t=0,
            episode=episode
        )

        mean_price = self._mean_window(
            t=0,
            episode=episode
        )

        return self.market.build_state(
            price,
            self.market.pos,
            mean_price
        )

    def _current_state3(self, t, episode):

        price = self._price_window(
            t=t,
            episode=episode
        )

        mean_price = self._mean_window(
            t=t,
            episode=episode
        )

        return self.market.build_state(
            price,
            self.market.pos,
            mean_price
        )

    def _next_state3(self, t, episode):

        next_t = t + 1

        price = self._price_window(
            t=next_t,
            episode=episode
        )

        mean_price = self._mean_window(
            t=next_t,
            episode=episode
        )

        return self.market.build_state(
            price,
            self.market.pos,
            mean_price
        )
    # ======================================================
    # AGENT 4
    # ======================================================

    def _initial_state4(
            self,
            episode
    ):

        price = self._price_window(
            0,
            episode
        )


        return self.market.build_state(
            price_window=price,
            pos=self.market.pos
        )

    def _current_state4(
            self,
            t,
            episode
    ):

        price = self._price_window(
            t,
            episode
        )


        return self.market.build_state(
            price_window=price,
            pos=self.market.pos
        )

    def _next_state4(
            self,
            t,
            episode
    ):

        next_t = t + 1

        price = self._price_window(
            next_t,
            episode
        )


        return self.market.build_state(
            price_window=price,
            pos=self.market.pos
        )


    def _pad_window(self, window):

        required_length = self.market.window_size + 1

        # required_length = self.market.window_size + 1

        if window.shape[1] < required_length:
            missing = required_length - window.shape[1]

            padding = torch.zeros(
                window.shape[0],
                missing,
                dtype=window.dtype,
                device=window.device
            )

            window = torch.cat(
                [padding, window],
                dim=1
            )

        return window

    # ======================================================
    # Episode tensors
    # ======================================================

    def _zero_episode_tensor(self):

        return torch.zeros(
            self.market.batch_size,
            dtype=torch.float32,
            device=self.market.device
        )

    # ======================================================
    # Train agent from transition
    # ======================================================

    def _train_step(
            self,
            current_state,
            action,
            reward,
            next_state,
            done
    ):

        # ==================================================
        # Store transition
        # ==================================================

        self.replay_buffer.push(
            current_state,
            action,
            reward,
            next_state,
            done
        )

        self.global_step += 1

        # ==================================================
        # Replay warm-up
        # ==================================================

        minimum_size = max(
            self.config.replay_start_size,
            self.config.replay_batch_size
        )

        if len(self.replay_buffer) < minimum_size:
            return None, None

        # ==================================================
        # Train only every N environment steps
        # ==================================================

        if (
                self.global_step
                % self.config.train_frequency
                != 0
        ):
            return None, None

        # ==================================================
        # Uniform replay
        # ==================================================

        if self.config.replay_type == "uniform":

            (
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_done
            ) = self.replay_buffer.sample()

            loss, td_errors = self.agent.train_step(
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_done
            )

        # ==================================================
        # Prioritized replay
        # ==================================================

        elif self.config.replay_type == "prioritized":

            (
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_done,
                indices,
                weights
            ) = self.replay_buffer.sample()

            loss, td_errors = self.agent.train_step(
                batch_state,
                batch_action,
                batch_reward,
                batch_next_state,
                batch_done,
                weights=weights
            )

            self.replay_buffer.update_priorities(
                indices,
                td_errors
            )

        else:
            raise ValueError(
                f"Unknown replay_type: "
                f"{self.config.replay_type}"
            )

        # ==================================================
        # Count actual optimizer updates
        # ==================================================

        self.optimizer_updates += 1

        # ==================================================
        # Update target network
        # ==================================================

        if (
                self.global_step
                % self.config.target_update_frequency
                == 0
        ):
            self.agent.update_target()

        return loss, td_errors


    # ======================================================
    # Train one episode
    # ======================================================

    def train_episode(
            self,
            episode
    ):

        # ==================================================
        # Generate episode
        # ==================================================

        if episode > 0:
            # Save previous episode history BEFORE reset
            self._store_previous_history()

            # Continue simulation
            self.market.reset()

        # ==================================================
        # State functions
        # ==================================================
        if self.debug:
            print("Price:")
            print(self.market.price_seq)

            print("Mean:")
            print(self.market.mean_price_seq)

            print("Log return:")
            print(self.market.log_price_seq)

            print("Log mean")
            print(self.market.log_mean_seq)

            print("-"*60)
        # ==================================================


        current_state_fn, next_state_fn = (
            self._get_state_functions(
                self.agent_type
            )
        )

        initial_state_fn = (
            self._get_initial_state_function(
                self.agent_type
            )
        )


        actions_by_timestep = []

        action_counts = torch.zeros(
            self.config.num_actions,
            dtype=torch.long,
            device=self.market.device
        )

        initial_position = (
            self.market.pos
            .detach()
            .cpu()
            .clone()
        )

        self.metrics.reset()
        losses = []
        td_errors_episode = []

        # ==================================================
        # Time loop
        #
        # t = 0 ... num_steps - 2
        # because reward uses t -> t+1
        # ==================================================

        for t in range(
                self.market.num_steps - 1
        ):

            # ==================================================
            # Current state
            # ==================================================

            if t == 0:

                current_state = (
                    initial_state_fn(episode)
                )

            else:

                current_state = (
                    current_state_fn(
                        t,
                        episode
                    )
                )

            # ==================================================
            # State debug for first 3 episodes
            # ==================================================

            if (
                    self.debug
                    and episode < 5
            ):
                print("\n" + "=" * 70)

                print(
                    f"STATE DEBUG | "
                    f"Episode {episode + 1} | "
                    f"t = {t}"
                )

                print("=" * 70)

                print(
                    "State shape:",
                    current_state.shape
                )

                print(
                    "First 3 states:"
                )

                torch.set_printoptions(
                    precision=4,
                    sci_mode=False
                )

                print(
                    current_state[:3]
                    .detach()
                    .cpu()
                )

                print(
                    "Current position:"
                )

                print(
                    self.market.pos[:3]
                    .detach()
                    .cpu()
                )

            # ==================================================
            # Replay buffer debug
            # ==================================================


                diagnostics = (
                    self.agent.get_diagnostics(
                        current_state
                    )
                )

                print("\n" + "=" * 60)

                print(
                    f"LEARNING DEBUG | "
                    f"Episode {episode + 1} | "
                    f"t = {t}"
                )

                print("=" * 60)

                # ==============================================
                # Replay
                # ==============================================

                print(
                    f"Replay type: "
                    f"{self.config.replay_type}"
                )

                if self.replay_buffer is not None:
                    print(
                        f"Replay size: "
                        f"{len(self.replay_buffer)}"
                    )

                print(
                    f"Replay minibatch: "
                    f"{self.config.replay_batch_size}"
                )

                # ==============================================
                # Q diagnostics
                # ==============================================

                q_mean = diagnostics[
                    "q_mean"
                ]

                print(
                    "Q mean "
                    "[SHORT, FLAT, LONG]:",
                    q_mean.numpy()
                )

                print(
                    f"Mean Q gap: "
                    f"{diagnostics['q_gap']:.6f}"
                )

                # ==============================================
                # Softmax
                # ==============================================

                if (
                        diagnostics[
                            "mean_probabilities"
                        ]
                        is not None
                ):
                    print(
                        "Mean softmax probs "
                        "[SHORT, FLAT, LONG]:",
                        diagnostics[
                            "mean_probabilities"
                        ].numpy()
                    )

                    print(
                        f"Temperature: "
                        f"{diagnostics['temperature']:.6f}"
                    )

                # ==============================================
                # Greedy action ratios
                # ==============================================

                ratios = diagnostics[
                    "greedy_ratios"
                ]

                print(
                    "Greedy ratios: "
                    f"SHORT={ratios[0].item():.1%} | "
                    f"FLAT={ratios[1].item():.1%} | "
                    f"LONG={ratios[2].item():.1%}"
                )
                print('')

            # ==================================================
            # Action
            # ==================================================
            # Save position before executing the action
            previous_position = self.market.pos.detach().clone()

            # Current market information at decision time
            market_price = self.market.price_seq[:, t]
            fundamental_price = self.market.mean_price_seq[:, t]

            action = self.agent.act(
                current_state,
                greedy=False,
            )

            action_counts += (
                torch.bincount(
                    action,
                    minlength=(
                        self.config.num_actions
                    )
                )
            )

            actions_by_timestep.append(
                action
                .detach()
                .cpu()
                .clone()
            )

            # ==================================================
            # Environment transition
            # ==================================================

            reward, done = (
                self.market.step(
                    action,
                    t
                )
            )

            # Position after executing the action
            new_position = self.market.pos.detach().clone()

            # ==================================================
            # Episode statistics
            # ==================================================

            self.metrics.update(
                market_return=self.market.reward_seq[:, t],
                previous_position=previous_position,
                new_position=new_position,
                reward=reward,
                transaction_cost=self.market.last_transaction_cost,
                risk_penalty=self.market.last_risk_penalty,
                market_price=market_price,
                fundamental_price=fundamental_price
            )

            next_state = (
                next_state_fn(
                    t,
                    episode
                )
            )

            # ==================================================
            # State shape check
            # ==================================================

            if (
                    current_state.shape
                    != next_state.shape
            ):
                raise RuntimeError(
                    f"State mismatch at "
                    f"episode={episode}, "
                    f"t={t}\n"
                    f"current_state: "
                    f"{current_state.shape}\n"
                    f"next_state: "
                    f"{next_state.shape}"
                )

            # ==================================================
            # Train
            # ==================================================
            (
                loss,
                mean_td_error
            ) = self._train_step(
                current_state,
                action,
                reward,
                next_state,
                done
            )

            if loss is not None:
                losses.append(
                    float(loss)
                )

            if mean_td_error is not None:
                td_errors_episode.append(
                    mean_td_error
                )


        # ==================================================
        # Store final training episode for plotting
        # ==================================================

        if len(actions_by_timestep) > 0:

            action_history = torch.stack(
                actions_by_timestep,
                dim=1
            )

            if (
                    episode + 1
                    == self.config.num_episodes
            ):
                self.final_training_actions = (
                    action_history.clone()
                )

                self.final_training_prices = (
                    self.market.price_seq
                    .detach()
                    .cpu()
                    .clone()
                )

                self.final_training_mean_prices = (
                    self.market.mean_price_seq
                    .detach()
                    .cpu()
                    .clone()
                )

                self.final_training_initial_position = (
                    initial_position.clone()
                )

        # ==================================================
        # Save statistics
        # ==================================================

        trading_summary = self.metrics.summary()
        switching_summary = trading_summary["switching"]

        self.switching_history.append({
            "episode": episode,
            "switching": switching_summary
        })



        self._save_episode_statistics(
            episode=episode,
            trading_summary=trading_summary,
            losses=losses
        )


    # ======================================================
    # Statistics
    # ======================================================

    def _save_episode_statistics(
            self,
            episode,
            trading_summary,
            losses
    ):

        # ==================================================
        # Training loss
        # ==================================================

        mean_loss = (
            float(np.mean(losses))
            if losses
            else np.nan
        )

        # ==================================================
        # Build episode statistics
        # ==================================================

        stats = {
            "episode": episode + 1,

            "reward": trading_summary["episode_reward"],
            "profit": trading_summary["gross_return"],
            "net_return": trading_summary["net_return"],

            "transaction_cost": trading_summary["transaction_cost"],
            "risk_penalty": trading_summary["risk_penalty"],
            "turnover": trading_summary["turnover"],

            "sharpe_ratio": trading_summary["sharpe_ratio"],
            "max_drawdown": trading_summary["max_drawdown"],
            "maximum_loss": trading_summary["maximum_loss"],

            "position_changes": trading_summary["position_changes"],
            "change_frequency": trading_summary["change_frequency"],

            "short": trading_summary["sell_ratio"],
            "flat": trading_summary["hold_ratio"],
            "long": trading_summary["buy_ratio"],

            "loss": mean_loss,

            "epsilon": self.agent.epsilon,
            "temperature": self.agent.temperature,
        }

        # ==================================================
        # Save training history
        # ==================================================

        self.episode_rewards.append(stats["reward"])
        self.episode_losses.append(stats["loss"])

        self.sell_ratios.append(stats["short"])
        self.hold_ratios.append(stats["flat"])
        self.buy_ratios.append(stats["long"])


        # ==================================================
        # Save and print
        # ==================================================

        self.episode_stats.append(stats)

        # # Keep the checkpoint with the highest observed episode reward.
        # # The model already includes the updates made during this episode.
        # reward = float(stats["reward"])
        # if np.isfinite(reward) and reward > self.best_reward:
        #     self.best_reward = reward
        #     self.best_episode = stats["episode"]
        #
        #     if self.save_weight:
        #         self.save_model(checkpoint="best")

        self._print_progress(stats)


    # ======================================================
    # Print progress
    # ======================================================

    def _print_progress(
            self,
            stats
    ):

        episode = stats["episode"]

        if (
                episode % 30 != 0
                and episode != self.config.num_episodes
        ):
            return

        # ==================================================
        # Training progress
        # ==================================================

        print(
            f"Episode {episode:4d} | "
            f"Reward: {stats['reward']:9.4f} | "
            f"Cost: {stats['transaction_cost']:9.4f} | "
            f"Risk: {stats['risk_penalty']:9.4f} | "
            f"Loss: {stats['loss']:9.5f} | "
            f"Epsilon: {self.agent.epsilon:6.3f} | "
            f"Temperature: {self.agent.temperature:6.3f} | "
            f"SHORT: {stats['short']:6.1%} | "
            f"FLAT: {stats['flat']:6.1%} | "
            f"LONG: {stats['long']:6.1%}"
        )

        # ==================================================
        # Position switching statistics
        # ==================================================

        if episode == self.config.num_episodes:

            switching = self.metrics.switching_summary()

            print("\n" + "=" * 80)
            print("FINAL EPISODE - POSITION SWITCHING THRESHOLDS")
            print("=" * 80)

            for name, values in switching.items():

                if values["count"] == 0:
                    continue

                print(
                    f"{name:<18} | "
                    f"Count: {values['count']:5d} | "
                    f"Mean: {values['mean']:8.4f} | "
                    f"Std: {values['std']:8.4f} | "
                    f"P10-P90: "
                    f"[{values['p10']:.4f}, "
                    f"{values['p90']:.4f}]"
                )

            print("=" * 80)

    # ======================================================
    # Main training
    # ======================================================

    def train(self):


        for episode in range(
                self.config.num_episodes
        ):

            # ==================================================
            # PER beta schedule
            # ==================================================

            self._update_per_beta(
                episode
            )

            # ==================================================
            # Train episode
            # ==================================================

            self.train_episode(
                episode
            )

            # ==================================================
            # Exploration
            # ==================================================

            self.agent.decay_epsilon()
            self.agent.decay_temperature()

            # if episode + 1 == self.config.num_episodes:
            #     self.metrics.plot_switching_distribution(
            #         title="Final Training Episode: Switching Price\nMarket Price/Fundamental Price"
            #     )




        if self.save_weight:
            self.save_model(checkpoint="final")

        print(
            f"Best training reward: {self.best_reward:.4f} "
            f"at episode {self.best_episode}"
        )

        return self.episode_stats


    # ======================================================
    # Save model
    # ======================================================

    def save_model(self, checkpoint="final"):

        if checkpoint not in ("best", "final"):
            raise ValueError("checkpoint must be 'best' or 'final'")

        output_dir = Path(__file__).resolve().parent.parent / SAVE_OUTPUT_PATH
        output_dir.mkdir(parents=True, exist_ok=True)

        parts = [
            f"agent{self.agent_type}",
            "model",
            f"ep{self.config.num_episodes}",
            f"cost{self.config.half_bidask}"
        ]

        if self.agent_type != 4:
            parts.append(f"lookback{self.config.window_size}")

        parts.append(self.config.replay_type)
        parts.append("DuelingDQN" if self.config.use_DuelingDQN else "DQN")
        parts.append("softmax" if self.config.use_softmax else "epsilon")
        parts.append(checkpoint)
        parts.append(self.checkpoint_timestamp)

        filepath = output_dir / ("_".join(parts) + ".pth")

        # Save a raw state_dict, compatible with the existing Evaluator.
        torch.save(self.agent.model.state_dict(), filepath)

        if checkpoint == "best":
            print(
                f"Best model updated | Episode {self.best_episode} | "
                f"Reward {self.best_reward:.4f} | {filepath.name}"
            )
        else:
            print(f"Final model saved | {filepath.name}")

        return filepath


    def _update_episode_statistics(
            self,
            episode_reward,
            episode_profit,
            episode_cost,
            episode_risk,
            episode_turnover,
            reward
    ):

        episode_reward += reward
        episode_profit += self.market.last_profit
        episode_cost += self.market.last_transaction_cost
        episode_risk += self.market.last_risk_penalty
        episode_turnover += self.market.last_turnover

    # ======================================================
    # Set Agent Type
    # ======================================================

    def _get_state_functions(
            self,
            agent_type
    ):

        if agent_type == 1:

            return (
                self._current_state1,
                self._next_state1
            )

        elif agent_type == 2:

            return (
                self._current_state2,
                self._next_state2
            )

        elif agent_type == 3:

            return (
                self._current_state3,
                self._next_state3
            )

        elif agent_type == 4:
            return (
                self._current_state4,
                self._next_state4
            )

        else:

            raise ValueError(
                f"Unknown agent_type: {agent_type}"
            )

    def _get_initial_state_function(
            self,
            agent_type
    ):

        if agent_type == 1:

            return self._initial_state1

        elif agent_type == 2:

            return self._initial_state2

        elif agent_type == 3:

            return self._initial_state3

        elif agent_type == 4:
            return self._initial_state4

        else:

            raise ValueError(
                f"Unknown agent_type: {agent_type}"
            )

    def save_plot_training(self):


        episodes = np.arange(1,len(self.episode_rewards) + 1)

        # ==================================================
        # 1. Loss
        # ==================================================

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            self.episode_losses
        )

        plt.xlabel("Episode")
        plt.ylabel("Mean Loss")

        if self.agent_type == 4:
            plt.title(
                f"Agent {self.agent_type} - Training Loss"
            )
        else:

            plt.title(
                f"Agent {self.agent_type} - Training Loss -\nLookback window = {self.config.window_size}"
            )

        plt.grid()
        plt.tight_layout()

        if self.save_plot:
            self._save_plot(filename="loss")



        # ==================================================
        # 2. Episode Reward
        # ==================================================

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            episodes,
            self.episode_rewards,
            alpha=0.5,
            label="Episode Reward"
        )

        # Moving average
        reward_window = min(10,len(self.episode_rewards))

        if reward_window > 1:
            rewards = np.array(
                self.episode_rewards
            )

            moving_average = np.convolve(
                rewards,
                np.ones(reward_window) / reward_window,
                mode="valid"
            )

            plt.plot(
                np.arange(
                    reward_window,
                    len(rewards) + 1
                ),
                moving_average,
                linewidth=2,
                label=f"{reward_window}-Episode Moving Average"
            )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")

        if self.agent_type == 4:
            plt.title(
                f"Agent {self.agent_type} - Episode Reward"
            )
        else:

            plt.title(
                f"Agent {self.agent_type} - Episode Reward -\nLookback window = {self.config.window_size}"
            )

        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.save_plot:
            self._save_plot(filename="rewards")


        # ==================================================
        # 3. Action Ratios
        # ==================================================

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            episodes,
            self.sell_ratios,
            label="SHORT"
        )

        plt.plot(
            episodes,
            self.hold_ratios,
            label="FLAT"
        )

        plt.plot(
            episodes,
            self.buy_ratios,
            label="LONG"
        )

        plt.xlabel("Episode")
        plt.ylabel("Action Ratio")

        if self.agent_type == 4:
            plt.title(
                f"Agent {self.agent_type} - Action Distribution"
            )
        else:
            plt.title(
                f"Agent {self.agent_type} - Action Distribution -\nLookback window = {self.config.window_size}"
            )

        plt.legend()
        plt.grid()
        plt.tight_layout()

        if self.save_plot:
            self._save_plot(filename="action")


    def plot_batch_strategy(
            self,
            action_history,
            episode,
            batch_number = 0,
    ):

        # ==================================================
        # Convert action history to NumPy
        # ==================================================

        if hasattr(action_history, "detach"):
            actions_all = (
                action_history
                .detach()
                .cpu()
                .numpy()
            )
        else:
            actions_all = np.asarray(action_history)

        if actions_all.ndim != 2:
            raise ValueError(
                "action_history must have shape (B, T)."
            )

        # ==================================================
        # Select a random trajectory
        # ==================================================

        actions = actions_all[batch_number]

        # Convert action indices to positions:
        # 0 -> -1 (SHORT)
        # 1 ->  0 (FLAT)
        # 2 -> +1 (LONG)

        positions = actions - 1

        # ==================================================
        # Align positions with market prices
        # ==================================================

        num_actions = len(positions)
        num_prices = self.market.price_seq.shape[1]

        # Assume actions correspond to transitions
        # from price[t] to price[t+1].
        if num_actions != num_prices - 1:
            raise ValueError(
                f"Expected {num_prices - 1} actions, "
                f"but received {num_actions}. "
                "Check the training time indices."
            )

        time_steps = np.arange(
            1,
            num_actions + 1
        )

        # ==================================================
        # Market and fundamental prices
        # ==================================================

        market_price = (
            self.market.price_seq[
                batch_number,
                1:num_actions + 1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        mean_price = (
            self.market.mean_price_seq[
                batch_number,
                1:num_actions + 1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # ==================================================
        # Create figure
        # ==================================================

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13, 8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3, 1]
            }
        )

        # ==================================================
        # Top: Market and fundamental prices
        # ==================================================

        axes[0].plot(
            time_steps,
            market_price,
            label="Market Price",
            linewidth=1.5
        )

        axes[0].plot(
            time_steps,
            mean_price,
            label="Fundamental Price",
            linestyle="--",
            linewidth=1.5
        )

        axes[0].set_title(
            f"Agent {self.agent_type} | "
            f"Episode {episode} | "
            f"Trajectory {batch_number}"
        )

        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # ==================================================
        # Bottom: Trading positions
        # ==================================================

        axes[1].step(
            time_steps,
            positions,
            where="post",
            linewidth=1.5
        )

        # Color points according to position
        colors = np.where(
            positions == -1,
            "red",
            np.where(
                positions == 0,
                "gray",
                "green"
            )
        )

        axes[1].scatter(
            time_steps,
            positions,
            c=colors,
            s=12,
            zorder=3
        )

        axes[1].axhline(
            0,
            color="gray",
            linewidth=0.8
        )

        axes[1].set_yticks([-1, 0, 1])

        axes[1].set_yticklabels([
            "SELL",
            "HOLD",
            "BUY"
        ])

        axes[1].set_ylim(-1.5, 1.5)

        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Position")

        legend_elements = [
            Patch(color="green", label="SELL"),
            Patch(color="gray", label="HOLD"),
            Patch(color="red", label="BUY")
        ]

        axes[1].legend(
            handles=legend_elements,
            loc="upper right"
        )

        axes[1].grid(alpha=0.3)

        # ==================================================
        # Save and display
        # ==================================================

        plt.tight_layout()

        if self.save_plot:
            self._save_plot(
                filename=f"strategy"
            )



    def save_training_statistics(self):

        output_folder = Path("../results")

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        parts = [
            f"agent{self.agent_type}",
            "training_statistics",
            f"ep{self.config.num_episodes}"
        ]

        if self.agent_type != 4:
            parts.append(
                f"lookback{self.config.window_size}"
            )

        parts.append(timestamp)

        filename = "_".join(parts) + ".csv"

        file_path = output_folder / filename

        fieldnames = list(self.episode_stats[0].keys())

        with file_path.open(
                mode="w",
                newline="",
                encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.episode_stats)

        print("Training stats are saved.")

    def save_config(self):

        # ==================================================
        # Output directory
        # ==================================================

        output_folder = (
                Path(__file__).resolve().parent.parent / "results"
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==================================================
        # Generate filename
        # ==================================================

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = [
            f"agent{self.agent_type}",
            "config",
            f"ep{self.config.num_episodes}"
        ]

        if self.agent_type != 4:
            parts.append(
                f"lookback{self.config.window_size}"
            )

        parts.append(timestamp)

        filename = "_".join(parts) + ".csv"


        file_path = output_folder / filename

        # ==================================================
        # Extract configuration
        # ==================================================

        config_dict = vars(self.config)

        # ==================================================
        # Save configuration
        # ==================================================

        with file_path.open(
                mode="w",
                newline="",
                encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "parameter",
                "value"
            ])

            for key, value in sorted(config_dict.items()):
                writer.writerow([
                    key,
                    value
                ])

        print(f"Config is saved.")


    def _save_plot(
            self,
            filename
    ):

        output_folder = Path("../plots")
        # Path("results") / "plots"

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = [
            f"agent{self.agent_type}",
            filename,
            f"ep{self.config.num_episodes}",
            f"w{self.config.window_size}"
        ]

        if self.config.use_DuelingDQN:
            parts.append("DuelingDQN")
        else:
            parts.append("DQN")

        if self.config.use_softmax:
            parts.append("softmax")
        else:
            parts.append("epsilon")

        if self.config.replay_type == "uniform":
            parts.append("uniform")
        elif self.config.replay_type == "prioritized":
            parts.append("prioritized")

        parts.append(timestamp)

        plot_name = "_".join(parts) + ".png"

        plt.savefig(
            output_folder / plot_name,
            dpi=300,
            bbox_inches="tight"
        )



    def _initialize_history(self):

        batch_size = self.market.log_price_seq.shape[0]

        w = self.market.window_size

        self.prev_price_history = torch.zeros(
            batch_size,
            w,
            dtype=self.market.log_price_seq.dtype,
            device=self.market.log_price_seq.device
        )

        self.prev_mean_history = torch.zeros(
            batch_size,
            w,
            dtype=self.market.log_mean_seq.dtype,
            device=self.market.log_mean_seq.device
        )

    def _store_previous_history(self):

        # if self.agent_type == 4:
        #     w = max(self.agent4_window, self.market.window_size)
        # else:
        #     w = self.market.window_size

        w = self.market.window_size

        self.prev_price_history = (
            self.market.log_price_seq[:, -(w + 1):]
            .detach()
            .clone()
        )

        self.prev_mean_history = (
            self.market.log_mean_seq[:, -(w + 1):]
            .detach()
            .clone()
        )

    def _price_window(
            self,
            t,
            episode
    ):

        # if self.agent_type == 4:
        #     w = max(self.agent4_window, self.market.window_size)
        # else:
        #     w = self.market.window_size

        w = self.market.window_size

        required = w + 1

        # ==================================================
        # Episode 0
        # No previous episode history exists
        # ==================================================

        if episode == 0:

            start = max(
                0,
                t - w
            )

            window = self.market.log_price_seq[
                :,
                start:t + 1
            ]

            # --------------------------------------------------
            # Not enough historical observations yet
            #
            # Instead of zero-padding:
            #
            #   [0, 0, 120]
            #
            # repeat the first available price:
            #
            #   [120, 120, 120]
            #
            # This is safe for technical indicators because
            # initial returns become zero instead of NaN/Inf.
            # --------------------------------------------------

            if window.shape[1] < required:
                missing = (
                        required
                        - window.shape[1]
                )

                first_price = window[:, :1]

                padding = first_price.repeat(
                    1,
                    missing
                )

                window = torch.cat(
                    [
                        padding,
                        window
                    ],
                    dim=1
                )

        # ==================================================
        # Episode 1+
        # Previous episode history is available
        # ==================================================

        else:

            if self.prev_price_history is None:
                raise RuntimeError(
                    "prev_price_history is None. "
                    "Call _store_previous_history() "
                    "before market.reset()."
                )

            # --------------------------------------------------
            # t = 0
            #
            # Use complete final window from previous episode.
            #
            # Example w=2:
            #
            # previous episode:
            # [..., 97, 101, 103]
            #
            # episode t=0:
            # [97, 101, 103]
            # --------------------------------------------------

            if t == 0:

                window = (
                    self.prev_price_history
                )

            # --------------------------------------------------
            # Beginning of new episode
            #
            # price_seq[:, 0] is the final price carried from
            # the previous episode.
            #
            # Therefore newly generated observations begin
            # at price_seq[:, 1].
            #
            # Example w=2:
            #
            # t=1:
            # [101, 103, new_1]
            #
            # t=2:
            # [103, new_1, new_2]
            # --------------------------------------------------

            elif t <= w:

                previous_needed = (
                        required - t
                )

                previous = (
                    self.prev_price_history[
                        :,
                        -previous_needed:
                    ]
                )

                current = (
                    self.market.log_price_seq[
                        :,
                        1:t + 1
                    ]
                )

                window = torch.cat(
                    [
                        previous,
                        current
                    ],
                    dim=1
                )

            # --------------------------------------------------
            # Enough observations from the new episode
            # --------------------------------------------------

            else:

                window = (
                    self.market.log_price_seq[
                        :,
                        t - w:t + 1
                    ]
                )

        # ==================================================
        # Safety checks
        # ==================================================

        if window.dim() != 2:
            raise RuntimeError(
                f"Price window must be 2D, "
                f"but got shape {window.shape}. "
                f"episode={episode}, t={t}"
            )

        if window.shape[1] != required:
            raise RuntimeError(
                f"Invalid price window width. "
                f"episode={episode}, "
                f"t={t}, "
                f"got={window.shape[1]}, "
                f"expected={required}"
            )

        if not torch.isfinite(window).all():
            raise RuntimeError(
                f"Non-finite value in price window. "
                f"episode={episode}, t={t}"
            )

        if (window <= 0).any():
            raise RuntimeError(
                f"Non-positive price found in window. "
                f"episode={episode}, t={t}"
            )

        return window

    def _mean_window(
            self,
            t,
            episode
    ):

        w = self.market.window_size
        required = w + 1

        # ==================================================
        # Episode 0
        # No previous episode history exists
        # ==================================================

        if episode == 0:

            start = max(
                0,
                t - w
            )

            window = self.market.log_mean_seq[
                :,
                start:t + 1
            ]

            # --------------------------------------------------
            # Not enough historical observations yet
            #
            # Instead of:
            #
            # [0, 0, mean_0]
            #
            # use:
            #
            # [mean_0, mean_0, mean_0]
            # --------------------------------------------------

            if window.shape[1] < required:
                missing = (
                        required
                        - window.shape[1]
                )

                first_mean = window[:, :1]

                padding = first_mean.repeat(
                    1,
                    missing
                )

                window = torch.cat(
                    [
                        padding,
                        window
                    ],
                    dim=1
                )

        # ==================================================
        # Episode 1+
        # ==================================================

        else:

            if self.prev_mean_history is None:
                raise RuntimeError(
                    "prev_mean_history is None. "
                    "Call _store_previous_history() "
                    "before market.reset()."
                )

            # --------------------------------------------------
            # t = 0
            #
            # Use the complete final mean window
            # from the previous episode
            # --------------------------------------------------

            if t == 0:

                window = (
                    self.prev_mean_history
                )

            # --------------------------------------------------
            # Beginning of new episode
            #
            # mean_price_seq[:, 0] is the carried final mean
            # from the previous episode.
            #
            # New values begin at index 1.
            # --------------------------------------------------

            elif t <= w:

                previous_needed = (
                        required - t
                )

                previous = (
                    self.prev_mean_history[
                        :,
                        -previous_needed:
                    ]
                )

                current = (
                    self.market.log_mean_seq[
                        :,
                        1:t + 1
                    ]
                )

                window = torch.cat(
                    [
                        previous,
                        current
                    ],
                    dim=1
                )

            # --------------------------------------------------
            # Enough new observations exist
            # --------------------------------------------------

            else:

                window = (
                    self.market.log_mean_seq[
                        :,
                        t - w:t + 1
                    ]
                )

        # ==================================================
        # Safety checks
        # ==================================================

        if window.dim() != 2:
            raise RuntimeError(
                f"Mean window must be 2D, "
                f"but got shape {window.shape}. "
                f"episode={episode}, t={t}"
            )

        if window.shape[1] != required:
            raise RuntimeError(
                f"Invalid MEAN window width. "
                f"episode={episode}, "
                f"t={t}, "
                f"got={window.shape[1]}, "
                f"expected={required}"
            )

        if not torch.isfinite(window).all():
            raise RuntimeError(
                f"Non-finite value in mean window. "
                f"episode={episode}, t={t}"
            )

        if (window <= 0).any():
            raise RuntimeError(
                f"Non-positive mean price found. "
                f"episode={episode}, t={t}"
            )

        return window

    # ======================================================
    # Update prioritized replay beta
    # ======================================================

    def _update_per_beta(
            self,
            episode
    ):

        if (
                self.config.replay_type
                != "prioritized"
        ):
            return

        progress = (
                episode
                / max(
            1,
            self.config.num_episodes - 1
        )
        )

        beta = (
                self.config.per_beta_start
                + progress
                * (
                        self.config.per_beta_end
                        - self.config.per_beta_start
                )
        )

        self.replay_buffer.set_beta(
            beta
        )

    def plot_training_positions(
            self,
            batch_idx=0
    ):

        # ==================================================
        # Safety
        # ==================================================

        batch_idx = max(
            0,
            min(
                batch_idx,
                self.market.batch_size - 1
            )
        )

        # ==================================================
        # Get final training episode
        # ==================================================

        prices = (
            self.final_training_prices[
                batch_idx
            ]
            .numpy()
        )

        mean_prices = (
            self.final_training_mean_prices[
                batch_idx
            ]
            .numpy()
        )

        actions = (
            self.final_training_actions[
                batch_idx
            ]
            .numpy()
        )

        initial_position = (
            self.final_training_initial_position[
                batch_idx
            ]
            .item()
        )

        # ==================================================
        # Actions -> positions
        #
        # 0 = SHORT -> -1
        # 1 = FLAT  ->  0
        # 2 = LONG  -> +1
        # ==================================================

        action_positions = (
                actions - 1
        )


        positions = np.concatenate(
            [
                np.array(
                    [initial_position]
                ),
                action_positions
            ]
        )

        # ==================================================
        # Check dimensions
        # ==================================================

        # print(
        #     "Prices:",
        #     prices.shape
        # )
        #
        # print(
        #     "Fundamental Prices:",
        #     mean_prices.shape
        # )
        #
        # print(
        #     "Actions:",
        #     actions.shape
        # )
        #
        # print(
        #     "Positions:",
        #     positions.shape
        # )

        assert (
                len(prices)
                == len(mean_prices)
                == len(positions)
        )

        # ==================================================
        # Previous positions
        # ==================================================

        prev_positions = np.empty_like(
            positions
        )

        prev_positions[0] = (
            initial_position
        )

        prev_positions[1:] = (
            positions[:-1]
        )

        changed = (
                positions != prev_positions
        )

        # ==================================================
        # Position changes
        # ==================================================

        long_idx = np.where(
            (positions == 1)
            & changed
        )[0]

        short_idx = np.where(
            (positions == -1)
            & changed
        )[0]

        close_idx = np.where(
            (positions == 0)
            & (prev_positions != 0)
        )[0]

        # ==================================================
        # Plot
        # ==================================================

        t = np.arange(
            len(prices)
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            t,
            prices,
            label="Market Price"
        )

        plt.plot(
            t,
            mean_prices,
            label="Fundamental Price"
        )

        # LONG
        plt.scatter(
            long_idx,
            prices[long_idx],
            color="red",
            marker="^",
            s=90,
            label="Long"
        )

        # SHORT
        plt.scatter(
            short_idx,
            prices[short_idx],
            color="green",
            marker="^",
            s=90,
            label="Short"
        )

        # CLOSE
        plt.scatter(
            close_idx,
            prices[close_idx],
            color="black",
            marker="v",
            s=90,
            label="Close"
        )

        plt.xlabel(
            "Time Step"
        )

        plt.ylabel(
            "Price"
        )

        plt.title(
            f"Training - Final Episode "
            f"- Batch {batch_idx}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        self._save_plot(
            filename=f"positions{batch_idx}"
        )


    def get_config(self):

        config_dict = vars(self.config)

        config_df = pd.DataFrame(
            sorted(config_dict.items()),
            columns=["parameter", "value"]
        )

        return config_df

    def get_training_statistics(self):

        if not self.episode_stats:
            return pd.DataFrame()

        stats_df = pd.DataFrame(
            self.episode_stats
        )

        return stats_df

    def get_transition_statistics(self, relative=True):

        transitions = self.metrics.get_transitions(relative=relative)

        rows = []

        for name, prices in transitions.items():
            prices = np.asarray(prices).reshape(-1)

            rows.append({
                "transition": name,
                "count": len(prices),
                "mean": np.mean(prices) if len(prices) else np.nan,
                "std": np.std(prices) if len(prices) else np.nan,
                "p10": np.percentile(prices, 10) if len(prices) else np.nan,
                "p90": np.percentile(prices, 90) if len(prices) else np.nan
            })

        return pd.DataFrame(rows).round(4)

    def save_results(self):

        output_folder = Path(__file__).resolve().parent.parent / "results"
        output_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = [
            f"agent{self.agent_type}",
            f"ep{self.config.num_episodes}",
            f"cost{self.config.half_bidask}"
        ]

        if self.agent_type != 4:
            parts.append(f"lookback{self.config.window_size}")

        if self.config.replay_type == "prioritized":
            parts.append(f"prioritized")
        else:
            parts.append(f"uniform")

        parts.append(timestamp)

        filename = "_".join(parts) + ".xlsx"
        file_path = output_folder / filename

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            self.get_config().to_excel(
                writer,
                sheet_name="Config",
                index=False
            )

            self.get_training_statistics().to_excel(
                writer,
                sheet_name="Training Statistics",
                index=False
            )


        print(f"Results and config are saved.")
