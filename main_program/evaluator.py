import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from main_program.agent import DQNAgent
from pathlib import Path

SAVE_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "save_model_weights"
)

SAVE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

class Evaluator:

    def __init__(
            self,
            market,
            config,
            model_path=None
    ):

        self.market = market
        self.config = config
        self.agent_type = config.agent_type
        self.batch_size = config.batch_size
        self.num_t = config.num_steps
        self.periods_per_year = config.T
        self.agent4_window = 51

        if model_path is None:
            self.model_path = SAVE_OUTPUT_PATH / f"dqn_model_agent{self.agent_type}.pth"
        else:
            given_path = Path(model_path)
            if given_path.suffix != ".pth":
                given_path = given_path.with_suffix(".pth")
            self.model_path = (
                given_path if given_path.is_absolute()
                else SAVE_OUTPUT_PATH / given_path
            )

        # Build an initial observation with the same state construction
        # that will be used for every evaluation step.
        self.prev_log_price_history = None
        self.prev_log_mean_history = None
        self.market.full_reset()
        initial_state = self._build_state(t=0, episode=0)

        self.agent = DQNAgent(config, initial_state)

        # ======================================================
        # Number of evaluation episodes
        # ======================================================

        self.num_episodes = config.num_evaluation_episodes

        # ======================================================
        # Load trained DQN
        # ======================================================

        checkpoint = torch.load(
            self.model_path,
            map_location=market.device,
            weights_only=True
        )


        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        self.agent.model.load_state_dict(
            state_dict
        )

        self.agent.model.to(market.device)
        self.agent.model.eval()

        print(
            f"Loaded trained model from: "
            f"{self.model_path}"
        )

        self._initialize_statistics()


    # ==========================================================
    # Initialize statistics
    # ==========================================================

    def _initialize_statistics(self):

        market = self.market

        # ------------------------------------------------------
        # Step-level histories
        # ------------------------------------------------------

        self.action_history = []
        self.reward_history = []

        self.price_history = []
        self.mean_price_history = []

        # ------------------------------------------------------
        # Episode-level histories
        # ------------------------------------------------------

        self.episode_reward_history = []
        self.episode_profit_history = []
        self.episode_cost_history = []
        self.episode_risk_history = []
        self.episode_turnover_history = []
        self.episode_position_changes_history = []

        # ------------------------------------------------------
        # Action counts
        # ------------------------------------------------------

        self.short_count = 0
        self.flat_count = 0
        self.long_count = 0
        self.total_count = 0

        self.short_ratio = 0.0
        self.flat_ratio = 0.0
        self.long_ratio = 0.0

        self.episode_short_ratio_history = []
        self.episode_flat_ratio_history = []
        self.episode_long_ratio_history = []

        self.net_return_history = []

        # ------------------------------------------------------
        # Total evaluation statistics
        # ------------------------------------------------------

        self.total_reward = torch.zeros(
            market.batch_size,
            dtype=torch.float32,
            device=market.device
        )

        self.total_profit = torch.zeros(
            market.batch_size,
            dtype=torch.float32,
            device=market.device
        )

        self.total_cost = torch.zeros(
            market.batch_size,
            dtype=torch.float32,
            device=market.device
        )

        self.total_risk = torch.zeros(
            market.batch_size,
            dtype=torch.float32,
            device=market.device
        )

        self.total_turnover = torch.zeros(
            market.batch_size,
            dtype=torch.float32,
            device=market.device
        )



    # ==========================================================
    # Store previous log history
    # ==========================================================

    def _store_previous_history(self):

        price_width = max(self.market.window_size, self.agent4_window) if self.agent_type == 4 else self.market.window_size
        required = price_width + 1

        self.prev_log_price_history = (
            self.market.log_price_seq[
                :,
                -required:
            ]
            .detach()
            .clone()
        )

        self.prev_log_mean_history = (
            self.market.log_mean_seq[
                :,
                -(self.market.window_size + 1):
            ]
            .detach()
            .clone()
        )


    # ==========================================================
    # Log price window
    # ==========================================================

    def _log_price_window(
            self,
            t,
            episode
    ):

        market = self.market


        if self.agent_type == 4:
            w = max(market.window_size, self.agent4_window)
        else:
            w = market.window_size

        required = w + 1

        # ======================================================
        # First evaluation episode
        # ======================================================

        if episode == 0:

            start = max(
                0,
                t - w
            )

            window = (
                market.log_price_seq[
                    :,
                    start:t + 1
                ]
            )

            # --------------------------------------------------
            # Not enough history yet
            # --------------------------------------------------

            if window.shape[1] < required:

                missing = (
                    required
                    - window.shape[1]
                )

                padding = (
                    window[:, :1]
                    .repeat(
                        1,
                        missing
                    )
                )

                window = torch.cat(
                    [
                        padding,
                        window
                    ],
                    dim=1
                )

        # ======================================================
        # Later continuous episodes
        # ======================================================

        else:

            if self.prev_log_price_history is None:

                raise RuntimeError(
                    "Previous evaluation price "
                    "history is missing."
                )

            # --------------------------------------------------
            # t = 0
            #
            # Use complete previous window.
            # --------------------------------------------------

            if t == 0:

                window = (
                    self.prev_log_price_history
                )

            # --------------------------------------------------
            # Beginning of new market segment
            #
            # log_price_seq[:, 0] duplicates the previous
            # final observation.
            #
            # Therefore the first NEW observation is index 1.
            # --------------------------------------------------

            elif t <= w:

                previous_needed = (
                    required - t
                )

                previous = (
                    self.prev_log_price_history[
                        :,
                        -previous_needed:
                    ]
                )

                current = (
                    market.log_price_seq[
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
            # Full new history available
            # --------------------------------------------------

            else:

                window = (
                    market.log_price_seq[
                        :,
                        t - w:t + 1
                    ]
                )

        # ======================================================
        # Safety
        # ======================================================

        if window.dim() != 2:

            raise RuntimeError(
                f"Evaluation price window must "
                f"be 2D, got {window.shape}"
            )

        if window.shape[1] != required:

            raise RuntimeError(
                f"Wrong evaluation price "
                f"window width: "
                f"{window.shape[1]}, "
                f"expected {required}"
            )

        if not torch.isfinite(window).all():

            raise RuntimeError(
                f"Non-finite price window. "
                f"episode={episode}, t={t}"
            )

        return window


    # ==========================================================
    # Log mean window
    # ==========================================================

    def _log_mean_window(
            self,
            t,
            episode
    ):

        market = self.market

        w = market.window_size
        required = w + 1

        # ======================================================
        # First evaluation episode
        # ======================================================

        if episode == 0:

            start = max(
                0,
                t - w
            )

            window = (
                market.log_mean_seq[
                    :,
                    start:t + 1
                ]
            )

            if window.shape[1] < required:

                missing = (
                    required
                    - window.shape[1]
                )

                padding = (
                    window[:, :1]
                    .repeat(
                        1,
                        missing
                    )
                )

                window = torch.cat(
                    [
                        padding,
                        window
                    ],
                    dim=1
                )

        # ======================================================
        # Later continuous episodes
        # ======================================================

        else:

            if self.prev_log_mean_history is None:

                raise RuntimeError(
                    "Previous evaluation mean "
                    "history is missing."
                )

            if t == 0:

                window = (
                    self.prev_log_mean_history
                )

            elif t <= w:

                previous_needed = (
                    required - t
                )

                previous = (
                    self.prev_log_mean_history[
                        :,
                        -previous_needed:
                    ]
                )

                current = (
                    market.log_mean_seq[
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

            else:

                window = (
                    market.log_mean_seq[
                        :,
                        t - w:t + 1
                    ]
                )

        # ======================================================
        # Safety
        # ======================================================

        if window.dim() != 2:

            raise RuntimeError(
                f"Evaluation mean window must "
                f"be 2D, got {window.shape}"
            )

        if window.shape[1] != required:

            raise RuntimeError(
                f"Wrong evaluation mean "
                f"window width: "
                f"{window.shape[1]}, "
                f"expected {required}"
            )

        if not torch.isfinite(window).all():

            raise RuntimeError(
                f"Non-finite mean window. "
                f"episode={episode}, t={t}"
            )

        return window


    # ==========================================================
    # Build state
    # ==========================================================

    def _build_state(
            self,
            t,
            episode
    ):

        market = self.market

        log_price_window = (
            self._log_price_window(
                t,
                episode
            )
        )

        # ======================================================
        # Agent 1
        # ======================================================

        if self.agent_type == 1:

            state = market.build_state(
                log_price_window,
                market.pos
            )

        # ======================================================
        # Agent 2
        # ======================================================

        elif self.agent_type == 2:

            log_mean_window = (
                self._log_mean_window(
                    t,
                    episode
                )
            )

            current_log_mean = (
                log_mean_window[:, -1]
            )

            state = market.build_state(
                log_price_window,
                market.pos,
                current_log_mean
            )

        # ======================================================
        # Agent 3
        #
        # Full market + mean windows
        # ======================================================

        elif self.agent_type == 3:

            log_mean_window = (
                self._log_mean_window(
                    t,
                    episode
                )
            )

            state = market.build_state(
                log_price_window,
                market.pos,
                log_mean_window
            )

        # ======================================================
        # Agent 4
        # ======================================================

        elif self.agent_type == 4:

            state = market.build_state(
                log_price_window,
                market.pos
            )

        else:

            raise ValueError(
                f"Unknown agent_type: "
                f"{self.agent_type}"
            )

        if state.dim() != 2:

            raise RuntimeError(
                f"Evaluation state must "
                f"be 2D, got {state.shape}"
            )

        if not torch.isfinite(state).all():

            raise RuntimeError(
                f"Evaluation state contains "
                f"NaN/Inf. "
                f"episode={episode}, t={t}"
            )

        return state

    # ==========================================================
    # Evaluate
    # ==========================================================

    def evaluate(
            self,
            greedy=True,
            debug=False
    ):

        market = self.market
        agent = self.agent

        # ======================================================
        # Start evaluation from a fresh market
        # ======================================================

        market.full_reset()

        self.prev_log_price_history = None
        self.prev_log_mean_history = None

        self._initialize_statistics()

        agent.model.eval()

        # ======================================================
        # Evaluation episodes
        # ======================================================

        with torch.no_grad():

            for episode in range(self.num_episodes):

                # ==================================================
                # Save current continuous market segment
                # ==================================================

                if episode == 0:

                    # First segment: keep all prices
                    self.price_history.append(
                        market.price_seq
                        .detach()
                        .cpu()
                        .clone()
                    )

                    self.mean_price_history.append(
                        market.mean_price_seq
                        .detach()
                        .cpu()
                        .clone()
                    )

                else:

                    # Later segments:
                    # first observation duplicates
                    # previous segment's final observation
                    self.price_history.append(
                        market.price_seq[:, 1:]
                        .detach()
                        .cpu()
                        .clone()
                    )

                    self.mean_price_history.append(
                        market.mean_price_seq[:, 1:]
                        .detach()
                        .cpu()
                        .clone()
                    )

                # ==================================================
                # Per-episode statistics
                # ==================================================

                episode_reward = torch.zeros(
                    market.batch_size,
                    dtype=torch.float32,
                    device=market.device
                )

                episode_gross_returns = torch.zeros_like(episode_reward)
                episode_transaction_costs = torch.zeros_like(episode_reward)
                episode_risk = torch.zeros_like(episode_reward)
                episode_turnover = torch.zeros_like(episode_reward)
                episode_position_changes = torch.zeros_like(episode_reward)

                episode_short_count = 0
                episode_flat_count = 0
                episode_long_count = 0
                episode_total_count = 0

                # ==================================================
                # Move through market segment
                # ==================================================

                for t in range(market.num_steps - 1):

                    # ==============================================
                    # State
                    # ==============================================

                    state = self._build_state(t,episode)

                    # ==============================================
                    # Greedy action
                    # ==============================================

                    action = agent.act(state,greedy=greedy)

                    # ==============================================
                    # Save action
                    # ==============================================

                    self.action_history.append(
                        action
                        .detach()
                        .cpu()
                    )

                    # ==============================================
                    # Action statistics
                    # ==============================================

                    episode_short_count += (action == 0).sum().item()
                    episode_flat_count += (action == 1).sum().item()
                    episode_long_count += (action == 2).sum().item()

                    episode_total_count += action.numel()

                    # ==============================================
                    # Calculate the reward
                    # ==============================================

                    # Count actual position changes
                    previous_position = market.pos.detach().clone()

                    new_position = action - 1

                    position_changed = (
                            new_position != previous_position
                    ).float()

                    episode_position_changes += position_changed

                    # Keep position before the trade for debug comparison
                    if debug and episode < 2 and t < 5:
                        previous_position = market.pos.detach().clone()

                    reward, done = market.step(action,t)

                    # Net return excludes the artificial risk penalty
                    net_return = (
                            market.last_profit
                            - market.last_transaction_cost
                    )

                    self.net_return_history.append(
                        net_return.detach().cpu().clone()
                    )

                    if debug and episode < 2 and t < 5:
                        self._print_step_debug(
                            episode=episode,
                            t=t,
                            state=state,
                            action=action,
                            previous_position=previous_position,
                            reward=reward,
                        )

                    self.reward_history.append(
                        reward
                        .detach()
                        .cpu()
                    )

                    # ==============================================
                    # Episode statistics
                    # ==============================================

                    episode_reward += reward

                    episode_gross_returns += market.last_profit

                    episode_transaction_costs += market.last_transaction_cost

                    episode_risk += market.last_risk_penalty

                    episode_turnover += market.last_turnover

                    # ==============================================
                    # Total statistics
                    # ==============================================

                    # self.total_reward += reward
                    #
                    # self.total_profit += market.last_profit
                    #
                    # self.total_cost += market.last_transaction_cost
                    #
                    #
                    # self.total_risk += market.last_risk_penalty
                    #
                    # self.total_turnover += market.last_turnover


                # ==================================================
                # Save episode-level results
                # ==================================================

                self.episode_reward_history.append(
                    episode_reward
                    .mean()
                    .item()
                )

                self.episode_profit_history.append(
                    episode_gross_returns
                    .mean()
                    .item()
                )

                self.episode_cost_history.append(
                    episode_transaction_costs
                    .mean()
                    .item()
                )

                self.episode_risk_history.append(
                    episode_risk
                    .mean()
                    .item()
                )

                self.episode_turnover_history.append(
                    episode_turnover
                    .mean()
                    .item()
                )

                self.episode_position_changes_history.append(
                    episode_position_changes
                    .mean()
                    .item()
                )

                self.episode_short_ratio_history.append(
                    episode_short_count / episode_total_count
                )

                self.episode_flat_ratio_history.append(
                    episode_flat_count / episode_total_count
                )

                self.episode_long_ratio_history.append(
                    episode_long_count / episode_total_count
                )

                self.print_episode_results(episode)


                # ==================================================
                # Generate next continuous market segment
                # ==================================================

                if episode < self.num_episodes - 1:
                    self._store_previous_history()

                    market.reset()


        # ======================================================
        # Stack histories
        # ======================================================

        self.action_history = torch.stack(
            self.action_history,
            dim=1
        )

        self.reward_history = torch.stack(
            self.reward_history,
            dim=1
        )

        self.net_return_history = torch.stack(
            self.net_return_history,
            dim=1
        )

        self.price_history = torch.cat(
            self.price_history,
            dim=1
        )

        self.mean_price_history = torch.cat(
            self.mean_price_history,
            dim=1
        )

        mean_short_ratio = np.mean(
            self.episode_short_ratio_history
        ).item()

        mean_flat_ratio = np.mean(
            self.episode_flat_ratio_history
        ).item()

        mean_long_ratio = np.mean(
            self.episode_long_ratio_history
        ).item()

        # ======================================================
        # Average metrics per evaluation episode
        # ======================================================

        mean_reward = np.mean(
            self.episode_reward_history
        ).item()

        mean_profit = np.mean(
            self.episode_profit_history
        ).item()

        mean_cost = np.mean(
            self.episode_cost_history
        ).item()

        mean_risk = np.mean(
            self.episode_risk_history
        ).item()

        mean_turnover = np.mean(
            self.episode_turnover_history
        ).item()

        mean_position_changes = np.mean(
            self.episode_position_changes_history
        ).item()


        return {
            "num_episodes": self.num_episodes,

            "reward": mean_reward,
            "profit": mean_profit,
            "cost": mean_cost,
            "risk": mean_risk,
            "turnover": mean_turnover,
            "position_changes": mean_position_changes,

            "short": mean_short_ratio,
            "flat": mean_flat_ratio,
            "long": mean_long_ratio,
        }


    # ==========================================================
    # Debug: 5 steps x 2 episodes x 2 batch trajectories
    # ==========================================================

    def _print_step_debug(
            self, episode, t, state, action,
            previous_position, reward
    ):
        market = self.market
        action_names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        n = min(2, market.batch_size)

        if t == 0:
            print(f"\n{'=' * 78}")
            print(f"DEBUG EPISODE {episode + 1} | FIRST 5 STEPS | FIRST {n} BATCH MEMBERS")
            print(f"{'=' * 78}")

        print(f"\nEpisode {episode + 1}, step {t + 1}")
        for b in range(n):
            a = int(action.reshape(-1)[b].item())
            old_pos = float(previous_position.reshape(-1)[b].item())
            new_pos = float(market.pos.reshape(-1)[b].item())
            price_now = float(market.price_seq[b, t].item())
            price_next = float(market.price_seq[b, t + 1].item())
            mean_now = float(market.mean_price_seq[b, t].item())

            print(
                f"  Batch {b} | action={a} ({action_names.get(a, 'UNKNOWN')}) "
                f"| position={old_pos:+.0f}->{new_pos:+.0f} "
                f"| price={price_now:.6f}->{price_next:.6f} "
                f"| mean={mean_now:.6f}"
            )
            print(
                f"          reward={reward.reshape(-1)[b].item():+.6f} "
                f"| gross={market.last_profit.reshape(-1)[b].item():+.6f} "
                f"| cost={market.last_transaction_cost.reshape(-1)[b].item():.6f} "
                f"| risk={market.last_risk_penalty.reshape(-1)[b].item():.6f} "
                f"| turnover={market.last_turnover.reshape(-1)[b].item():.2f}"
            )
            print(
                f"          state_shape={tuple(state.shape)} "
                f"| state_first5={state[b].reshape(-1)[:5].detach().cpu().tolist()}"
            )

    def print_episode_results(self, episode):

        print(
            f"Evaluation Episode {episode + 1:4d} | "
            f"Reward: {self.episode_reward_history[-1]:10.4f} | "
            f"Profit: {self.episode_profit_history[-1]:10.4f} | "
            f"Cost: {self.episode_cost_history[-1]:10.4f} | "
            f"Risk: {self.episode_risk_history[-1]:10.4f} | "
            f"Turnover: {self.episode_turnover_history[-1]:10.2f} | "
            f"Short: {self.episode_short_ratio_history[-1]:6.1%} | "
            f"Flat: {self.episode_flat_ratio_history[-1]:6.1%} | "
            f"Long: {self.episode_long_ratio_history[-1]:6.1%}"
        )

    # ==========================================================
    # Plot reward across evaluation episodes
    # ==========================================================

    def plot_episode_reward(self):

        rewards = np.asarray(
            self.episode_reward_history
        )

        episodes = np.arange(
            1,
            len(rewards) + 1
        )

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            episodes,
            rewards,
            linewidth=1.5
        )

        plt.axhline(
            0,
            linewidth=1
        )

        plt.xlabel(
            "Evaluation Episode"
        )

        plt.ylabel(
            "Reward"
        )

        plt.title(
            "Greedy Policy - Reward per Evaluation Episode"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()



    def plot_price_with_positions(
            self,
            episode_idx=0,
            batch_idx=0
    ):

        # ==================================================
        # Safety
        # ==================================================

        episode_idx = max(
            0,
            min(
                episode_idx,
                self.num_episodes - 1
            )
        )

        batch_idx = max(
            0,
            min(
                batch_idx,
                self.batch_size - 1
            )
        )

        # ==================================================
        # Entire continuous path for selected batch
        # ==================================================

        all_prices = (
            self.price_history[
                batch_idx
            ]
            .numpy()
        )

        all_mean_prices = (
            self.mean_price_history[
                batch_idx
            ]
            .numpy()
        )

        all_actions = (
            self.action_history[
                batch_idx
            ]
            .numpy()
        )

        # ==================================================
        # Find selected episode in continuous history
        # ==================================================

        steps_per_episode = (
                self.market.num_steps - 1
        )

        price_start = (
                episode_idx
                * steps_per_episode
        )

        price_end = (
                price_start
                + self.market.num_steps
        )

        action_start = (
                episode_idx
                * steps_per_episode
        )

        action_end = (
                action_start
                + steps_per_episode
        )

        # ==================================================
        # Extract selected episode
        # ==================================================

        prices = all_prices[
            price_start:price_end
        ]

        mean_prices = all_mean_prices[
            price_start:price_end
        ]

        actions = all_actions[
            action_start:action_end
        ]

        # ==================================================
        # Convert actions to positions
        # ==================================================

        action_positions = (
                actions - 1
        )

        # Position entering this episode
        if episode_idx == 0:

            initial_position = 0

        else:

            initial_position = (
                    all_actions[
                        action_start - 1
                        ]
                    - 1
            )

        positions = np.concatenate(
            [
                np.array(
                    [initial_position]
                ),
                action_positions
            ]
        )

        # print(
        #     "Prices:",
        #     prices.shape
        # )
        #
        # print(
        #     "Mean prices:",
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

        # ==================================================
        # Position changes
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
                positions
                != prev_positions
        )

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
            figsize=(14, 6)
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

        plt.scatter(
            long_idx,
            prices[long_idx],
            color="green",
            marker="^",
            s=80,
            label="Long"
        )

        plt.scatter(
            short_idx,
            prices[short_idx],
            color="red",
            marker="v",
            s=80,
            label="Short"
        )

        plt.scatter(
            close_idx,
            prices[close_idx],
            color="black",
            marker="v",
            s=80,
            label="Close"
        )

        plt.xlabel(
            "Time step within episode"
        )

        plt.ylabel(
            "Price"
        )

        plt.title(
            f"Evaluation - Agent {self.config.agent_type}"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.show()

    def plot_full_price_with_positions(
            self,
            batch_idx=0
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        def to_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        # ==================================================
        # 1. Validation
        # ==================================================

        if isinstance(self.action_history, list):
            raise RuntimeError(
                "Run evaluate() before plotting."
            )

        batch_idx = max(
            0,
            min(batch_idx, self.batch_size - 1)
        )

        # ==================================================
        # 2. Entire continuous path for selected batch
        # ==================================================

        prices = to_numpy(
            self.price_history[batch_idx]
        ).reshape(-1)

        mean_prices = to_numpy(
            self.mean_price_history[batch_idx]
        ).reshape(-1)

        actions = to_numpy(
            self.action_history[batch_idx]
        ).reshape(-1)

        # ==================================================
        # 3. Convert actions to positions
        # ==================================================

        # 0 = SHORT -> -1
        # 1 = FLAT  ->  0
        # 2 = LONG  -> +1

        action_positions = actions.astype(np.int64) - 1

        # Evaluation starts FLAT
        initial_position = 0

        positions = np.concatenate(
            [
                np.array([initial_position]),
                action_positions
            ]
        )

        # ==================================================
        # 4. Dimension check
        # ==================================================

        if not (
                len(prices)
                == len(mean_prices)
                == len(positions)
        ):
            raise ValueError(
                f"Dimension mismatch: "
                f"prices={len(prices)}, "
                f"mean_prices={len(mean_prices)}, "
                f"positions={len(positions)}"
            )

        # ==================================================
        # 5. Identify BUY and SELL signals
        # ==================================================

        previous_positions = np.concatenate(
            [
                np.array([initial_position]),
                positions[:-1]
            ]
        )

        changed = positions != previous_positions

        # BUY:
        # - Enter LONG (FLAT -> LONG)
        # - Close SHORT (SHORT -> FLAT)
        # - Reverse SHORT -> LONG

        long_idx = np.flatnonzero(
            changed & (
                    (positions == 1) |
                    (
                            (previous_positions == -1) &
                            (positions == 0)
                    )
            )
        )

        # SELL:
        # - Enter SHORT (FLAT -> SHORT)
        # - Close LONG (LONG -> FLAT)
        # - Reverse LONG -> SHORT

        short_idx = np.flatnonzero(
            changed & (
                    (positions == -1) |
                    (
                            (previous_positions == 1) &
                            (positions == 0)
                    )
            )
        )

        # ==================================================
        # 6. Plot market and fundamental prices
        # ==================================================

        t = np.arange(len(prices))

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            t,
            prices,
            color="tab:blue",
            linewidth=1.5,
            label="Market price"
        )

        ax.plot(
            t,
            mean_prices,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Fundamental price"
        )

        # ==================================================
        # 7. Plot trading signals
        # ==================================================

        # BUY / Close SHORT
        ax.scatter(
            long_idx,
            prices[long_idx],
            color="green",
            marker="^",
            s=70,
            label="BUY",
            zorder=3
        )

        # SELL / Close LONG
        ax.scatter(
            short_idx,
            prices[short_idx],
            color="red",
            marker="v",
            s=70,
            label="SELL",
            zorder=3
        )

        # ==================================================
        # 8. Formatting
        # ==================================================

        ax.set_xlabel("Continuous Time Step")
        ax.set_ylabel("Price")

        ax.set_title(
            f"Full Price Trajectory - Agent {self.config.agent_type}"
        )

        ax.legend(
            loc="upper left",
            fontsize=10
        )

        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()

        return fig, ax

    def summary_tables2(self, print_tables=True):
        import numpy as np
        import pandas as pd

        def as_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def statistic(values, operation):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(operation(values)) if values.size else np.nan

        # ==========================================================
        # 1. Validation
        # ==========================================================

        if len(self.episode_reward_history) == 0:
            raise RuntimeError(
                "Run evaluate() before calling summary_tables2()."
            )

        if isinstance(self.action_history, list):
            raise RuntimeError(
                "Call this method after evaluate() finishes."
            )

        # ==========================================================
        # 2. Episode performance statistics
        # ==========================================================

        reward = np.asarray(
            self.episode_reward_history,
            dtype=float
        )

        profit = np.asarray(
            self.episode_profit_history,
            dtype=float
        )

        cost = np.asarray(
            self.episode_cost_history,
            dtype=float
        )

        risk = np.asarray(
            self.episode_risk_history,
            dtype=float
        )

        turnover = np.asarray(
            self.episode_turnover_history,
            dtype=float
        )

        position_changes = np.asarray(
            self.episode_position_changes_history,
            dtype=float
        )

        metrics = {
            "Reward": reward,
            "Gross profit": profit,
            "Transaction cost": cost,
            "Net profit (gross - cost)": profit - cost,
            "Risk penalty": risk,
            "Action turnover": turnover,
            "Position changes": position_changes,
        }

        statistics = pd.DataFrame(
            [
                {
                    "Metric": name,
                    "Mean per episode": values.mean(),
                    "Std across episodes": values.std(ddof=0),
                    "Min episode": values.min(),
                    "Max episode": values.max(),
                    "Total per trajectory": values.sum(),
                }
                for name, values in metrics.items()
            ]
        ).set_index("Metric")

        # ==========================================================
        # 3. Risk-adjusted performance statistics
        # ==========================================================

        sharpe = self.calculate_sharpe_ratio()

        sortino = self.calculate_sortino_ratio(
            target_return=0.0,
            periods_per_year=52,
        )

        volatility = self.calculate_volatility()

        risk_metrics = {
            "Annualized Sharpe ratio": sharpe,
            "Annualized Sortino ratio": sortino,
            "Reward volatility": volatility["reward"],
            "Net return volatility": volatility["net_return"],
            "Annualized reward volatility": volatility["reward_annual"],
            "Annualized net return volatility": volatility["net_return_annual"],
        }

        sharpe_summary = pd.DataFrame(
            [
                {
                    "Metric": name,
                    "Mean": values["mean"],
                    "Std": values["std"],
                    "Min": values["min"],
                    "Max": values["max"],
                }
                for name, values in risk_metrics.items()
            ]
        ).set_index("Metric")

        # ==========================================================
        # 4. Prepare market prices and actions
        # ==========================================================

        prices = as_numpy(self.price_history)
        actions = as_numpy(self.action_history)
        fundamental_prices = as_numpy(
            self.mean_price_history
        )

        if prices.ndim != 2:
            raise ValueError(
                "Expected price_history with shape [batch, time]."
            )

        batch_size, price_steps = prices.shape

        if actions.size != batch_size * (price_steps - 1):
            raise ValueError(
                "Action and price histories are not aligned."
            )

        actions = actions.reshape(
            batch_size,
            price_steps - 1
        )

        if not np.isin(actions, [0, 1, 2]).all():
            raise ValueError(
                "Expected action codes: 0=SHORT, 1=FLAT, 2=LONG."
            )

        if fundamental_prices.shape != prices.shape:
            raise ValueError(
                "Fundamental and market price histories must align."
            )

        decision_prices = prices[:, :-1]
        decision_fundamentals = fundamental_prices[:, :-1]

        if not (
                np.isfinite(decision_prices).all()
                and np.isfinite(decision_fundamentals).all()
        ):
            raise ValueError(
                "Market or fundamental prices contain NaN/Inf."
            )

        # ==========================================================
        # 5. Price deviations
        # ==========================================================

        deviation = (
                decision_prices
                - decision_fundamentals
        )

        relative_deviation = np.full_like(
            deviation,
            np.nan,
            dtype=float
        )

        np.divide(
            100.0 * deviation,
            decision_fundamentals,
            out=relative_deviation,
            where=decision_fundamentals != 0,
        )

        # ==========================================================
        # 6. Position changes
        # ==========================================================

        previous_actions = np.concatenate(
            [
                np.full(
                    (batch_size, 1),
                    1,
                    dtype=actions.dtype
                ),
                actions[:, :-1],
            ],
            axis=1
        )

        changed = actions != previous_actions

        # ==========================================================
        # 7. Action-specific statistics
        # ==========================================================

        rows = []

        for code, name in enumerate(
                ["SHORT", "FLAT", "LONG"]
        ):
            mask = actions == code

            differences = deviation[mask]
            percentages = relative_deviation[mask]

            change_differences = deviation[
                mask & changed
                ]

            rows.append({
                "Action": name,

                "Count": int(mask.sum()),

                "Action ratio (%)": (
                        100.0 * mask.mean()
                ),

                "Deviation min": statistic(
                    differences, np.min
                ),

                "Deviation max": statistic(
                    differences, np.max
                ),

                "E[deviation | action]": statistic(
                    differences, np.mean
                ),

                "Relative deviation min (%)": statistic(
                    percentages, np.min
                ),

                "Relative deviation max (%)": statistic(
                    percentages, np.max
                ),

                "E[relative deviation (%) | action]": statistic(
                    percentages, np.mean
                ),

                "Position changes": int(
                    change_differences.size
                ),

                "Change deviation min": statistic(
                    change_differences, np.min
                ),

                "Change deviation max": statistic(
                    change_differences, np.max
                ),

                "E[deviation | action, position changed]": statistic(
                    change_differences, np.mean
                ),
            })

        action_summary = pd.DataFrame(
            rows
        ).set_index("Action")

        # ==========================================================
        # 8. Print results
        # ==========================================================

        if print_tables:
            print("\n" + "=" * 85)
            print("EVALUATION PERFORMANCE STATISTICS")
            print("=" * 85)

            print(
                statistics.to_string(
                    float_format=lambda x: f"{x:,.6f}"
                )
            )

            print("\n" + "=" * 85)
            print("RISK-ADJUSTED PERFORMANCE — ACROSS TRAJECTORIES")
            print("=" * 85)

            print(
                sharpe_summary.to_string(
                    float_format=lambda x: f"{x:,.6f}"
                )
            )

            print("\n" + "=" * 85)
            print("ACTION-SPECIFIC PRICE DEVIATION STATISTICS")
            print("=" * 85)

            print(
                action_summary.to_string(
                    float_format=lambda x: f"{x:,.4f}"
                )
            )

        # ==========================================================
        # 9. Return results
        # ==========================================================

        return statistics, sharpe_summary, action_summary

    def summary_tables(self, print_tables=True):
        import numpy as np
        import pandas as pd

        def as_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def statistic(values, operation):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(operation(values)) if values.size else np.nan

        if len(self.episode_reward_history) == 0:
            raise RuntimeError(
                "Run evaluate() before calling summary_tables()."
            )

        # ==========================================================
        # 1. Evaluation performance statistics
        # ==========================================================

        reward = np.asarray(self.episode_reward_history, dtype=float)
        profit = np.asarray(self.episode_profit_history, dtype=float)
        cost = np.asarray(self.episode_cost_history, dtype=float)
        risk = np.asarray(self.episode_risk_history, dtype=float)
        turnover = np.asarray(self.episode_turnover_history, dtype=float)

        metrics = {
            "Reward": reward,
            "Gross profit": profit,
            "Transaction cost": cost,
            "Net profit (gross - cost)": profit - cost,
            "Risk penalty": risk,
            "Action turnover": turnover,
        }

        statistics = pd.DataFrame(
            [
                {
                    "Metric": name,
                    "Mean per episode": values.mean(),
                    "Std across episodes": values.std(ddof=0),
                    "Min episode": values.min(),
                    "Max episode": values.max(),
                    "Total per trajectory": values.sum(),
                }
                for name, values in metrics.items()
            ]
        ).set_index("Metric")

        # ==========================================================
        # 2. Prepare market prices and actions
        # ==========================================================

        if isinstance(self.action_history, list):
            raise RuntimeError(
                "Call this method after evaluate() finishes."
            )

        prices = as_numpy(self.price_history)
        actions = as_numpy(self.action_history)
        fundamentals = as_numpy(self.mean_price_history)

        if prices.ndim != 2:
            raise ValueError(
                "Expected price_history with shape [batch, time]."
            )

        batch_size, price_steps = prices.shape

        if actions.size != batch_size * (price_steps - 1):
            raise ValueError(
                "Action and price histories are not aligned."
            )

        actions = actions.reshape(batch_size, price_steps - 1)

        if fundamentals.shape != prices.shape:
            raise ValueError(
                "Fundamental and market price histories must align."
            )

        if not np.isin(actions, [0, 1, 2]).all():
            raise ValueError(
                "Expected action codes: 0=SHORT, 1=FLAT, 2=LONG."
            )

        # Prices observed BEFORE selecting each action
        decision_prices = prices[:, :-1]
        decision_fundamentals = fundamentals[:, :-1]

        if not (
                np.isfinite(decision_prices).all()
                and np.isfinite(decision_fundamentals).all()
        ):
            raise ValueError(
                "Market or fundamental prices contain NaN/Inf."
            )

        # ==========================================================
        # 3. Calculate price deviations
        # ==========================================================

        # Absolute price deviation
        deviation = decision_prices - decision_fundamentals

        # Relative deviation in percentage
        # Positive: price above fundamental
        # Negative: price below fundamental

        relative_deviation = np.full_like(
            deviation,
            np.nan,
            dtype=float
        )

        np.divide(
            100.0 * deviation,
            decision_fundamentals,
            out=relative_deviation,
            where=decision_fundamentals != 0
        )

        # ==========================================================
        # 4. Identify position changes
        # ==========================================================

        # Evaluation starts from FLAT (action = 1).
        # Positions continue across evaluation episodes.

        previous_actions = np.concatenate(
            [
                np.full(
                    (batch_size, 1),
                    1,
                    dtype=actions.dtype
                ),
                actions[:, :-1]
            ],
            axis=1
        )

        changed = actions != previous_actions

        # ==========================================================
        # 5. Action-specific price deviation statistics
        # ==========================================================

        action_names = {
            0: "SHORT",
            1: "FLAT",
            2: "LONG"
        }

        rows = []

        for code, name in action_names.items():
            mask = actions == code
            change_mask = mask & changed

            abs_dev = deviation[mask]
            rel_dev = relative_deviation[mask]

            change_abs_dev = deviation[change_mask]
            change_rel_dev = relative_deviation[change_mask]

            rows.append({
                "Action": name,

                "Count": int(mask.sum()),
                "Action ratio (%)": 100.0 * mask.mean(),

                # Absolute deviation statistics
                "Deviation mean": statistic(abs_dev, np.mean),
                "Deviation median": statistic(abs_dev, np.median),
                "Deviation P10": statistic(
                    abs_dev, lambda x: np.percentile(x, 10)
                ),
                "Deviation P90": statistic(
                    abs_dev, lambda x: np.percentile(x, 90)
                ),

                # Relative deviation statistics
                "Relative mean (%)": statistic(rel_dev, np.mean),
                "Relative median (%)": statistic(rel_dev, np.median),
                "Relative P10 (%)": statistic(
                    rel_dev, lambda x: np.percentile(x, 10)
                ),
                "Relative P90 (%)": statistic(
                    rel_dev, lambda x: np.percentile(x, 90)
                ),
                "Relative min (%)": statistic(rel_dev, np.min),
                "Relative max (%)": statistic(rel_dev, np.max),

                # Only actual position changes
                "Position changes": int(change_mask.sum()),
                "Change relative mean (%)": statistic(
                    change_rel_dev, np.mean
                ),
                "Change relative P10 (%)": statistic(
                    change_rel_dev,
                    lambda x: np.percentile(x, 10)
                ),
                "Change relative P90 (%)": statistic(
                    change_rel_dev,
                    lambda x: np.percentile(x, 90)
                ),
                "Change deviation mean": statistic(
                    change_abs_dev, np.mean
                ),
            })

        action_summary = pd.DataFrame(rows).set_index("Action")

        # ==========================================================
        # 6. Action distribution by price-deviation interval
        # ==========================================================

        # Bins represent percentage deviation from fundamental.
        # Example: [-5, -2) means market price is 2%-5%
        # below fundamental price.

        bins = np.array([
            -np.inf,
            -20,
            -10,
            -5,
            -2,
            0,
            2,
            5,
            10,
            20,
            np.inf
        ])

        labels = [
            "< -20%",
            "-20% to -10%",
            "-10% to -5%",
            "-5% to -2%",
            "-2% to 0%",
            "0% to 2%",
            "2% to 5%",
            "5% to 10%",
            "10% to 20%",
            ">= 20%"
        ]

        valid = np.isfinite(relative_deviation)

        flat_deviation = relative_deviation[valid]
        flat_actions = actions[valid]

        # Assign each observation to a deviation interval.
        interval_index = np.digitize(
            flat_deviation,
            bins[1:-1],
            right=False
        )

        interval_rows = []

        for i, label in enumerate(labels):

            interval_mask = interval_index == i
            interval_actions = flat_actions[interval_mask]

            count = interval_actions.size

            if count == 0:
                short_ratio = np.nan
                flat_ratio = np.nan
                long_ratio = np.nan
            else:
                short_ratio = 100.0 * np.mean(
                    interval_actions == 0
                )
                flat_ratio = 100.0 * np.mean(
                    interval_actions == 1
                )
                long_ratio = 100.0 * np.mean(
                    interval_actions == 2
                )

            interval_rows.append({
                "Price deviation": label,
                "Observations": count,
                "Short (%)": short_ratio,
                "Flat (%)": flat_ratio,
                "Long (%)": long_ratio,
            })

        deviation_summary = pd.DataFrame(
            interval_rows
        ).set_index("Price deviation")

        # ==========================================================
        # 7. Print summary tables
        # ==========================================================

        if print_tables:
            print("\n" + "=" * 85)
            print("EVALUATION PERFORMANCE STATISTICS")
            print("=" * 85)

            print(
                statistics.to_string(
                    float_format=lambda x: f"{x:,.6f}"
                )
            )

            print("\n" + "=" * 85)
            print("ACTION-SPECIFIC PRICE DEVIATION STATISTICS")
            print("=" * 85)

            print(
                action_summary.to_string(
                    float_format=lambda x: f"{x:,.4f}"
                )
            )

            print("\n" + "=" * 85)
            print("ACTION DISTRIBUTION BY RELATIVE PRICE DEVIATION")
            print("=" * 85)

            print(
                deviation_summary.to_string(
                    float_format=lambda x: f"{x:,.2f}"
                )
            )

        return statistics, action_summary, deviation_summary

    def plot_price_with_positions2(self, episode_idx=0, batch_idx=0):
        import numpy as np
        import matplotlib.pyplot as plt

        def to_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        # ======================================================
        # 1. Validation
        # ======================================================

        if isinstance(self.action_history, list):
            raise RuntimeError("Run evaluate() before plotting.")

        all_prices = to_numpy(self.price_history)
        all_means = to_numpy(self.mean_price_history)
        all_actions = to_numpy(self.action_history)

        if not 0 <= episode_idx < self.num_episodes:
            raise ValueError("episode_idx is outside the evaluation range.")

        if not 0 <= batch_idx < all_prices.shape[0]:
            raise ValueError("batch_idx is outside the batch range.")

        # ======================================================
        # 2. Extract continuous evaluation history
        # ======================================================

        steps_per_episode = self.market.num_steps - 1

        # Include episodes 0 through episode_idx.
        # Duplicate boundary prices are already removed.
        end = (episode_idx + 1) * steps_per_episode

        prices = all_prices[batch_idx, :end + 1]
        mean_prices = all_means[batch_idx, :end + 1]
        actions = all_actions[batch_idx].reshape(-1)[:end]

        if (
                len(prices) != end + 1
                or len(mean_prices) != end + 1
                or len(actions) != end
        ):
            raise ValueError(
                "Price and action histories are not aligned."
            )

        # ======================================================
        # 3. Convert actions to positions
        # ======================================================

        # 0 = SHORT -> -1
        # 1 = FLAT  ->  0
        # 2 = LONG  -> +1

        positions = actions.astype(np.int64) - 1

        # Initially FLAT; positions persist across episodes.
        previous_positions = np.concatenate(
            ([0], positions[:-1])
        )

        changed = positions != previous_positions

        # ======================================================
        # 4. Identify BUY and SELL signals
        # ======================================================

        # BUY signal:
        # - Enter LONG
        # - Close SHORT (SHORT -> FLAT)
        # - Reverse SHORT -> LONG

        long_idx = np.flatnonzero(
            changed & (
                    (positions == 1) |
                    (
                            (previous_positions == -1) &
                            (positions == 0)
                    )
            )
        )

        # SELL signal:
        # - Enter SHORT
        # - Close LONG (LONG -> FLAT)
        # - Reverse LONG -> SHORT

        short_idx = np.flatnonzero(
            changed & (
                    (positions == -1) |
                    (
                            (previous_positions == 1) &
                            (positions == 0)
                    )
            )
        )

        # ======================================================
        # 5. Create price plot
        # ======================================================

        fig, ax = plt.subplots(figsize=(14, 6))

        t = np.arange(len(prices))

        # Market price
        ax.plot(
            t,
            prices,
            color="tab:blue",
            linewidth=1.5,
            label="Market price",
        )

        # Fundamental price
        ax.plot(
            t,
            mean_prices,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Fundamental price",
        )

        # ======================================================
        # 6. Plot trading signals
        # ======================================================

        # BUY: Enter LONG or close SHORT
        ax.scatter(
            long_idx,
            prices[long_idx],
            color="green",
            marker="^",
            s=70,
            label="BUY",
            zorder=3,
        )

        # SELL: Enter SHORT or close LONG
        ax.scatter(
            short_idx,
            prices[short_idx],
            color="red",
            marker="v",
            s=70,
            label="SELL",
            zorder=3,
        )

        # ======================================================
        # 7. Mark episode boundaries
        # ======================================================

        for episode in range(1, episode_idx + 1):
            ax.axvline(
                episode * steps_per_episode,
                color="gray",
                linestyle=":",
                alpha=0.4,
            )

        # ======================================================
        # 8. Formatting
        # ======================================================

        ax.set_xlabel("Continuous Time Step")
        ax.set_ylabel("Price")

        ax.set_title(
            f"Evaluation — Agent {self.config.agent_type} | "
            f"Trajectory {batch_idx} | "
            f"Episodes 1-{episode_idx + 1}"
        )

        ax.legend(
            loc="upper left",
            fontsize=10
        )

        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()

        return fig, ax

    def calculate_sharpe_ratio(
            self,
            risk_free_rate=0.0
    ):
        import numpy as np

        if len(self.net_return_history) == 0:
            raise RuntimeError(
                "Run evaluate() before calculating Sharpe ratio."
            )

        returns = np.asarray(
            self.net_return_history,
            dtype=float
        )

        # Shape: [batch_size, total_time_steps]
        if returns.ndim != 2:
            raise ValueError(
                "Expected returns with shape [batch, time]."
            )

        if not np.isfinite(returns).all():
            raise ValueError("Net returns contain NaN or Inf.")

        if returns.shape[1] < 2:
            raise ValueError("At least two returns are required.")

        rf_period = (
                (1 + risk_free_rate) ** (1 / self.periods_per_year)
                - 1
        )

        excess_returns = returns - rf_period

        mean_returns = np.mean(
            excess_returns,
            axis=1
        )

        std_returns = np.std(
            returns,
            axis=1,
            ddof=1
        )

        sharpe_ratios = np.full(
            returns.shape[0],
            np.nan
        )

        valid = std_returns > 0

        sharpe_ratios[valid] = (
                np.sqrt(self.periods_per_year)
                * mean_returns[valid]
                / std_returns[valid]
        )

        valid_sharpe = sharpe_ratios[
            np.isfinite(sharpe_ratios)
        ]

        if valid_sharpe.size == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan
            }

        return {
            "mean": float(np.mean(valid_sharpe)),
            "std": float(np.std(valid_sharpe)),
            "min": float(np.min(valid_sharpe)),
            "max": float(np.max(valid_sharpe))
        }

    def calculate_volatility(self):
        import numpy as np

        def as_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        if (
                isinstance(self.reward_history, list)
                or isinstance(self.net_return_history, list)
        ):
            raise RuntimeError(
                "Run evaluate() before calculating volatility."
            )

        reward = as_numpy(self.reward_history)
        net_return = as_numpy(self.net_return_history)

        if reward.shape != net_return.shape:
            raise ValueError(
                "Reward and net return histories must have identical shapes."
            )

        if reward.ndim != 2 or reward.shape[1] < 2:
            raise ValueError(
                "Expected histories with shape [batch, time], "
                "with at least two time steps."
            )

        if not (
                np.isfinite(reward).all()
                and np.isfinite(net_return).all()
        ):
            raise ValueError(
                "Reward or net return contains NaN/Inf."
            )

        # ======================================================
        # 1. Volatility for each trajectory
        # ======================================================

        reward_volatility = np.std(
            reward,
            axis=1,
            ddof=1
        )

        net_return_volatility = np.std(
            net_return,
            axis=1,
            ddof=1
        )

        # ======================================================
        # 2. Annualized volatility
        # ======================================================

        annualization = np.sqrt(self.periods_per_year)

        reward_volatility_annual = (
                reward_volatility * annualization
        )

        net_return_volatility_annual = (
                net_return_volatility * annualization
        )

        # ======================================================
        # 3. Summary statistics
        # ======================================================

        def summarize(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "reward": summarize(reward_volatility),
            "net_return": summarize(net_return_volatility),
            "reward_annual": summarize(reward_volatility_annual),
            "net_return_annual": summarize(net_return_volatility_annual),
        }

    def plot_price_with_positions3(
            self,
            episode_idx=0,
            batch_idx=0
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        def to_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        # ======================================================
        # 1. Validation
        # ======================================================

        if isinstance(self.action_history, list):
            raise RuntimeError(
                "Run evaluate() before plotting."
            )

        prices_all = to_numpy(self.price_history)
        means_all = to_numpy(self.mean_price_history)
        actions_all = to_numpy(self.action_history)

        if not 0 <= episode_idx < self.num_episodes:
            raise ValueError("Invalid episode_idx.")

        if not 0 <= batch_idx < prices_all.shape[0]:
            raise ValueError("Invalid batch_idx.")

        # ======================================================
        # 2. Extract continuous evaluation history
        # ======================================================

        steps_per_episode = self.market.num_steps - 1

        end = (episode_idx + 1) * steps_per_episode

        prices = prices_all[batch_idx, :end + 1]
        means = means_all[batch_idx, :end + 1]

        actions = (
            actions_all[batch_idx]
            .reshape(-1)[:end]
        )

        if not (
                len(prices) == len(means) == len(actions) + 1
        ):
            raise ValueError(
                "Price and action histories are not aligned."
            )

        if np.any(means <= 0):
            raise ValueError(
                "Fundamental prices must be positive."
            )

        # ======================================================
        # 3. Relative market price deviation (%)
        # ======================================================

        relative_deviation = (
                                     (prices - means) / means
                             ) * 100.0

        # ======================================================
        # 4. Convert actions to positions
        # ======================================================

        # 0 = SHORT -> -1
        # 1 = FLAT  ->  0
        # 2 = LONG  -> +1

        positions = actions.astype(int) - 1

        previous_positions = np.concatenate(
            ([0], positions[:-1])
        )

        changed = positions != previous_positions

        # Enter LONG or close SHORT
        long_idx = np.flatnonzero(
            changed & (
                    (positions == 1) |
                    ((previous_positions == -1) & (positions == 0))
            )
        )

        # Enter SHORT or close LONG
        short_idx = np.flatnonzero(
            changed & (
                    (positions == -1) |
                    ((previous_positions == 1) & (positions == 0))
            )
        )

        # ======================================================
        # 5. Create three plots
        # ======================================================

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(15, 11),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2, 2, 1]
            }
        )

        t = np.arange(len(prices))
        action_t = np.arange(len(actions))

        # ======================================================
        # Plot 1: Market and fundamental prices
        # ======================================================

        ax = axes[0]

        ax.plot(
            t,
            prices,
            label="Market price",
            color="tab:blue",
            linewidth=1.5
        )

        ax.plot(
            t,
            means,
            label="Fundamental price",
            color="tab:orange",
            linestyle="--",
            linewidth=1.5
        )

        ax.set_ylabel("Price")
        ax.set_title(
            "Market Price vs Fundamental Price"
        )

        ax.legend()
        ax.grid(alpha=0.3)

        # ======================================================
        # Plot 2: Relative price deviation and actions
        # ======================================================

        ax = axes[1]

        ax.plot(
            t,
            relative_deviation,
            color="tab:blue",
            linewidth=1.2,
            label="Relative price deviation"
        )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Fundamental price level"
        )

        ax.fill_between(
            t,
            relative_deviation,
            0,
            where=relative_deviation >= 0,
            color="red",
            alpha=0.08
        )

        ax.fill_between(
            t,
            relative_deviation,
            0,
            where=relative_deviation < 0,
            color="green",
            alpha=0.08
        )

        # LONG signal:
        # Enter LONG or close SHORT
        ax.scatter(
            long_idx,
            relative_deviation[long_idx],
            color="green",
            marker="^",
            s=65,
            label="LONG",
            zorder=3
        )

        # SHORT signal:
        # Enter SHORT or close LONG
        ax.scatter(
            short_idx,
            relative_deviation[short_idx],
            color="red",
            marker="v",
            s=65,
            label="SHORT",
            zorder=3
        )

        ax.set_ylabel("Relative Deviation (%)")
        ax.set_title(
            "Relative Price Deviation and Trading Decisions"
        )

        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.3)

        # ======================================================
        # Plot 3: Agent position
        # ======================================================

        ax = axes[2]

        ax.step(
            action_t,
            positions,
            where="post",
            color="tab:purple",
            linewidth=1.5,
            label="Agent position"
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8
        )

        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([
            "SHORT",
            "FLAT",
            "LONG"
        ])

        ax.set_ylim(-1.3, 1.3)
        ax.set_ylabel("Position")
        ax.set_xlabel("Continuous Time Step")
        ax.set_title("Trading Position")

        ax.grid(alpha=0.3)

        # ======================================================
        # 6. Mark episode boundaries
        # ======================================================

        for episode in range(1, episode_idx + 1):

            boundary = episode * steps_per_episode

            for ax in axes:
                ax.axvline(
                    boundary,
                    color="gray",
                    linestyle=":",
                    alpha=0.4
                )

        # ======================================================
        # 7. Display
        # ======================================================

        fig.suptitle(
            f"Agent {self.agent_type} | "
            f"Trajectory {batch_idx} | "
            f"Episodes 1-{episode_idx + 1}",
            fontsize=14
        )

        fig.tight_layout()
        plt.show()

        return fig, axes

    def plot_price_with_positions4(
            self,
            episode_idx=0,
            batch_idx=0
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        def to_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        # ======================================================
        # 1. Validation
        # ======================================================

        if isinstance(self.action_history, list):
            raise RuntimeError(
                "Run evaluate() before plotting."
            )

        prices_all = to_numpy(self.price_history)
        means_all = to_numpy(self.mean_price_history)
        actions_all = to_numpy(self.action_history)

        if not 0 <= episode_idx < self.num_episodes:
            raise ValueError("Invalid episode_idx.")

        if not 0 <= batch_idx < prices_all.shape[0]:
            raise ValueError("Invalid batch_idx.")

        # ======================================================
        # 2. Extract continuous evaluation history
        # ======================================================

        steps_per_episode = self.market.num_steps - 1

        end = (episode_idx + 1) * steps_per_episode

        prices = prices_all[batch_idx, :end + 1]
        means = means_all[batch_idx, :end + 1]

        actions = (
            actions_all[batch_idx]
            .reshape(-1)[:end]
        )

        if not (
                len(prices) == len(means) == len(actions) + 1
        ):
            raise ValueError(
                "Price and action histories are not aligned."
            )

        # ======================================================
        # 3. Convert actions to positions
        # ======================================================

        # ======================================================
        # 3. Convert actions to positions
        # ======================================================

        # 0 = SHORT -> -1
        # 1 = FLAT  ->  0
        # 2 = LONG  -> +1

        positions = actions.astype(int) - 1

        previous_positions = np.concatenate(
            ([0], positions[:-1])
        )

        # Change in position
        position_delta = positions - previous_positions

        # Buy signal:
        # - Enter LONG
        # - Close SHORT
        # - Reverse SHORT to LONG
        long_idx = np.flatnonzero(
            position_delta > 0
        )

        # Sell signal:
        # - Enter SHORT
        # - Close LONG
        # - Reverse LONG to SHORT
        short_idx = np.flatnonzero(
            position_delta < 0
        )

        # ======================================================
        # 4. Create two plots
        # ======================================================

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(15, 8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3, 1]
            }
        )

        t = np.arange(len(prices))
        action_t = np.arange(len(actions))

        # ======================================================
        # Plot 1: Market and fundamental prices with positions
        # ======================================================

        ax = axes[0]

        # Fundamental price
        ax.plot(
            t,
            means,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label="Fundamental price",
            zorder=2
        )

        # # ------------------------------------------------------
        # # Market price colored by position
        # # ------------------------------------------------------
        #
        # points = np.column_stack((t, prices))
        #
        # segments = np.stack(
        #     [points[:-1], points[1:]],
        #     axis=1
        # )
        #
        # position_colors = {
        #     -1: "red",
        #     0: "gray",
        #     1: "green"
        # }
        #
        # colors = [
        #     position_colors[int(p)]
        #     for p in positions
        # ]
        #
        # collection = LineCollection(
        #     segments,
        #     colors=colors,
        #     linewidths=2.0,
        #     zorder=3
        # )
        #
        # ax.add_collection(collection)
        #
        # # Set axis limits
        # ax.set_xlim(t[0], t[-1])
        #
        # price_min = min(np.min(prices), np.min(means))
        # price_max = max(np.max(prices), np.max(means))
        #
        # margin = 0.05 * max(price_max - price_min, 1.0)
        #
        # ax.set_ylim(
        #     price_min - margin,
        #     price_max + margin
        # )

        # ------------------------------------------------------
        # Market price as one blue line
        # ------------------------------------------------------

        ax.plot(
            t,
            prices,
            color="tab:blue",
            linewidth=1.8,
            label="Market price",
            zorder=3
        )

        # Set axis limits
        ax.set_xlim(t[0], t[-1])

        price_min = min(np.min(prices), np.min(means))
        price_max = max(np.max(prices), np.max(means))

        margin = 0.05 * max(price_max - price_min, 1.0)

        ax.set_ylim(
            price_min - margin,
            price_max + margin
        )

        # ======================================================
        # Mark BUY and SELL signals
        # ======================================================

        # LONG / BUY signal
        ax.scatter(
            long_idx,
            prices[long_idx],
            color="green",
            marker="^",
            s=50,
            label="LONG / BUY",
            zorder=5
        )

        # SHORT / SELL signal
        ax.scatter(
            short_idx,
            prices[short_idx],
            color="red",
            marker="v",
            s=50,
            label="SHORT / SELL",
            zorder=5
        )

        # ------------------------------------------------------
        # Legend
        # ------------------------------------------------------

        legend_elements = [
            Line2D(
                [0], [0],
                color="tab:blue",
                linewidth=1.8,
                label="Market price"
            ),
            Line2D(
                [0], [0],
                color="tab:orange",
                linestyle="--",
                linewidth=1.5,
                label="Fundamental price"
            ),
            Line2D(
                [0], [0],
                color="green",
                marker="^",
                linestyle="None",
                markersize=7,
                label="LONG"
            ),
            Line2D(
                [0], [0],
                color="red",
                marker="v",
                linestyle="None",
                markersize=7,
                label="SHORT"
            )
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=13,
            ncol=2
        )

        ax.set_ylabel("Price")

        ax.grid(alpha=0.3)

        # ======================================================
        # Plot 2: Agent trading position
        # ======================================================

        ax = axes[1]

        # Extend the final position to the last price observation
        position_t = np.arange(len(positions) + 1)

        position_values = np.concatenate(
            [positions, positions[-1:]]
        )

        ax.step(
            position_t,
            position_values,
            where="post",
            color="tab:purple",
            linewidth=1.5,
            label="Agent position"
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8
        )

        ax.set_yticks([-1, 0, 1])

        ax.set_yticklabels([
            "SHORT",
            "FLAT",
            "LONG"
        ])

        ax.set_ylim(-1.3, 1.3)

        ax.set_ylabel("Position")
        ax.set_xlabel("Continuous Time Step")

        ax.set_title("Trading Position")

        ax.grid(alpha=0.3)

        # ======================================================
        # 5. Mark episode boundaries
        # ======================================================

        for episode in range(1, episode_idx + 1):

            boundary = episode * steps_per_episode

            for ax in axes:
                ax.axvline(
                    boundary,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.5
                )

        # ======================================================
        # 6. Display
        # ======================================================

        fig.suptitle(
            f"Agent {self.agent_type} | Trajectory {batch_idx}",
            fontsize=14
        )

        fig.tight_layout()

        plt.show()

        return fig, axes

    def calculate_sortino_ratio(
            self,
            target_return=0.0,
            periods_per_year=52,
    ):
        import numpy as np

        def as_numpy(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        prices = as_numpy(self.price_history).astype(float)
        actions = as_numpy(self.action_history)

        if prices.ndim != 2:
            raise ValueError(
                "Expected price_history with shape [batch, time]."
            )

        batch_size, price_steps = prices.shape

        actions = actions.reshape(
            batch_size,
            price_steps - 1
        )

        # SHORT = -1, FLAT = 0, LONG = +1
        positions = actions.astype(float) - 1.0

        # Market returns
        market_returns = (
                prices[:, 1:] / prices[:, :-1] - 1.0
        )

        # Trading returns before transaction costs
        gross_returns = positions * market_returns

        # Previous position, initially flat
        previous_positions = np.concatenate(
            [
                np.zeros((batch_size, 1)),
                positions[:, :-1],
            ],
            axis=1
        )

        turnover = np.abs(
            positions - previous_positions
        )

        # Transaction costs
        transaction_costs = (
                self.config.half_bidask * turnover
        )

        # Net trading returns
        net_returns = (
                gross_returns - transaction_costs
        )

        # Target return per time step
        excess_returns = (
                net_returns - target_return
        )

        # Downside deviations
        downside_returns = np.minimum(
            excess_returns,
            0.0
        )

        # Downside deviation per trajectory
        downside_deviation = np.sqrt(
            np.mean(downside_returns ** 2, axis=1)
        )

        mean_excess_return = np.mean(
            excess_returns,
            axis=1
        )

        # Annualized Sortino ratio
        sortino = np.full(
            batch_size,
            np.nan,
            dtype=float
        )

        np.divide(
            mean_excess_return,
            downside_deviation,
            out=sortino,
            where=downside_deviation > 0,
        )

        sortino *= np.sqrt(periods_per_year)

        valid = sortino[np.isfinite(sortino)]

        if valid.size == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        return {
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "min": float(valid.min()),
            "max": float(valid.max()),
        }