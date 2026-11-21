import torch
import torch.optim as optim
import numpy as np
from main_program.config import Config
from main_program.dqn_model import SimpleDQN, DuelingDQN, GDQN
import matplotlib.pyplot as plt
from collections import defaultdict



class DQNAgent:

    def __init__(self, config: Config, state):

        self.config = config
        self.device = config.device

        # ==================================================
        # Model selection
        # ==================================================

        self.use_DuelingDQN = config.use_DuelingDQN
        self.use_GDQN = config.use_GDQN
        self.use_softmax = config.use_softmax

        if self.use_GDQN and self.use_DuelingDQN:
            raise ValueError(
                "use_GDQN and use_DuelingDQN cannot both be True."
            )

        # ==================================================
        # State dimensions
        # ==================================================

        """
        Simple DQN / Dueling DQN:

            state shape:
            (batch_size, state_size)

        GDQN:

            state shape:
            (batch_size, window_size, num_features)

        Therefore state.shape[-1] means:

            Simple DQN:
                flattened state size

            GDQN:
                number of features per timestep
        """

        if self.use_GDQN:

            if state.dim() != 3:
                raise ValueError(
                    "GDQN expects state shape "
                    "(batch_size, window_size, num_features), "
                    f"but received {tuple(state.shape)}"
                )

            self.state_size = state.shape[-1]

        else:

            if state.dim() != 2:
                raise ValueError(
                    "Simple/Dueling DQN expects state shape "
                    "(batch_size, state_size), "
                    f"but received {tuple(state.shape)}"
                )

            self.state_size = state.shape[-1]

        self.batch_size = state.shape[0]
        self.num_actions = config.num_actions

        # ==================================================
        # Learning parameters
        # ==================================================

        self.gamma = config.gamma
        self.learning_rate = config.learning_rate

        # ==================================================
        # Epsilon-greedy parameters
        # ==================================================

        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay

        # ==================================================
        # Softmax parameters
        # ==================================================

        self.temperature = config.temperature
        self.temperature_min = config.temperature_min
        self.temperature_decay = config.temperature_decay

        # ==================================================
        # Networks
        # ==================================================

        if self.use_GDQN:

            self.model = GDQN(
                input_size=self.state_size,
                num_actions=self.num_actions,
                hidden_size=64
            ).to(self.device)

            self.target_model = GDQN(
                input_size=self.state_size,
                num_actions=self.num_actions,
                hidden_size=64
            ).to(self.device)

        elif self.use_DuelingDQN:

            self.model = DuelingDQN(
                self.state_size,
                self.num_actions,
                self.config
            ).to(self.device)

            self.target_model = DuelingDQN(
                self.state_size,
                self.num_actions,
                self.config
            ).to(self.device)

        else:

            self.model = SimpleDQN(
                self.state_size,
                self.num_actions,
                self.config
            ).to(self.device)

            self.target_model = SimpleDQN(
                self.state_size,
                self.num_actions,
                self.config
            ).to(self.device)

        # ==================================================
        # Initialize target network
        # ==================================================

        self.update_target()

        # ==================================================
        # Optimizer and loss
        # ==================================================

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.loss_fn = torch.nn.SmoothL1Loss()


    # ======================================================
    # Epsilon-greedy action selection
    # ======================================================

    def epsilon_greedy(self, state):

        """
        Parameters
        ----------
        state

        Simple/Dueling DQN:
            (batch_size, state_size)

        GDQN:
            (batch_size, window_size, num_features)

        Returns
        -------
        action:
            (batch_size,)
        """

        state = state.to(
            dtype=torch.float32,
            device=self.device
        )

        batch_size = state.shape[0]

        # --------------------------------------------------
        # Get greedy actions
        # --------------------------------------------------

        self.model.eval()

        with torch.no_grad():

            q_values = self.model(state)

            greedy_actions = torch.argmax(
                q_values,
                dim=1
            )

        # --------------------------------------------------
        # Random actions
        # --------------------------------------------------

        random_actions = torch.randint(
            low=0,
            high=self.num_actions,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device
        )

        # --------------------------------------------------
        # Decide independently for every simulated path
        # --------------------------------------------------

        explore_mask = (
            torch.rand(
                batch_size,
                device=self.device
            )
            < self.epsilon
        )

        actions = torch.where(
            explore_mask,
            random_actions,
            greedy_actions
        )

        return actions


    # ======================================================
    # Softmax action selection
    # ======================================================

    def softmax_action(self, state, debug = False):

        state = state.to(
            dtype=torch.float32,
            device=self.device
        )


        self.model.eval()

        with torch.no_grad():

            q_values = self.model(state)

        # Avoid division by zero
        temperature = max(
            self.temperature,
            self.temperature_min
        )

        scaled_q = (
            q_values / temperature
        )

        # Numerical stability
        scaled_q = (
            scaled_q
            - scaled_q.max(
                dim=1,
                keepdim=True
            ).values
        )

        probabilities = torch.softmax(
            scaled_q,
            dim=1
        )

        if debug:
            print(
                "Q mean:",
                q_values.mean(dim=0).cpu().numpy()
            )

            print(
                "Q gap:",
                (
                        q_values.max(dim=1).values
                        - q_values.min(dim=1).values
                ).mean().item()
            )

            print(
                "Mean probs:",
                probabilities.mean(dim=0).cpu().numpy()
            )

            print(
                "Temperature:",
                temperature
            )

        action = torch.multinomial(
            probabilities,
            num_samples=1
        ).squeeze(1)

        return action


    # ======================================================
    # Select action
    # ======================================================

    def act(self, state, greedy=False, debug=False):

        state = state.to(
            dtype=torch.float32,
            device=self.device
        )

        # Evaluation: no exploration
        if greedy:

            self.model.eval()

            with torch.no_grad():
                q_values = self.model(state)

                return torch.argmax(
                    q_values,
                    dim=1
                )

        # Training
        if self.use_softmax:
            return self.softmax_action(state, debug)

        return self.epsilon_greedy(state)

    # ======================================================
    # Training diagnostics
    # ======================================================

    def get_diagnostics(self, state):

        state = state.to(
            dtype=torch.float32,
            device=self.device
        )

        self.model.eval()

        with torch.no_grad():

            q_values = self.model(state)

            # ----------------------------------------------
            # Q mean for SHORT / FLAT / LONG
            # ----------------------------------------------

            q_mean = q_values.mean(dim=0)

            # ----------------------------------------------
            # Average difference between best and worst Q
            # ----------------------------------------------

            q_gap = (
                    q_values.max(dim=1).values
                    - q_values.min(dim=1).values
            ).mean()

            # ----------------------------------------------
            # Greedy actions
            # ----------------------------------------------

            greedy_actions = torch.argmax(
                q_values,
                dim=1
            )

            greedy_counts = torch.bincount(
                greedy_actions,
                minlength=self.num_actions
            ).float()

            greedy_ratios = (
                    greedy_counts
                    / greedy_counts.sum()
            )

            # ----------------------------------------------
            # Softmax probabilities
            # ----------------------------------------------

            if self.use_softmax:

                temperature = max(
                    self.temperature,
                    self.temperature_min
                )

                scaled_q = (
                        q_values / temperature
                )

                scaled_q = (
                        scaled_q
                        - scaled_q.max(
                    dim=1,
                    keepdim=True
                ).values
                )

                probabilities = torch.softmax(
                    scaled_q,
                    dim=1
                )

                mean_probabilities = (
                    probabilities.mean(dim=0)
                )

            else:

                mean_probabilities = None
                temperature = None

        return {
            "q_mean": q_mean.cpu(),
            "q_gap": q_gap.item(),
            "mean_probabilities": (
                None
                if mean_probabilities is None
                else mean_probabilities.cpu()
            ),
            "greedy_ratios": greedy_ratios.cpu(),
            "temperature": temperature
        }


    # # ======================================================
    # # DQN training step
    # # ======================================================
    #
    # def train_step(
    #         self,
    #         state,
    #         action,
    #         reward,
    #         next_state,
    #         done,
    #         weights=None
    # ):
    #
    #     self.model.train()
    #
    #     # ==================================================
    #     # Current Q-values
    #     # ==================================================
    #
    #     q_values = self.model(
    #         state
    #     )
    #
    #     selected_q_values = (
    #         q_values
    #         .gather(
    #             1,
    #             action.unsqueeze(1)
    #         )
    #         .squeeze(1)
    #     )
    #
    #     # ==================================================
    #     # Target Q-values
    #     # ==================================================
    #
    #     with torch.no_grad():
    #
    #         next_q_values = (
    #             self.target_model(
    #                 next_state
    #             )
    #         )
    #
    #         next_q_max = (
    #             next_q_values
    #             .max(dim=1)
    #             .values
    #         )
    #
    #         target_q_values = (
    #                 reward
    #                 + self.gamma
    #                 * (~done).float()
    #                 * next_q_max
    #         )
    #
    #     # ==================================================
    #     # TD error
    #     #
    #     # delta = target - prediction
    #     # ==================================================
    #
    #     td_errors = (
    #             target_q_values
    #             - selected_q_values
    #     )
    #
    #     # ==================================================
    #     # Loss per transition
    #     # ==================================================
    #
    #     loss_per_sample = (
    #         F.smooth_l1_loss(
    #             selected_q_values,
    #             target_q_values,
    #             reduction="none"
    #         )
    #     )
    #
    #     # ==================================================
    #     # PER weighting
    #     # ==================================================
    #
    #     if weights is not None:
    #
    #         loss = (
    #                 weights
    #                 * loss_per_sample
    #         ).mean()
    #
    #     else:
    #
    #         loss = (
    #             loss_per_sample
    #             .mean()
    #         )
    #
    #     mean_td_error = (
    #         td_errors
    #         .detach()
    #         .abs()
    #         .mean()
    #         .item()
    #     )
    #
    #
    #
    #     # ==================================================
    #     # Optimize
    #     # ==================================================
    #
    #     self.optimizer.zero_grad()
    #
    #     loss.backward()
    #
    #     self.optimizer.step()
    #
    #     return (
    #         loss.item(),
    #         td_errors.detach(),
    #         mean_td_error
    #     )

    def train_step(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            weights=None
    ):

        # ==================================================
        # Training mode
        # ==================================================

        self.model.train()

        # ==================================================
        # Make shapes and dtypes explicit
        # ==================================================

        state = state.to(
            dtype=torch.float32,
            device=self.device
        )

        next_state = next_state.to(
            dtype=torch.float32,
            device=self.device
        )

        action = (
            action
            .to(
                dtype=torch.long,
                device=self.device
            )
            .view(-1)
        )

        reward = (
            reward
            .to(
                dtype=torch.float32,
                device=self.device
            )
            .view(-1)
        )

        done = (
            done
            .to(
                dtype=torch.bool,
                device=self.device
            )
            .view(-1)
        )

        # ==================================================
        # Current Q(s,a)
        # ==================================================

        q_values = self.model(
            state
        )

        selected_q_values = (
            q_values
            .gather(
                1,
                action.unsqueeze(1)
            )
            .squeeze(1)
        )

        # ==================================================
        # SIMPLE DQN target
        #
        # y = r + gamma max Q_target(s', a)
        # ==================================================

        with torch.no_grad():

            next_q_values = (
                self.target_model(
                    next_state
                )
            )

            next_q_max = (
                next_q_values
                .max(dim=1)
                .values
            )

            target_q_values = (
                    reward
                    + self.gamma
                    * (~done).float()
                    * next_q_max
            )

        # ==================================================
        # Safety check
        # ==================================================

        if not torch.isfinite(
                selected_q_values
        ).all():
            raise RuntimeError(
                "Non-finite current Q-values."
            )

        if not torch.isfinite(
                target_q_values
        ).all():
            raise RuntimeError(
                "Non-finite target Q-values."
            )

        # ==================================================
        # TD error
        # ==================================================

        td_errors = (
                target_q_values
                - selected_q_values
        )

        # ==================================================
        # Huber loss
        # ==================================================

        loss_per_sample = (
            torch.nn.functional.smooth_l1_loss(
                selected_q_values,
                target_q_values,
                reduction="none"
            )
        )

        if weights is not None:

            weights = (
                weights
                .to(self.device)
                .view(-1)
            )

            loss = (
                    weights
                    * loss_per_sample
            ).mean()

        else:

            loss = (
                loss_per_sample
                .mean()
            )

        # ==================================================
        # Backpropagation
        # ==================================================

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )

        self.optimizer.step()

        return (
            loss.item(),
            td_errors.detach()
        )



    # ======================================================
    # Update target network
    # ======================================================

    def update_target(self):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()


    # ======================================================
    # Decay epsilon
    # ======================================================

    def decay_epsilon(self):

        if not self.use_softmax:

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon
                * self.epsilon_decay
            )


    # ======================================================
    # Decay softmax temperature
    # ======================================================

    def decay_temperature(self):

        if self.use_softmax:

            self.temperature = max(
                self.temperature_min,
                self.temperature * self.temperature_decay
            )


    # ======================================================
    # Plot agent strategy
    # ======================================================
    def plot_training_strategy(self,
            market,
            action_history,
            batch_number=0,
            episode_number=None
    ):
        """
        Plot the actual actions taken during training.

        Action mapping:
            0 = SHORT (-1)
            1 = FLAT  (0)
            2 = LONG  (+1)
        """

        # ======================================================
        # Validate batch
        # ======================================================

        batch_number = max(
            0,
            min(
                batch_number,
                market.batch_size - 1
            )
        )

        # ======================================================
        # Data for selected batch
        # ======================================================

        price = (
            market.price_seq[batch_number]
            .detach()
            .cpu()
            .numpy()
        )

        mean_price = (
            market.mean_price_seq[batch_number]
            .detach()
            .cpu()
            .numpy()
        )

        actions = (
            action_history[batch_number]
            .detach()
            .cpu()
            .numpy()
        )

        # action:
        # 0 -> -1
        # 1 ->  0
        # 2 -> +1

        positions = actions - 1

        # ======================================================
        # Time
        # ======================================================

        time = (
                np.arange(market.num_steps)
                * market.dt
        )

        action_start = (
                market.window_size - 1
        )

        action_indices = np.arange(
            action_start,
            action_start + len(actions)
        )

        action_time = (
                action_indices
                * market.dt
        )

        # ======================================================
        # Figure
        # ======================================================

        fig, (ax_price, ax_action) = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [4, 1]
            }
        )

        # ======================================================
        # TOP: price and mean
        # ======================================================

        ax_price.plot(
            time,
            price,
            color="blue",
            linewidth=2,
            label="Price"
        )

        ax_price.plot(
            time,
            mean_price,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Mean Price"
        )

        if episode_number is None:

            title = (
                f"Training Strategy - "
                f"Batch {batch_number}"
            )

        else:

            title = (
                f"Training Strategy - "
                f"Episode {episode_number}, "
                f"Batch {batch_number}"
            )

        ax_price.set_title(
            title
        )

        ax_price.set_ylabel(
            "Price"
        )

        ax_price.legend()

        ax_price.grid(
            alpha=0.3
        )

        # ======================================================
        # Masks
        # ======================================================

        short_mask = (
                positions == -1
        )

        flat_mask = (
                positions == 0
        )

        long_mask = (
                positions == 1
        )

        # ======================================================
        # BOTTOM: action bars
        # ======================================================

        bar_width = (
                market.dt * 0.9
        )

        # ------------------------------------------------------
        # LONG = green
        # ------------------------------------------------------

        # LONG (+1)
        ax_action.bar(
            action_time[long_mask],
            np.ones(long_mask.sum()),
            width=bar_width,
            align="center",
            color="green",
            label="LONG (+1)"
        )

        # ------------------------------------------------------
        # SHORT = red
        # ------------------------------------------------------

        # SHORT (-1)
        ax_action.bar(
            action_time[short_mask],
            -np.ones(short_mask.sum()),
            width=bar_width,
            align="center",
            color="red",
            label="SHORT (-1)"
        )

        # ------------------------------------------------------
        # FLAT = gray
        #
        # A zero-height bar cannot be seen, so give it a small
        # visible height around zero.
        # ------------------------------------------------------


        # FLAT (0)
        ax_action.scatter(
            action_time[flat_mask],
            np.zeros(flat_mask.sum()),
            marker="o",
            s=30,
            color="gray",
            label="FLAT (0)"
        )

        # ======================================================
        # Zero line
        # ======================================================

        ax_action.axhline(
            0,
            color="black",
            linewidth=1
        )

        # ======================================================
        # Format action axis
        # ======================================================

        ax_action.set_ylim(
            -1.25,
            1.25
        )

        ax_action.set_yticks(
            [-1, 0, 1]
        )

        ax_action.set_yticklabels(
            [
                "SHORT",
                "FLAT",
                "LONG"
            ]
        )

        ax_action.set_xlabel(
            "Time"
        )

        ax_action.set_ylabel(
            "Action"
        )

        ax_action.legend(
            loc="upper right",
            ncol=3
        )

        ax_action.grid(
            alpha=0.2
        )

        # ======================================================
        # Layout
        # ======================================================

        plt.tight_layout()

        plt.show()

    import numpy as np
    from collections import defaultdict

    def action_run_stats(
            self,
            action_history,
            batch_number,
            print_sequence=False
    ):
        """
        Analyze consecutive action runs for one batch.

        action_history shape:
            (batch_size, num_steps)

        Actions:
            0 = SHORT
            1 = FLAT
            2 = LONG

        Parameters
        ----------
        print_sequence : bool
            If True, print every consecutive action run.
            If False, print only the summary statistics.
        """

        # --------------------------------------------------
        # Get actions for selected batch
        # --------------------------------------------------
        if isinstance(action_history, list):
            actions = np.array(action_history)

            # If stored as (time, batch), select batch
            if actions.ndim == 2:
                actions = actions[:, batch_number]

        else:
            actions = action_history

            if hasattr(actions, "detach"):
                actions = actions.detach().cpu().numpy()

            actions = actions[batch_number]

        actions = np.asarray(actions).flatten()

        action_names = {
            0: "SHORT",
            1: "FLAT",
            2: "LONG"
        }

        # --------------------------------------------------
        # Find consecutive runs
        # --------------------------------------------------
        runs = []

        current_action = actions[0]
        run_length = 1
        start_step = 0

        for t in range(1, len(actions)):

            if actions[t] == current_action:
                run_length += 1

            else:
                runs.append({
                    "action": action_names[int(current_action)],
                    "length": run_length,
                    "start": start_step,
                    "end": t - 1
                })

                current_action = actions[t]
                run_length = 1
                start_step = t

        # Add final run
        runs.append({
            "action": action_names[int(current_action)],
            "length": run_length,
            "start": start_step,
            "end": len(actions) - 1
        })

        # --------------------------------------------------
        # Collect lengths by action
        # --------------------------------------------------
        run_lengths = defaultdict(list)

        for run in runs:
            run_lengths[run["action"]].append(run["length"])

        # --------------------------------------------------
        # Header
        # --------------------------------------------------
        print(f"\nAction Run Analysis - Batch {batch_number}")
        print("=" * 60)

        # --------------------------------------------------
        # Optional detailed sequence
        # --------------------------------------------------
        if print_sequence:

            print("\nSequence:")

            for run in runs:
                print(
                    f"{run['action']:5s} "
                    f"{run['length']:3d} steps "
                    f"(t={run['start']} -> {run['end']})"
                )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        print("\nSummary")
        print("-" * 60)

        print(
            f"{'Action':<10}"
            f"{'Runs':>8}"
            f"{'Avg Run':>12}"
            f"{'Max Run':>12}"
            f"{'Total Steps':>14}"
        )

        print("-" * 60)

        for action in ["SHORT", "FLAT", "LONG"]:

            lengths = run_lengths[action]

            if len(lengths) > 0:

                print(
                    f"{action:<10}"
                    f"{len(lengths):>8}"
                    f"{np.mean(lengths):>12.2f}"
                    f"{np.max(lengths):>12}"
                    f"{np.sum(lengths):>14}"
                )

            else:
                print(
                    f"{action:<10}"
                    f"{0:>8}"
                    f"{0:>12.2f}"
                    f"{0:>12}"
                    f"{0:>14}"
                )

        # --------------------------------------------------
        # Number of switches
        # --------------------------------------------------
        switches = np.sum(actions[1:] != actions[:-1])

        if len(actions) > 1:
            switch_rate = switches / (len(actions) - 1)
        else:
            switch_rate = 0.0

        print("\nPosition switching")
        print("-" * 60)

        print(f"Total steps      : {len(actions)}")
        print(f"Total runs       : {len(runs)}")
        print(f"Action switches  : {switches}")
        print(f"Switch rate      : {switch_rate:.2%}")
        print(f"Hold same action : {len(actions) - 1 - switches}")

        return runs