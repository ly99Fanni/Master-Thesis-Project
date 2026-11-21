import numpy as np
import torch

from main_program.config import Config


class PrioritizedReplayBuffer:

    def __init__(self, config: Config):

        self.config = config

        self.capacity = (
            config.replay_capacity
        )

        self.replay_batch_size = (
            config.replay_batch_size
        )

        self.device = config.device

        # ==================================================
        # PER parameters
        # ==================================================

        self.alpha = config.per_alpha

        self.beta = (
            config.per_beta_start
        )

        self.priority_epsilon = (
            config.per_epsilon
        )

        # ==================================================
        # Storage
        # ==================================================

        self.buffer = []

        self.priorities = []

        self.position = 0

    # ======================================================
    # Add transitions
    # ======================================================

    def push(
            self,
            state,
            action,
            reward,
            next_state,
            done
    ):

        batch_size = state.shape[0]

        # Give new samples maximum priority
        if len(self.priorities) == 0:

            max_priority = 1.0

        else:

            max_priority = max(
                self.priorities
            )

        for i in range(batch_size):

            transition = (
                state[i].detach().cpu(),
                action[i].detach().cpu(),
                reward[i].detach().cpu(),
                next_state[i].detach().cpu(),
                done[i].detach().cpu()
            )

            # ----------------------------------------------
            # Buffer not yet full
            # ----------------------------------------------

            if len(self.buffer) < self.capacity:

                self.buffer.append(
                    transition
                )

                self.priorities.append(
                    max_priority
                )

            # ----------------------------------------------
            # Replace oldest transition
            # ----------------------------------------------

            else:

                self.buffer[
                    self.position
                ] = transition

                self.priorities[
                    self.position
                ] = max_priority

            self.position = (
                self.position + 1
            ) % self.capacity

    # ======================================================
    # Sample prioritized minibatch
    # ======================================================

    def sample(self):

        buffer_size = len(
            self.buffer
        )

        if (
                buffer_size
                < self.replay_batch_size
        ):

            raise RuntimeError(
                f"Replay buffer contains only "
                f"{buffer_size} transitions, "
                f"but batch size is "
                f"{self.replay_batch_size}."
            )

        # ==================================================
        # Convert priorities
        # ==================================================

        priorities = np.asarray(
            self.priorities[
                :buffer_size
            ],
            dtype=np.float64
        )

        # ==================================================
        # Sampling probabilities
        #
        # P(i) = p_i^alpha / sum p_k^alpha
        # ==================================================

        scaled_priorities = (
            priorities ** self.alpha
        )

        probabilities = (
            scaled_priorities
            / scaled_priorities.sum()
        )

        # ==================================================
        # Sample indices
        # ==================================================

        indices = np.random.choice(
            buffer_size,
            self.replay_batch_size,
            replace=False,
            p=probabilities
        )

        transitions = [
            self.buffer[index]
            for index in indices
        ]

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*transitions)

        # ==================================================
        # Importance sampling weights
        #
        # w_i = (N * P(i)) ^ (-beta)
        # ==================================================

        weights = (
            buffer_size
            * probabilities[indices]
        ) ** (-self.beta)

        # Normalize
        weights = (
            weights
            / weights.max()
        )

        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device
        )

        # ==================================================
        # Convert transitions to tensors
        # ==================================================

        states = (
            torch.stack(states)
            .float()
            .to(self.device)
        )

        actions = (
            torch.stack(actions)
            .long()
            .to(self.device)
        )

        rewards = (
            torch.stack(rewards)
            .float()
            .to(self.device)
        )

        next_states = (
            torch.stack(next_states)
            .float()
            .to(self.device)
        )

        dones = (
            torch.stack(dones)
            .bool()
            .to(self.device)
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights
        )

    # ======================================================
    # Update priorities
    # ======================================================

    def update_priorities(
            self,
            indices,
            td_errors
    ):

        td_errors = (
            td_errors
            .detach()
            .abs()
            .cpu()
            .numpy()
        )

        for index, error in zip(
                indices,
                td_errors
        ):

            self.priorities[
                index
            ] = (
                float(error)
                + self.priority_epsilon
            )

    # ======================================================
    # Update beta
    # ======================================================

    def set_beta(
            self,
            beta
    ):

        self.beta = beta

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):

        return len(self.buffer)