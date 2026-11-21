import random
from collections import deque
import torch
from main_program.config import Config


class ReplayBuffer:

    def __init__(self, config: Config):

        self.config = config

        self.replay_batch_size = (
            config.replay_batch_size
        )

        self.device = config.device

        self.buffer = deque(
            maxlen=config.replay_capacity
        )

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

        for i in range(batch_size):

            self.buffer.append(
                (
                    state[i].detach().cpu(),
                    action[i].detach().cpu(),
                    reward[i].detach().cpu(),
                    next_state[i].detach().cpu(),
                    done[i].detach().cpu()
                )
            )

    # ======================================================
    # Sample minibatch
    # ======================================================

    def sample(self):

        if len(self.buffer) < self.replay_batch_size:

            raise RuntimeError(
                f"Replay buffer contains only "
                f"{len(self.buffer)} transitions, "
                f"but batch size is "
                f"{self.replay_batch_size}."
            )

        transitions = random.sample(
            self.buffer,
            self.replay_batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*transitions)

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
            dones
        )

    # ======================================================
    # Buffer length
    # ======================================================

    def __len__(self):

        return len(self.buffer)