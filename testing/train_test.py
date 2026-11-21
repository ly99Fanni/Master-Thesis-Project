import numpy as np
import matplotlib.pyplot as plt
import torch

from main_program.config import Config
from main_program.environment import Environment
from main_program.agent import DQNAgent
from main_program.replay_buffer import ReplayBuffer


# ==========================================================
# Configuration and Environment
# ==========================================================

config = Config()

market = Environment(config)

# Start training from the original S0
market.full_reset()


# ==========================================================
# Initial state
# ==========================================================

price = market.price_seq[:, :market.window_size + 1]
mean_price = market.mean_price_seq[:, :market.window_size + 1]

current_state = market.build_state(
    price,
    mean_price,
    market.pos
)

agent = DQNAgent(
    config,
    current_state
)

replay_buffer = ReplayBuffer(config)

print_batch_info = False

# ==========================================================
# Store training statistics
# ==========================================================

episode_rewards = []
episode_losses = []

sell_ratios = []
hold_ratios = []
buy_ratios = []

episode_stats = []



# ==========================================================
# Training
# ==========================================================

for episode in range(config.num_episodes):

    # ------------------------------------------------------
    # Continue market after previous episode
    # ------------------------------------------------------

    # Episode 0 already contains the sequence generated
    # by full_reset().
    if episode > 0:
        market.reset()

    actions_by_timestep = []

    # ======================================================
    # Episode statistics
    # ======================================================

    episode_reward = torch.zeros(
        market.batch_size,
        dtype=torch.float32,
        device=market.device
    )

    episode_profit = torch.zeros(
        market.batch_size,
        dtype=torch.float32,
        device=market.device
    )

    episode_transaction_cost = torch.zeros(
        market.batch_size,
        dtype=torch.float32,
        device=market.device
    )

    episode_risk_penalty = torch.zeros(
        market.batch_size,
        dtype=torch.float32,
        device=market.device
    )

    episode_turnover = torch.zeros(
        market.batch_size,
        dtype=torch.float32,
        device=market.device
    )


    # ------------------------------------------------------
    # Action counts
    # ------------------------------------------------------

    action_counts = torch.zeros(
        config.num_actions,
        dtype=torch.long,
        device=market.device
    )


    # ------------------------------------------------------
    # Conditional-action statistics for THIS episode
    # ------------------------------------------------------

    sell_above = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )

    buy_below = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )

    hold_near = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )

    num_above = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )

    num_below = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )

    num_near = torch.tensor(
        0,
        dtype=torch.long,
        device=market.device
    )


    losses = []


    # ======================================================
    # Move through current market segment
    # ======================================================

    for t in range(
        market.window_size,
        market.num_steps - 1
    ):

        # ==================================================
        # Current state
        # ==================================================

        start = t - market.window_size

        current_price = market.price_seq[
            :, start:t + 1
        ]

        current_mean = market.mean_price_seq[
            :, start:t + 1
        ]

        current_state = market.build_state(
            current_price,
            current_mean,
            market.pos
        )


        # ==================================================
        # Current deviation from mean
        # ==================================================

        current_diff = (
            current_price[:, -1]
            - current_mean[:, -1]
        ) / current_mean[:, -1]

        # ==================================================
        # 1. Agent chooses action
        # ==================================================

        action = agent.act(
            current_state,
            greedy=False
        )

        actions_by_timestep.append(
            action.detach().cpu().clone()
        )

        # --------------------------------------------------
        # Count actions
        # --------------------------------------------------

        action_counts += torch.bincount(
            action,
            minlength=config.num_actions
        )



        # ==================================================
        # 2. Environment step
        # ==================================================

        reward, done = market.step(
            action,
            t
        )


        # ==================================================
        # Episode accounting
        # ==================================================

        episode_reward += reward

        episode_profit += (
            market.last_profit
        )

        episode_transaction_cost += (
            market.last_transaction_cost
        )

        episode_risk_penalty += (
            market.last_risk_penalty
        )

        episode_turnover += (
            market.last_turnover
        )


        # ==================================================
        # 3. Next state
        # ==================================================

        next_price = market.price_seq[
            :, start + 1:t + 2
        ]

        next_mean = market.mean_price_seq[
            :, start + 1:t + 2
        ]

        next_state = market.build_state(
            next_price,
            next_mean,
            market.pos
        )


        # ==================================================
        # 4. Store transition
        # ==================================================

        replay_buffer.push(
            current_state.detach(),
            action.detach(),
            reward.detach(),
            next_state.detach(),
            done.detach()
        )


        # ==================================================
        # 5. Train agent
        # ==================================================

        if config.buffer_is_on:

            if len(replay_buffer) >= config.replay_start_size:

                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done
                ) = replay_buffer.sample()

                loss = agent.train_step(
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done
                )

                losses.append(
                    float(loss)
                )

        else:

            loss = agent.train_step(
                current_state,
                action,
                reward,
                next_state,
                done
            )

            losses.append(
                float(loss)
            )


        # ==================================================
        # End episode
        # ==================================================

        if done.all():
            break


    # ======================================================
    # Episode reward and loss
    # ======================================================

    mean_episode_reward = (
        episode_reward
        .mean()
        .item()
    )

    episode_rewards.append(
        mean_episode_reward
    )


    if len(losses) > 0:

        mean_loss = float(
            np.mean(losses)
        )

    else:

        mean_loss = np.nan


    episode_losses.append(
        mean_loss
    )


    # ======================================================
    # Action ratios
    # ======================================================

    total_count = (
        action_counts
        .sum()
        .item()
    )

    if total_count > 0:

        sell_ratio = (
            action_counts[0].item()
            / total_count
        )

        hold_ratio = (
            action_counts[1].item()
            / total_count
        )

        buy_ratio = (
            action_counts[2].item()
            / total_count
        )

    else:

        sell_ratio = 0.0
        hold_ratio = 0.0
        buy_ratio = 0.0


    sell_ratios.append(
        sell_ratio
    )

    hold_ratios.append(
        hold_ratio
    )

    buy_ratios.append(
        buy_ratio
    )


    action_history = torch.stack(
        actions_by_timestep,
        dim=1
    )


    # ======================================================
    # Save episode statistics
    # ======================================================

    episode_stats.append({

        "episode":
            episode + 1,

        "reward":
            mean_episode_reward,

        "profit":
            episode_profit.mean().item(),

        "transaction_cost":
            episode_transaction_cost.mean().item(),

        "risk_penalty":
            episode_risk_penalty.mean().item(),

        "turnover":
            episode_turnover.mean().item(),

        "short":
            sell_ratio,

        "flat":
            hold_ratio,

        "long":
            buy_ratio,

        "loss":
            mean_loss
    })


    # ======================================================
    # Progress
    # ======================================================

    if (episode + 1) % 10 == 0:

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {mean_episode_reward:9.4f} | "
            f"Cost: {episode_transaction_cost.mean().item():9.4f} | "
            f"Risk: {episode_risk_penalty.mean().item():9.4f} | "
            f"Loss: {mean_loss:9.5f} | "
            f"Epsilon: {agent.epsilon:6.3f} | "
            f"Temperature: {agent.temperature:6.3f} | "
            f"SHORT: {sell_ratio:6.1%} | "
            f"FLAT: {hold_ratio:6.1%} | "
            f"LONG: {buy_ratio:6.1%}"
        )


    # ======================================================
    # Plot
    # ======================================================

    if print_batch_info:
        # if episode == 0 or (episode + 1) % 100 == 0 or (episode + 1) == config.num_episodes:
        if episode == 0 or (episode + 1) % 20 == 0:
            agent.plot_training_strategy(
                market,
                action_history,
                batch_number=17,
                episode_number=episode + 1
            )


            agent.action_run_stats(
                action_history,
                batch_number=17,
                print_sequence=False
            )

    # ======================================================
    # Target network update
    # ======================================================

    if (episode + 1) % 10 == 0:

        agent.update_target()


    # ======================================================
    # Exploration decay
    # ======================================================

    agent.decay_epsilon()
    agent.decay_temperature()


# ==========================================================
# Save model
# ==========================================================

torch.save(
    agent.model.state_dict(),
    "dqn_model.pth"
)


# ==========================================================
# Final action distribution
# ==========================================================

print("\n" + "-" * 50)
print("Final Episode")
print("-" * 50)

print(
    f"SHORT ratio: {sell_ratios[-1]:.2%}"
)

print(
    f"FLAT ratio: {hold_ratios[-1]:.2%}"
)

print(
    f"LONG ratio:  {buy_ratios[-1]:.2%}"
)



# ==========================================================
# Plot action distribution
# ==========================================================

episodes = np.arange(
    1,
    config.num_episodes + 1
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    episodes,
    sell_ratios,
    label="SHORT"
)

plt.plot(
    episodes,
    hold_ratios,
    label="FLAT"
)

plt.plot(
    episodes,
    buy_ratios,
    label="LONG"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Action Ratio"
)

plt.title(
    "Agent Action Distribution"
)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



# ==========================================================
# Plot reward per episode
# ==========================================================

rewards = np.array(
    episode_rewards
)

episodes = np.arange(
    1,
    len(rewards) + 1
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    episodes,
    rewards,
    alpha=0.4,
    label="Episode Reward"
)


# Only calculate moving average if enough episodes exist
reward_window = min(
    10,
    len(rewards)
)

if reward_window > 1:

    moving_avg = np.convolve(
        rewards,
        np.ones(reward_window) / reward_window,
        mode="valid"
    )

    plt.plot(
        np.arange(
            reward_window,
            len(rewards) + 1
        ),
        moving_avg,
        linewidth=2,
        label=f"{reward_window}-Episode Moving Average"
    )


plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Cumulative Reward"
)

plt.title(
    "Training Reward"
)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# Training statistics
# ==========================================================

print("\nTraining Statistics")
print("-" * 50)

print(
    f"Mean reward:        "
    f"{rewards.mean():.4f}"
)

print(
    f"Median reward:      "
    f"{np.median(rewards):.4f}"
)

print(
    f"Std reward:         "
    f"{rewards.std():.4f}"
)

print(
    f"Minimum reward:     "
    f"{rewards.min():.4f}"
)

print(
    f"Maximum reward:     "
    f"{rewards.max():.4f}"
)

print(
    f"Positive episodes:  "
    f"{(rewards > 0).mean():.2%}"
)


# ==========================================================
# Learning comparison
# ==========================================================

comparison_size = min(
    20,
    len(rewards)
)

first_rewards = rewards[
    :comparison_size
]

last_rewards = rewards[
    -comparison_size:
]

print("\nLearning Comparison")
print("-" * 50)

print(
    f"First {comparison_size} mean reward: "
    f"{first_rewards.mean():.4f}"
)

print(
    f"Last {comparison_size} mean reward:  "
    f"{last_rewards.mean():.4f}"
)

print(
    f"Improvement: "
    f"{last_rewards.mean() - first_rewards.mean():.4f}"
)


# ==========================================================
# Plot losses
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    np.arange(
        1,
        len(episode_losses) + 1
    ),
    episode_losses
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Mean Loss"
)

plt.title(
    "DQN Training Loss"
)

plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# Selected episode table
# ==========================================================

selected_episodes = set(range(50, config.num_episodes + 1, 50))
selected_episodes.add(config.num_episodes)


print("\n")

print(
    f"{'Episode':>7} "
    f"{'Reward':>10} "
    f"{'Profit':>10} "
    f"{'Cost':>10} "
    f"{'Risk':>10} "
    f"{'Turnover':>10} "
    f"{'SHORT':>9} "
    f"{'FLAT':>9} "
    f"{'LONG':>9} "
    f"{'LOSS':>12}"
)

print("-" * 115)


for stats in episode_stats:

    if stats["episode"] in selected_episodes:

        print(
            f"{stats['episode']:>7} "
            f"{stats['reward']:>10.4f} "
            f"{stats['profit']:>10.4f} "
            f"{stats['transaction_cost']:>10.4f} "
            f"{stats['risk_penalty']:>10.4f} "
            f"{stats['turnover']:>10.2f} "
            f"{stats['short']:>9.1%} "
            f"{stats['flat']:>9.1%} "
            f"{stats['long']:>9.1%} "
            f"{stats['loss']:>12.5f}"
        )