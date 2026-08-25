import numpy as np
import matplotlib.pyplot as plt
import torch

from config import Config
from environment import Environment
from agent import DQNAgent
from replay_buffer import ReplayBuffer

config = Config()

market = Environment(config)

market.print_env()
market.plot_price()
print('')

position = market.pos
price = market.price_seq[:, :market.window_size]
mean_price = market.mean_price_seq[:,:market.window_size]

current_state = market.build_state(price, mean_price, position)
agent = DQNAgent(config,current_state)

replay_buffer = ReplayBuffer(config)

# ==========================================================
# Store statistics
# ==========================================================


sell_ratios = []
hold_ratios = []
buy_ratios = []

sell_above = torch.tensor(0, dtype=torch.long, device=config.device)
buy_below = torch.tensor(0, dtype=torch.long, device=config.device)
hold_near = torch.tensor(0, dtype=torch.long, device=config.device)

num_above = torch.tensor(0, dtype=torch.long, device=config.device)
num_below = torch.tensor(0, dtype=torch.long, device=config.device)
num_near = torch.tensor(0, dtype=torch.long, device=config.device)

episode_accuracies = []
episode_rewards = []
episode_losses = []
episode_stats = []


# ==================================================
# Training
# ==================================================

for episode in range(config.num_episodes):

    market.reset()

    episode_reward = torch.zeros(market.batch_size,dtype=torch.float32,device=market.device)
    action_counts = torch.zeros(config.num_actions,dtype=torch.long,device=market.device)

    episode_profit = torch.zeros(market.batch_size, device=market.device)
    episode_transaction_cost = torch.zeros(market.batch_size, device=market.device)
    episode_risk_penalty = torch.zeros(market.batch_size, device=market.device)
    episode_turnover = torch.zeros(market.batch_size,device=market.device)

    correct_actions = 0
    total_actions = 0
    losses = []

    for t in range(market.window_size - 1, market.num_steps - 1):
        start = t - config.window_size + 1

        current_price = market.price_seq[:, start:t + 1]
        current_mean = market.mean_price_seq[:, start:t + 1]

        current_state = market.build_state(current_price,current_mean,market.pos)

        # Current relative difference only
        current_diff = (current_price[:, -1]- current_mean[:, -1]) / current_mean[:, -1]

        # 1. Agent chooses action
        action = agent.act(current_state)
        action_counts += torch.bincount(action,minlength=config.num_actions)

        optimal_action = torch.ones(market.batch_size, dtype=torch.long, device=market.device) # default = 1 HOLD
        optimal_action[current_diff < - config.threshold] = 2 # BUY
        optimal_action[current_diff > config.threshold] = 0 # SELL

        # correct_actions += (action == optimal_action).sum().item()
        total_actions += market.batch_size

        num_above += (current_diff > config.threshold).sum()
        num_below += (current_diff < -config.threshold).sum()
        num_near += (torch.abs(current_diff) <= config.threshold).sum()

        sell_above += ((action == 0)& (current_diff > config.threshold)).sum()
        buy_below += ((action == 2)& (current_diff < -config.threshold)).sum()
        hold_near += ((action == 1)& (torch.abs(current_diff) <= config.threshold)).sum()


        # 2. Environment step
        next_state_r, reward, done = market.step(action,t)

        episode_profit += market.last_profit
        episode_transaction_cost += market.last_transaction_cost
        episode_risk_penalty += market.last_risk_penalty
        episode_turnover += market.last_turnover

        # next_price = next_state_r[0]
        next_pos = next_state_r[1]
        # next_mean_price = market.mean_price_seq[:, t + 1]
        # next_state = market.build_state(next_price, current_price, next_mean_price, current_mean_price, next_pos)

        next_price = market.price_seq[:,start + 1:t + 2]
        next_mean = market.mean_price_seq[:,start + 1:t + 2]
        next_state = market.build_state(next_price,next_mean,next_pos)

        # Add reward
        episode_reward += reward

        # 3. Store transition
        replay_buffer.push(current_state, action, reward, next_state, done)

        # 4. Random replay training if memory has enough examples
        if config.buffer_is_on:
            if len(replay_buffer) >= config.replay_start_size:
                (batch_state, batch_action, batch_reward, batch_next_state, batch_done) = replay_buffer.sample()

                # 5. Calculate the loss function
                loss = agent.train_step(batch_state, batch_action, batch_reward, batch_next_state, batch_done)
                losses.append(loss)
        else:
            # 5. Calculate the loss function
            loss = agent.train_step(current_state, action, reward, next_state, done)
            losses.append(loss)

        current_price = next_price
        current_state = next_state

        if done.all():
            break


    # Mean reward across all simulated paths
    mean_episode_reward = episode_reward.mean().item()
    episode_losses.append(np.mean(losses))
    episode_rewards.append(mean_episode_reward)

    total_count = action_counts.sum().item()
    sell_ratio = (action_counts[0].item() / total_count)
    hold_ratio = (action_counts[1].item() / total_count)
    buy_ratio = (action_counts[2].item() / total_count)

    sell_ratios.append(sell_ratio)
    hold_ratios.append(hold_ratio)
    buy_ratios.append(buy_ratio)

    # accuracy = (correct_actions / total_actions)
    # episode_accuracies.append(accuracy)

    episode_stats.append({
        "episode": episode + 1,
        "reward": episode_reward.mean().item(),
        "profit": episode_profit.mean().item(),
        "transaction_cost":
            episode_transaction_cost.mean().item(),
        "risk_penalty":
            episode_risk_penalty.mean().item(),
        "turnover":
            episode_turnover.mean().item(),
        "sell": sell_ratio,
        "hold": hold_ratio,
        "buy": buy_ratio,
        "loss": np.mean(losses)
    })

    sell_above_ratio = (sell_above.float() / num_above).item() if num_above.item() > 0 else 0.0
    buy_below_ratio = (buy_below.float() / num_below).item() if num_below.item() > 0 else 0.0
    hold_near_ratio = (hold_near.float() / num_near).item() if num_near.item() > 0 else 0.0


    # Print every 50 episodes
    if (episode + 1) % 20 == 0:

        print(
            f"Episode {episode + 1:4d} |"
            f"Reward: {episode_rewards[episode]:8.3f} |"
            f"Loss: {episode_losses[episode]:8.4f} |"
            # f"Epsilon: {agent.epsilon:.4f}"

        )

    if episode % 10 == 0:
        agent.update_target()

    agent.decay_epsilon()
    agent.decay_temperature()

# Save the weight for evaluation test
torch.save(agent.model.state_dict(),"dqn_model.pth")

#--------------------------------------------------------------------------------
# RESULT - Ratio of Actions
#--------------------------------------------------------------------------------
print("-" * 70)
print("Final Episode")
print("-" * 40)

print(f"SELL ratio: {sell_ratios[-1]:.2%}")
print(f"HOLD ratio: {hold_ratios[-1]:.2%}")
print(f"BUY ratio:  {buy_ratios[-1]:.2%}")
# print(f"Accuracy:   {episode_accuracies[-1]:.2%}")

#------------------------------------------------------------------------------

episodes = range(1,config.num_episodes + 1)

plt.figure(figsize=(10, 6))
plt.plot(episodes,sell_ratios,label="SELL")
plt.plot(episodes,hold_ratios,label="HOLD")
plt.plot(episodes,buy_ratios,label="BUY")

plt.xlabel("Episode")
plt.ylabel("Action Ratio")

plt.title("Agent Action Distribution")

plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
#---------------------------------------------------------------------

# plt.figure(figsize=(10, 6))
# plt.plot(episodes,episode_accuracies)
#
# plt.xlabel("Episode")
# plt.ylabel("Accuracy")
# plt.title("Action Accuracy per Episode")
#
# plt.grid()
# plt.tight_layout()
# plt.show()

#-------------------------------------------------------------------------

print(f"SELL above mean: {sell_above_ratio:.1%}")
print(f"BUY below mean:  {buy_below_ratio:.1%}")
print(f"HOLD near mean:  {hold_near_ratio:.1%}")

# ==================================================
# Plot reward per episode
# ==================================================
rewards = np.array(episode_rewards)
episodes = np.arange(1, len(rewards) + 1)

reward_window = 10

moving_avg = np.convolve(
    rewards,
    np.ones(reward_window) / reward_window,
    mode="valid"
)

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    rewards,
    alpha=0.4,
    label="Episode Reward"
)

plt.plot(
    np.arange(reward_window, len(rewards) + 1),
    moving_avg,
    linewidth=2,
    label=f"{reward_window}-Episode Moving Average"
)

plt.axhline(0, linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training Reward")

plt.legend()
plt.grid()
plt.show()

# ==================================================
# Summary Statistics
# ==================================================

rewards = np.array(episode_rewards)

print("\nTraining Statistics")
print("-" * 40)

print(f"Mean reward:        {rewards.mean():.3f}")
print(f"Median reward:      {np.median(rewards):.3f}")
print(f"Std reward:         {rewards.std():.3f}")
print(f"Minimum reward:     {rewards.min():.3f}")
print(f"Maximum reward:     {rewards.max():.3f}")
print(f"Positive episodes:  {(rewards > 0).mean():.2%}")

first_20 = rewards[:20]
last_20 = rewards[-20:]

print("\nLearning Comparison")
print("-" * 40)

print(f"First 20 mean reward: {first_20.mean():.3f}")
print(f"Last 20 mean reward:  {last_20.mean():.3f}")
print(f"Improvement:          "f"{last_20.mean() - first_20.mean():.3f}")

# ==================================================
# Plot Losses
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(range(1, len(episode_losses) + 1),episode_losses)

plt.xlabel("Episode")
plt.ylabel("Mean Loss")
plt.title("DQN Training Loss")

plt.grid()
plt.tight_layout()
plt.show()

#---------------------------------------------------------------

selected_episodes = {10, 50}

selected_episodes.update(range(100, config.num_episodes + 1, 100))
selected_episodes.add(config.num_episodes)

print('\n')
print(
    f"{'Episode':>7} "
    f"{'Reward':>10} "
    f"{'Profit':>10} "
    f"{'Cost':>10} "
    f"{'Risk':>10} "
    f"{'Turnover':>10} "
    f"{'SELL':>10} "
    f"{'HOLD':>10} "
    f"{'BUY':>10}"
    f"{'LOSS':>10}"
)

print("-" * 110)

for stats in episode_stats:

    if stats["episode"] in selected_episodes:

        print(
            f"{stats['episode']:>7} "
            f"{stats['reward']:>10.2f} "
            f"{stats['profit']:>10.2f} "
            f"{stats['transaction_cost']:>10.4f} "
            f"{stats['risk_penalty']:>10.4f} "
            f"{stats['turnover']:>10.2f} "
            f"{stats['sell']:>10.1%} "
            f"{stats['hold']:>10.1%} "
            f"{stats['buy']:>10.1%}"
            f"{stats['loss']:>10.3f}"
        )