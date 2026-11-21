import numpy as np
import torch

from main_program.config import Config
from environment import Environment
from main_program.agent import DQNAgent
import matplotlib.pyplot as plt


config = Config()
market = Environment(config)

# --------------------------------------------------
# Build initial state so agent knows state_size
# --------------------------------------------------

market.reset()

price = market.price_seq[:, :market.window_size]
mean_price = market.mean_price_seq[:,:market.window_size]
position = market.pos

state = market.build_state(price, mean_price, position)

# --------------------------------------------------
# Create agent
# --------------------------------------------------

agent = DQNAgent(config,state)

# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

agent.model.load_state_dict(torch.load("../dqn_model.pth", map_location=market.device))

# Evaluation mode
agent.model.eval()

# No random exploration
agent.epsilon = 0.0
agent.use_softmax = False


evaluation_rewards = []
episode_stats = []
sell_ratios = []
hold_ratios = []
buy_ratios = []


for episode in range(config.num_episodes):

    market.reset()

    episode_reward = torch.zeros(market.batch_size,dtype=torch.float32,device=market.device)
    episode_profit = torch.zeros(market.batch_size, device=market.device)
    episode_transaction_cost = torch.zeros(market.batch_size, device=market.device)
    episode_risk_penalty = torch.zeros(market.batch_size, device=market.device)
    action_counts = torch.zeros(config.num_actions, dtype=torch.long, device=market.device)

    total_actions = 0

    for t in range(market.window_size - 1, market.num_steps - 1):
        start = t - config.window_size + 1

        current_price = market.price_seq[:, start:t + 1]
        current_mean = market.mean_price_seq[:, start:t + 1]
        position = market.pos

        current_state = market.build_state(current_price, current_mean, position)

        action = agent.act(current_state)

        for a in range(config.num_actions):
            action_counts[a] += (action == a).sum()

        next_state_r, reward, done = market.step(action,t)

        episode_profit += market.last_profit
        episode_transaction_cost += market.last_transaction_cost
        episode_risk_penalty += market.last_risk_penalty

        next_pos = next_state_r[1]
        next_price = market.price_seq[:, start + 1:t + 2]
        next_mean = market.mean_price_seq[:, start + 1:t + 2]
        next_state = market.build_state(next_price, next_mean,next_pos)

        episode_reward += reward
        current_price = next_price
        current_state = next_state
        current_mean = next_mean

        if done.all():
            break

    evaluation_rewards.append(episode_reward.mean().item())

    total_count = action_counts.sum().item()
    sell_ratio = (action_counts[0].item() / total_count)
    hold_ratio = (action_counts[1].item() / total_count)
    buy_ratio = (action_counts[2].item() / total_count)

    episode_stats.append({
        "episode": episode + 1,
        "profit": episode_profit.mean().item(),
        "transaction_cost":
            episode_transaction_cost.mean().item(),
        "risk_penalty":
            episode_risk_penalty.mean().item(),
        "sell": sell_ratio,
        "hold": hold_ratio,
        "buy": buy_ratio
    })

evaluation_rewards = np.array(evaluation_rewards)


### Print Results
print("\nEvaluation Results")
print("-" * 40)
print(f"Mean reward: {evaluation_rewards.mean():.3f}")
print(f"Median reward: {np.median(evaluation_rewards):.3f}")
print(f"Std reward: {evaluation_rewards.std():.3f}")
print(f"Minimum reward: {evaluation_rewards.min():.3f}")
print(f"Maximum reward: {evaluation_rewards.max():.3f}")
print(f"Positive episodes: {(evaluation_rewards > 0).mean():.2%}")


fig = plt.figure(figsize=(13, 6))

episodes = range(1, len(evaluation_rewards) + 1)

plt.plot(episodes, evaluation_rewards, label="Evaluation Reward")

plt.axhline(0, linestyle="--")

plt.xlabel("Evaluation Episode")
plt.ylabel("Cumulative Reward")
plt.title("DQN Evaluation Reward")

plt.legend()
plt.grid()

plt.show()

#--------------------------------------------------------------------------------
selected_episodes = {10, 50}

selected_episodes.update(range(100, config.num_evaluation_episodes + 1, 100))
selected_episodes.add(config.num_evaluation_episodes)

print('\n')
print(
    f"{'Episode':>7} "
    f"{'Profit':>10} "
    f"{'Cost':>10} "
    f"{'Risk':>10} "
    f"{'SELL':>10} "
    f"{'HOLD':>10} "
    f"{'BUY':>10}"
)

print("-" * 90)

for stats in episode_stats:

    if stats["episode"] in selected_episodes:

        print(
            f"{stats['episode']:>7} "
            f"{stats['profit']:>10.2f} "
            f"{stats['transaction_cost']:>10.2f} "
            f"{stats['risk_penalty']:>10.2f} "
            f"{stats['sell']:>10.1%} "
            f"{stats['hold']:>10.1%} "
            f"{stats['buy']:>10.1%}"
        )