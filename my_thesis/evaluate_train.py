from main_program.environment import Environment
from main_program.config import Config
from main_program.evaluator import Evaluator, set_seed

set_seed(42)

config = Config()
config.num_steps = 100
config.agent_type = 4

market = Environment(config)


# Trained with uniform replay
# train_weight = "agent1_model_ep1000_cost0.0003_lookback30_uniform_DuelingDQN_softmax_final_20261120_224059.pth"
# train_weight = "agent2_model_ep1000_cost0.0003_lookback30_uniform_DuelingDQN_softmax_final_20261121_024049.pth"
# train_weight = "agent3_model_ep1000_cost0.0003_lookback30_uniform_DuelingDQN_softmax_final_20261120_012937.pth"
# train_weight = "agent4_model_ep1000_cost0.0003_uniform_DuelingDQN_softmax_final_20261120_025839.pth"


# Trained with experience replay
# train_weight = "agent1_model_ep1000_cost0.0003_lookback30_prioritized_DuelingDQN_softmax_final_20261120_075725.pth"
# train_weight = "agent2_model_ep1000_cost0.0003_lookback30_prioritized_DuelingDQN_softmax_final_20261121_040816.pth"
# train_weight = "agent3_model_ep1000_cost0.0003_lookback30_prioritized_DuelingDQN_softmax_final_20261120_101809.pth"
train_weight = "agent4_model_ep1000_cost0.0003_prioritized_DuelingDQN_softmax_final_20261121_000835.pth"


evaluator = Evaluator(market=market,
                      config=config,
                      model_path=train_weight)

evaluator.evaluate(debug=False)

# evaluator.plot_price_with_positions2(episode_idx=1)

evaluator.plot_price_with_positions4(
    episode_idx=9,
    batch_idx=10
)

evaluator.plot_full_price_with_positions(batch_idx=10)


statistics, sharpe_summary, action_summary = (
    evaluator.summary_tables2()
)





# -----------------------------------------------------------------------------------------------------------------------
import numpy as np

prices = evaluator.price_history.detach().cpu().numpy()
fundamentals = evaluator.mean_price_history.detach().cpu().numpy()

relative_deviation = (
    (prices - fundamentals) / fundamentals * 100
)

# Find global maximum
max_idx = np.unravel_index(
    np.argmax(relative_deviation),
    relative_deviation.shape
)

# Find global minimum
min_idx = np.unravel_index(
    np.argmin(relative_deviation),
    relative_deviation.shape
)

print(
    f"Maximum: {relative_deviation[max_idx]:.4f}% "
    f"| Batch: {max_idx[0]} | Step: {max_idx[1]}"
)

print(
    f"Minimum: {relative_deviation[min_idx]:.4f}% "
    f"| Batch: {min_idx[0]} | Step: {min_idx[1]}"
)