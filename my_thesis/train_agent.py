from main_program.config import Config
from main_program.environment import Environment
from main_program.trainer import Trainer


config = Config()
market = Environment(config)

trainer = Trainer(
    config=config,
    market=market,
    save_weight=True,
    save_plot=True,
)

episode_stats = trainer.train()
trainer.save_results()
trainer.save_plot_training()

# -----------------------------------------------------------------------------------------
#
# agent_types = [1, 2, 3, 4]
#
# for agent_type in agent_types:
#
#     print(f"\n{'=' * 60}")
#     print(f"TRAINING AGENT TYPE {agent_type}")
#     print(f"{'=' * 60}")
#
#     config = Config()
#     config.replay_type = "prioritized"
#     config.agent_type = agent_type
#
#     market = Environment(config)
#
#     trainer = Trainer(
#         config=config,
#         market=market,
#         save_weight=True,
#         save_plot=True,
#     )
#
#     episode_stats = trainer.train()
#
#     trainer.save_results()
#     trainer.save_plot_training()
#
#     print(f"Agent type {agent_type} completed.")

# ---------------------------------------------------------------------------------------------------

# for replay_type in ["uniform", "prioritized"]:
#
#     config = Config()
#     config.replay_type = replay_type
#
#     market = Environment(config)
#
#     trainer = Trainer(
#         config=config,
#         market=market,
#         save_weight=True,
#         save_plot=True,
#     )
#
#     print(f"\nTraining with {replay_type} experience replay")
#
#     episode_stats = trainer.train()
#     trainer.save_results()
#     trainer.save_plot_training()


# runs = [
#     (1, "uniform"),
#     (4, "prioritized"),
#     (2, "uniform"),
#     (2, "prioritized"),
# ]
#
# for agent_type, replay_type in runs:
#     config = Config()
#     config.agent_type = agent_type
#     config.replay_type = replay_type
#
#     market = Environment(config)
#     trainer = Trainer(
#         config=config,
#         market=market,
#         save_weight=True,
#         save_plot=True,
#     )
#
#     episode_stats = trainer.train()
#     trainer.save_results()
#     trainer.save_plot_training()
#
#     print(f"Agent type {agent_type} with {replay_type} replay completed.")