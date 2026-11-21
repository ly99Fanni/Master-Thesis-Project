from main_program.config import Config
from main_program.environment import Environment
import matplotlib.pyplot as plt

config = Config()
market = Environment(config)

market_price = market.price_seq[0]
fundamental_price = market.mean_price_seq[0]

plt.figure(figsize=(10,6))
plt.plot(market_price, label="Market Price")
plt.plot(fundamental_price, label="Fundamental Price")
plt.xlabel("Time steps")
plt.title("Simulated prices")
plt.legend()
plt.show()
