import random
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, action_to_pos, pos_to_action
from dqn_model import SimpleDQN, DuelingDQN, GDQN


class DQNAgent:
    def __init__(self, config: Config, state):
        self.config = config
        self.use_DuelingDQN = self.config.use_DuelingDQN
        self.use_GDQN = self.config.use_GDQN
        self.use_softmax = self.config.use_softmax

        self.state_size = state.shape[-1] # state = (diff, trend signal, price diff, position) has shape of (batch_size, 4)
        self.num_actions = self.config.num_actions
        self.batch_size = state.shape[0]
        self.gamma = self.config.gamma

        self.learning_rate = self.config.learning_rate

        # Epsilon-greedy policy
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.device = self.config.device

        # Softmax policy
        self.temperature = self.config.temperature
        self.temperature_min = self.config.temperature_min
        self.temperature_decay = self.config.temperature_decay

        if self.use_GDQN:
            self.model = GDQN(input_size=self.state_size,num_actions=self.num_actions,hidden_size=64,).to(self.device)
            self.target_model = GDQN(input_size=self.state_size,num_actions=self.num_actions,hidden_size=64,).to(self.device)

        elif self.use_DuelingDQN:
            self.model = DuelingDQN(self.state_size, self.num_actions, self.config).to(self.device)
            self.target_model = DuelingDQN(self.state_size, self.num_actions, self.config).to(self.device)

        else:
            self.model = SimpleDQN(self.state_size, self.num_actions, self.config).to(self.device)
            self.target_model = SimpleDQN(self.state_size, self.num_action, self.config).to(self.device)


        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()


    def epsilon_greedy(self, state):
        """
                state shape: (batch_size, state_size)
                :return: action shape: (batch_size,)
                """
        state = state.to(dtype=torch.float32, device=self.device)

        batch_size = state.shape[0]

        # Exploration
        if random.random() < self.epsilon:
            return torch.randint(low=0, high=self.num_actions, size=(batch_size,), dtype=torch.long,
                                 device=self.device)

        # Exploitation
        with torch.no_grad():
            q_values = self.model(state)

        action = torch.argmax(q_values, dim=1)

        return action

    def softmax_action(self, state):
        state = state.to(dtype=torch.float32, device = self.device)

        with torch.no_grad():
            q_values = self.model(state)

        scaled_q = q_values / self.temperature
        scaled_q = scaled_q - scaled_q.max(dim=1, keepdim=True).values

        probabilities = torch.softmax(scaled_q, dim = 1)

        action = torch.multinomial(probabilities, num_samples = 1).squeeze(1)

        return action


    def act(self, state):
        if self.use_softmax:
            return self.softmax_action(state)

        return self.epsilon_greedy(state)

    def train_step(self, state, action, reward, next_state, done):
        self.model.train()

        state = state.to(dtype=torch.float32, device=self.device)
        next_state = next_state.to(dtype=torch.float32, device = self.device)
        action = action.to(dtype=torch.long, device=self.device)
        reward = reward.to(dtype=torch.float32, device=self.device)
        done = done.to(dtype=torch.float32, device=self.device)

        # Current Q-value
        q_values = self.model(state)
        current_q = q_values.gather(dim=1, index=action.unsqueeze(1)).squeeze(1)
        # For each batch element, select the Q-value corresponding to the action that was actually taken.
        # But gather() needs the index tensor to have the same number of dimensions as q_values: q_values=(batch_size, num_actions), action=(num_actions,)
        # unsqueeze(1) = Add a dimension at position 1 --> action = (num_actions, 1)
        # squeeze(1) = Remove the dimension at position 1


        # Bellman target
        # y_t = r_t + gamma * (1-done) * max Q(next_state, new_action)

        self.model.eval()

        with torch.no_grad():
            next_q_values = self.target_model(next_state)
            max_next_q = next_q_values.max(dim=1).values
            target_q = reward + self.gamma * max_next_q * (1- done)

        # Train network
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if not self.use_softmax:
            self.epsilon = max(self.epsilon_min, self.epsilon *  self.epsilon_decay)

    def decay_temperature(self):
        if self.use_softmax:
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)


### SANITY CHECK ###
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 5
    state_size = 2
    num_actions = 3

    state = torch.tensor([
        [120.0, 0.0],
        [118.0, 1.0],
        [121.0, -1.0],
        [119.0, 0.0],
        [123.0, 1.0]
    ])

    next_state = torch.tensor([
        [121.0, 1.0],
        [117.5, 0.0],
        [120.0, -1.0],
        [120.5, 1.0],
        [122.0, 0.0]
    ])

    reward = torch.tensor([
        1.0,
        -0.5,
        0.8,
        1.5,
        -1.0
    ])

    done = torch.tensor([
        False,
        False,
        False,
        False,
        True
    ])

    print("\n--- DQN Sanity Check ---")

    config = Config()
    agent = DQNAgent(config, state)

    # --------------------------------------------------
    # 1. Check state shape
    # --------------------------------------------------

    print("State shape:", state.shape)

    # --------------------------------------------------
    # 2. Check network output
    # --------------------------------------------------

    with torch.no_grad():
        q_values = agent.model(state.float())

    print("\nQ-values:")
    print(q_values)

    print("Q-value shape:", q_values.shape)

    # --------------------------------------------------
    # 3. Check action selection
    # --------------------------------------------------

    actions = agent.act(state)

    print("\nActions:")
    print(actions)

    print("Action shape:", actions.shape)

    assert actions.shape == (batch_size,)
    assert actions.min() >= 0
    assert actions.max() < num_actions

    # --------------------------------------------------
    # 4. Check training step
    # --------------------------------------------------

    loss = agent.train_step(
        state,
        actions,
        reward,
        next_state,
        done
    )

    print("\nLoss:")
    print(loss)

    assert loss >= 0

    # --------------------------------------------------
    # 5. Check epsilon decay
    # --------------------------------------------------

    old_epsilon = agent.epsilon

    agent.decay_epsilon()

    print("\nEpsilon:")
    print("Before:", old_epsilon)
    print("After: ", agent.epsilon)

    assert agent.epsilon <= old_epsilon
    assert agent.epsilon >= agent.epsilon_min

    print("\nAll sanity checks passed.")