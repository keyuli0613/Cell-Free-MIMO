import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Enhanced Q-Network with deeper architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Replay Buffer (unchanged structure, capacity adjusted later)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# 3. DQN Agent with optimizations
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-5, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)  # Reduced capacity

    def select_action(self, state):
        epsilon = max(
            self.epsilon_final,
            self.epsilon_start - (self.epsilon_start - self.epsilon_final)
            * self.steps_done / self.epsilon_decay
        )

        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states)
            best_actions = next_q_values_policy.argmax(1)
            next_q_values_target = self.target_net(next_states)
            next_q_value = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = nn.SmoothL1Loss()(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 4. Training function with enhancements
def train_dqn(num_episodes=100, batch_size=128, target_update=50):
    from rl_environment import RLEnvironment  # Ensure this is correctly imported
    user_data = [5, 6, 12, 14, 15, 18, 11, 8, 9, 6, 3]  # Example user data
    env = RLEnvironment(user_data=user_data)
    state_dim = 4  # [num_UE, cluster_size, precoding_idx, rho_tot]
    action_dim = 5  # Actions 0-4
    agent = DQNAgent(state_dim, action_dim)

    # Normalization parameters (example values, adjust based on your environment)
    min_UE, max_UE = min(user_data), max(user_data)
    NUM_AP = 10  # Example: adjust based on your env
    PRECODING_SCHEMES = 3  # Example: adjust based on your env
    min_rho_tot, max_rho_tot = 0.1, 2.0  # Example range

    rewards = []
    avg_rewards = []

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    for episode in range(num_episodes):
        state = env.reset()
        # Normalize state
        state = np.array([
            (state[0] - min_UE) / (max_UE - min_UE),  # num_UE
            (state[1] - 1) / (NUM_AP - 1),           # cluster_size
            state[2] / (PRECODING_SCHEMES - 1),      # precoding_idx
            (state[3] - min_rho_tot) / (max_rho_tot - min_rho_tot)  # rho_tot
        ])
        total_reward = 0

        for t in range(len(user_data)):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            # Normalize next_state
            next_state = np.array([
                (next_state[0] - min_UE) / (max_UE - min_UE),
                (next_state[1] - 1) / (NUM_AP - 1),
                next_state[2] / (PRECODING_SCHEMES - 1),
                (next_state[3] - min_rho_tot) / (max_rho_tot - min_rho_tot)
            ])
            # Normalize reward (example: adjust max_SE and max_energy as needed)
            max_SE, max_energy = 10.0, 1000.0  # Placeholder values
            lambda_energy = 0.1  # Example: adjust based on env
            normalized_reward = (reward[0] / max_SE) - lambda_energy * (reward[1] / max_energy) if isinstance(reward, tuple) else reward
            
            agent.replay_buffer.push(state, action, normalized_reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            total_reward += normalized_reward
            if done:
                break

        rewards.append(total_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:]) if rewards else total_reward
            avg_rewards.append(avg_reward)
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}")
            agent.update_target()  # More frequent updates

    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.plot([i * 10 for i in range(len(avg_rewards))], avg_rewards, label="Avg Reward (last 10)", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_dqn(num_episodes=200)