import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T

env = gym.make("ALE/Pacman-v5")  # Load Atari Pacman
obs, info = env.reset()  # Reset environment and get initial observation
env.close()

print("Observation Space:", env.observation_space.shape)  # (210, 160, 3) -> RGB Image
print("Action Space:", env.action_space.n)  # 5 possible actions
print("Action meaning:", env.unwrapped.get_action_meanings()) # action meaning
plt.imshow(obs) # show obs as a frame

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # No activation (raw Q-values)

class ReplayBuffer:
    def __init__(self, device, capacity=100000):
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.long).to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, device, action_dim):
        self.device = device
        self.q_network = DQN(action_dim).to(self.device)
        self.target_network = DQN(action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(self.device)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.start_training = 10000
        self.train_every = 4
        self.update_target_every = 1000
        self.step_count = 0

    def select_action(self, state):
        """Epsilon-Greedy Policy"""
        if random.random() < self.epsilon:
            return random.randint(0, action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self):
        self.step_count += 1

        if self.step_count < self.start_training:
            return

        if self.step_count % self.train_every == 0:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        """Save the trained Q-network to a file."""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

def Preprocess():
  env = gym.make("ALE/MsPacman-v5")
  env.reset()
  env = gym.wrappers.ResizeObservation(env, (84, 84))
  env = gym.wrappers.GrayscaleObservation(env)
  env = gym.wrappers.FrameStackObservation(env, 4)
  return env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Preprocess()

action_dim = env.action_space.n
agent = DQNAgent(device, action_dim)

num_episodes = 10000
reward_history = []

with open('training_log.txt', 'w') as f:
    f.write("Episode\tScore\tEpsilon\n")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.train()

            obs = next_obs
            total_reward += reward
            if done:
                break

        reward_history.append(total_reward)
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
        f.write(f"{episode}\t{total_reward}\t{agent.epsilon}\n")

    env.close()

agent.save_model('dqn_pacman_model.pth')
