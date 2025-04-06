import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Preprocessing function
def preprocess_observation(obs):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((84, 84)),
        T.ToTensor()
    ])
    return transform(obs)

class FrameStack:
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k, 84, 84), dtype=np.uint8)
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()
        obs = preprocess_observation(obs).numpy()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.array(self.frames).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = preprocess_observation(obs).numpy()
        self.frames.append(obs)
        return np.array(self.frames).astype(np.float32), reward, terminated, truncated, info

    def close(self):
        self.env.close()

# Modern CNN Architecture (similar to DQN Nature paper)
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x.clone().detach().to(torch.float32)  # Fix tensor creation issue
        x = x.squeeze(2)  # Remove unnecessary dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.long).to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32).to(self.device),
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
        if random.random() < self.epsilon:
            return random.randint(0, action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/MsPacman-v5")
env = FrameStack(env, 4)

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
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
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
