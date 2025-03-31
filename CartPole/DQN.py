import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.98, lr=0.0005, buffer_size=10000, batch_size=32):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.batch_size = batch_size

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.action_dim = action_dim
    
    def sync_qnet(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(state).argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 2

agent = DQNAgent(state_dim, action_dim)
epsilon = 0.1

for episode in range(100):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.update()
    
    if episode % 20 == 0:
        agent.sync_qnet()

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
