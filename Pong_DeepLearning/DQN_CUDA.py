import gym
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import cv2
from gym.wrappers import AtariPreprocessing
import torch.nn.functional as F
#更新テスト
#更新テスト2
#更新テスト3

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in states]),  
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack([torch.tensor(ns, dtype=torch.float32, device=device) for ns in next_states]),
            torch.tensor(dones, dtype=torch.float32, device=device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()

        # 畳み込み層（深い特徴抽出）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)  # 新たに256フィルタを追加

        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        # 全結合層
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)  # 出力サイズを増加させる
        self.fc2 = nn.Linear(1024, action_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(0.5)  # ドロップアウト率を50%に設定

    def forward(self, x):
        # 畳み込み層を通す
        x = F.relu(self.bn1(self.conv1(x)))  # バッチ正規化を追加
        x = F.relu(self.bn2(self.conv2(x)))  # バッチ正規化を追加
        x = F.relu(self.bn3(self.conv3(x)))  # バッチ正規化を追加
        x = F.relu(self.bn4(self.conv4(x)))  # バッチ正規化を追加
        x = F.relu(self.bn5(self.conv5(x)))  # バッチ正規化を追加
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # 全結合層を通す
        x = F.relu(self.fc1(x))
        
        # ドロップアウトを適用
        x = self.dropout(x)
        
        return self.fc2(x)
    
class DQNAgent:
    def __init__(self, action_dim, gamma=0.99, lr=0.0005, buffer_size=10000, batch_size=128):
        self.q_network = QNetwork(action_dim).to(device)
        self.target_network = QNetwork(action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.batch_size = batch_size
    
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.action_dim = action_dim
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)
    
env = gym.make('Pong-v4', render_mode=None, frameskip=4)
action_dim = env.action_space.n

agent = DQNAgent(action_dim)

num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 1000
target_update_frequency = 2
learning_frequency = 5

for episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess_frame(state)
    state = torch.tensor(state, dtype=torch.float32, device=device)
    done = False
    total_reward = 0
    step = 0

    #if episode % 10 == 0:
        #env.render("human")  # レンダリングを行う
    
    while not done:
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay)
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_frame(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        agent.replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        state = next_state
    
        if step % learning_frequency == 0:
            agent.update()
        step += 1
        total_reward += reward
    
    if episode % target_update_frequency == 0:
        agent.update_target_network()
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
