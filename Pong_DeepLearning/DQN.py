import gym
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import numpy as np
import cv2
from gym.wrappers import AtariPreprocessing

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32) # データ型をfloat32(float64では計算コストが高い)
        next_state = np.array(next_state, dtype=np.float32)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None) 
        self.buffer[self.position] = (state, action, reward, next_state, done) 
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) #self_bufferからrandomにbatch_size個取得
        states, actions, rewards, next_states, dones = zip(*batch) #batchを個々のリストに分割

        return (
            #statesはリストの中に各配列が格納されている。スカラー値はそのままtensorによってテンソル化する
            #ベクトル量は一度tensor化させることでリスト内に個々のテンソルを作成、そしてstackにより1つのテンソルとして結合(NNに入力するための下処理)
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]),  
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states]),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
        
class QNetwork(nn.Module):
    def __init__(self, action_dim):
        #畳み込み層
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #全結合層
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    

        
    
    
class DQNAgent:
    def __init__(self, action_dim, gamma=0.99, lr=0.0001, buffer_size=5000, batch_size=32):
        self.q_network = QNetwork(action_dim)
        self.target_network = QNetwork(action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict()) #ターゲットネットワークに現在のQネットワークを更新する
        self.target_network.eval() #推論モード、ターゲットネットワークの学習を防ぐ
        self.batch_size = batch_size
    
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.action_dim = action_dim
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) #次元を追加する。バッチ処理を想定
            with torch.no_grad(): #不要な勾配計算を防ぐ
                return self.q_network(state).argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        #Q学習のターゲット
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0] #各行ごとの最大値を取得した配列
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) #'1'で列方向にインデックス方向を指定し、行動インデックスに従い指定された行動のQ値を取り出す
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def preprocess_frame(frame):
    """Pongのフレームをグレースケール & 84x84にリサイズ"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # グレースケール変換
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # 84x84 にリサイズ
    frame = frame.astype(np.float32) / 255.0  # 正規化
    return np.expand_dims(frame, axis=0)  # チャネル次元を追加
    
env = gym.make('Pong-v4', render_mode="human", frameskip=4)
state_dim = 84 * 84
action_dim = env.action_space.n

agent = DQNAgent(action_dim)

num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 5000
target_update_frequency = 5
learning_frequency = 5

for episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess_frame(state)  # 画像処理
    state = torch.tensor(state, dtype=torch.float32)  # (1, 1, 84, 84) にする
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / num_episodes)
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_frame(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
    
        if step % learning_frequency == 0:  # 一定ステップごとに学習
            agent.update()  # バッチ学習を行う
        step += 1
        total_reward += reward
    
    if episode % target_update_frequency == 0:
        agent.update_target_network()
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
        
        
