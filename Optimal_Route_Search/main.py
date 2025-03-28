import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, deque

class GridWorld:
    #初期設定
    def __init__(self):
        self.map = np.array([ #0が道、1が壁、2がゴール
            [0, 0, 0, 1, 1, 2], 
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1]
        ])
        
        self.start_state = (4, 0)
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }
        self.agent_state = (4, 0)  # 初期位置
        
    
    
    @property
    def height(self):
        return len(self.map)
    
    @property
    def width(self):
        return len(self.map[0])
    
    @property
    def shape(self):
        return self.map.shape
    
    def actions(self): #すべての行動選択肢
        return self.action_space
    
    def states(self): #すべてのマス目
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    def next_state(self, state, action): #移動先の場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state
        
        #移動先が道でない場合、next_stateをstateにまた戻す
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height: #枠外
            next_state = state
        elif self.map[ny][nx] == 1:
            next_state = state
            
        return next_state #次の状態を返す
            
    def reward(self, state, action, next_state, step_count):
        y, x = next_state
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return -5
        cell_value = self.map[y][x]
        # 報酬の決定
        if cell_value == 1:  # 壁
            return -1
        elif cell_value == 2:  # ゴール
            return 1 - (step_count / 100)
        else:  # 通路（0）
            return -0.01
    
    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state, step_count)
        # ゴール判定をself.mapの値'2'で判断
        self.done = (self.map[next_state] == 2)  # 修正

        self.agent_state = next_state #ステップによりself.agent_stateを更新
        return next_state, reward, self.done

class QLearningAgent:
    #初期設定
    def __init__(self):
        self.gamma = 0.8
        self.alpha = 0.7
        self.epsilon = 0.4
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        #Q関数を辞書型で定義
        self.Q = defaultdict(lambda: 0)
        
    def set_epsilon_to_zero(self):
        # 学習後、epsilonを0に設定するメソッド
        self.epsilon = 0
        
    #描画関連
    def draw(self, env, final=False, episode=None, step_count=None, ):
        self.agent_map = env.map.copy()
        self.agent_map[state] = 3
        # 文字をマップ上に追加する
        if not final: 
            plt.text(-1.9, 1.00, f"Episode: {episode}", color="red", fontsize=10)
            plt.text(-1.9, 1.5, f"Steps: {step_count}", color="red", fontsize=10)
           
        plt.imshow(self.agent_map, cmap="coolwarm", origin="upper")
        plt.xticks([]) 
        plt.yticks([])
        plt.pause(0.05)
        if env.done:
            plt.pause(1)
        plt.clf()
    
    #行動決定
    def get_action(self, state):
        if np.random.rand() < self.epsilon: #εより小さいとき、ランダムな行動を選択する
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)] #辞書型のfor文。あるstateに対して取りうるすべての行動aを考える
            return np.argmax(qs) #qsのうち、最もq関数が最大になる行動を選択する
        
    def update(self, state, action, reward, next_state, done): #Q関数を設定する
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
            
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
    
env = GridWorld()
agent = QLearningAgent()
episodes = 100
visualize_interval = 5
max_steps = 30

for episode in range(episodes):
    state = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        action = agent.get_action(state) #stateを基に報酬を計算
        next_state, reward, done = env.step(action) #stateでのアクションを基に、次の状態と報酬、ゴール判定を設定
        agent.update(state, action, reward, next_state, done)
        state = next_state
        step_count += 1
    
    if (episode + 1) % visualize_interval == 0:
        print(f"Episode {episode  + 1} / {episodes}")
        state = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            agent.draw(env, False, episode=episode + 1, step_count=step_count)  # 逐次描画
            action = agent.get_action(state)
            state, _, done = env.step(action)
            step_count += 1
        
        print(f"Episode {episode + 1} 終了\n")
        
        if done:
            print(f"ゴールに到達しました。")
            time.sleep(1)

        
print("学習後のエージェントの動き")
agent.set_epsilon_to_zero()
state = env.reset()
done = False

while not done:
    agent.draw(env)
    action = agent.get_action(state)
    state, _, done = env.step(action)

env.draw(env.map, True)

    
        