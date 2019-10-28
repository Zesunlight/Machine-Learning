# DQN

## 特点

### 深度卷积神经网络 (Deep Q-Learning)

利用深度卷积神经网络逼近值函数

### 经验回放 Experience Replay (NIPS DQN)

NIPS DQN在基本的Deep Q-Learning算法的基础上使用了Experience Replay经验池。通过将训练得到的数据储存起来然后随机采样的方法降低了数据样本的相关性。

- 在学习过程中，智能体将数据存储到一个数据库中，然后利用均匀随机采样的方法从数据库中抽取数据，然后利用抽取的数据对神经网络进行训练。

- 经验回放可以打破数据间的关联，克服经验数据的相关性（correlated data）和非平稳分布（non-stationary distribution）问题

- 从以往的状态转移（经验）中随机采样进行训练，使得数据利用率高，因为一个样本被多次使用；连续样本的相关性会使参数更新的方差（variance）比较大，该机制可减少这种相关性。
  

### 独立处理TD偏差 (Nature DQN)

Nature DQN做了一个改进，就是增加Target Q网络。也就是我们在计算目标Q值时使用专门的一个目标Q网络来计算，而不是直接使用预更新的Q网络。这样做的目的是为了减少目标计算与当前值的相关性。

- 独立设置了目标网络来单独处理时间差分算法中的TD偏差

- $\theta_{t+1}=\theta_{t}+\alpha\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta^{-}\right)-Q(s, a ; \theta)\right] \nabla Q(s, a ; \theta)$  梯度下降的更新过程
- TD目标的网络表示为 $\theta^-$ ；计算值函数逼近的网络表示为 $\theta$ 
- 用于动作值函数逼近的网络每一步都更新，而用于计算TD目标的网络每个固定的步数更新一次

## 代码

![1565700671849](pictures\DQN with experience replay.png)

（[源码]([https://github.com/Zesunlight/Machine-Learning/blob/master/Reinforcement%20Learning/DQN%20with%20CartPole.py](DQN with CartPole.py))）

```python
import os
import gym
import numpy as np
import random
import torch as t
import torch.nn.functional as F


class Net(t.nn.Module):
    def __init__(self, observations, actions, init_weights=True):
        super(Net, self).__init__()

        self.fc1 = t.nn.Linear(observations, 12)
        self.fc2 = t.nn.Linear(12, 6)
        self.fc3 = t.nn.Linear(6, actions)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        if os.path.exists('params.pkl'):
            self.load_state_dict(t.load('params.pkl'))
            print('load parameters')
        else:
            for m in self.modules():
                if isinstance(m, t.nn.Linear):
                    t.nn.init.normal_(m.weight)
                    if m.bias is not None:
                        t.nn.init.constant_(m.bias, 0)


class DQN:
    def __init__(self, observations, actions, epsilon=0.1):
        self.net = Net(observations, actions)
        self.actions = actions
        self.memory = []
        self.epsilon = epsilon

    def store(self, state, action, reward, next_state, done):
        self.memory.append([t.tensor(state).float().unsqueeze(0), action, reward, t.tensor(next_state).float().unsqueeze(0), done])

    def execute(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.actions - 1)
        else:
            # select action with max q value
            with t.no_grad():
                action = np.argmax(self.net(t.tensor(state).float().unsqueeze(0))).item()

        return action

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            pass
        else:
            batch = random.sample(self.memory, BATCH_SIZE)
            for state, action, reward, next_state, done in batch:
                with t.no_grad():
                    q_values = self.net(state)
                    if not done:
                        q_update = reward + GAMMA * np.max(self.net(next_state)[0].detach().numpy())
                    else:
                        q_update = reward
                    q_values[0][action] = q_update
                outputs = self.net(state)

                criterion = t.nn.MSELoss()
                optimizer = t.optim.SGD(self.net.parameters(), lr=LEARNING_RATE)
                optimizer.zero_grad()
                loss = criterion(outputs, q_values)
                loss.backward()
                optimizer.step()

            self.epsilon *= EXPLORATION_DECAY


def cart_pole():
    env = gym.make(ENV_NAME).unwrapped
    solver = DQN(env.observation_space.shape[0], env.action_space.n)

    for episode in range(EPISODE_SIZE):
        print(f'episode: {episode}')
        state = env.reset()
        for epoch in range(EPOCH_SIZE):
            action = solver.execute(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            solver.store(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            else:
                solver.experience_replay()

    t.save(solver.net.state_dict(), 'params.pkl')

    average_step = 0
    for _ in range(20):
        state = env.reset()
        done = False
        step = 0
        solver.epsilon = 0
        while not done:
            step += 1
            # env.render()
            solver.epsilon = 0
            action = solver.execute(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        average_step += step
        print(f'step: {step}')
    print(f'average step {average_step / 20}')


if __name__ == '__main__':
    ENV_NAME = "CartPole-v1"
    GAMMA = 0.95
    EPSILON = 0.1
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 1000000
    BATCH_SIZE = 20
    EPISODE_SIZE = 10
    EPOCH_SIZE = 200
    ALPHA = 0.85
    EXPLORATION_DECAY = 0.995

    cart_pole()
```

## 相关论文

### 模型

- **Playing Atari with Deep Reinforcement Learning**
- **Human-level control through deep reinforcement learning** 
- **Deep Reinforcement Learning with Double Q-learning**
- **Prioritized Experience Replay** 
- **Dueling Network Architectures for Deep Reinforcement** 

![Improvements](pictures\Improvements since Nature DQN.jpg)

- Double DQN：目的是减少因为max Q值计算带来的计算偏差，或者称为过度估计（over estimation）问题，用当前的Q网络来选择动作，用目标Q网络来计算目标Q。
- Prioritised replay：也就是优先经验的意思。优先级采用目标Q值与当前Q值的差值来表示。优先级高，那么采样的概率就高。
- Dueling Network：将Q网络分成两个通道，一个输出V，一个输出A，最后再合起来得到Q。

### 改进

- 改进目标Q值计算：[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- 改进随机采样：[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- 改进网络结构，评估单独动作价值：[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) ( 本文为ICML最佳论文之一）
- 改进探索状态空间方式：（1）[Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) （2）[Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models](https://arxiv.org/abs/1507.00814)
- 改变网络结构，增加RNN：[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)（非DeepMind出品，效果很一般，谈不上改进）
- 实现DQN训练的迁移学习：（1）[Policy Distillation](https://arxiv.org/abs/1511.06295) （2） [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](httpss://arxiv.org/abs/1511.06342)
- 解决高难度游戏Montezuma‘s Revenge：[Unifying Count-Based Exploration and Intrinsic Motivation](httpss://arxiv.org/abs/1606.01868)
- 加快DQN训练速度：[Asynchronous Methods for Deep Reinforcement Learning](httpss://arxiv.org/abs/1602.01783) （这篇文章还引出了可以替代DQN的A3C算法，效果4倍Nature DQN）
- 改变DQN使之能够应用在连续控制上面：[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)

## 参考

- https://github.com/qqiang00/reinforce/blob/master/main-RL-QiangYe.pdf
- https://zhuanlan.zhihu.com/p/26052182
- https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
- https://zhuanlan.zhihu.com/p/21477488
- https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
- https://github.com/gsurma/cartpole/blob/master/cartpole.py
