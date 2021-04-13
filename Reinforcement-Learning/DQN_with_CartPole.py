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
        # self.memory.append([state, action, reward, next_state, done])

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
            # q_values_batch, state_batch = [], []
            for state, action, reward, next_state, done in batch:
                with t.no_grad():
                    q_values = self.net(state)
                    if not done:
                        q_update = reward + GAMMA * np.max(self.net(next_state)[0].detach().numpy())
                    else:
                        q_update = reward
                    q_values[0][action] = q_update
                outputs = self.net(state)

                # state_batch.append(state)
                # q_values_batch.append(q_values)

                criterion = t.nn.MSELoss()
                optimizer = t.optim.SGD(self.net.parameters(), lr=LEARNING_RATE)
                optimizer.zero_grad()
                loss = criterion(outputs, q_values)
                loss.backward()
                optimizer.step()

            self.epsilon *= EXPLORATION_DECAY


def cart_pole():
    env = gym.make(ENV_NAME).unwrapped

    # total = 0
    # for _ in range(100):
    #     state = env.reset()
    #     done = False
    #     step = 0
    #     while not done:
    #         step += 1
    #         env.render()
    #         action = random.randint(0, 1)
    #         next_state, reward, done, _ = env.step(action)
    #     print(f'step: {step}')
    #     total += step
    # print(total / 100)
    # exit()

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
