import gym
import numpy as np
import random


def epsilon_greedy(state_values, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(state_values)))
    else:
        return np.argmax(state_values)


def sarsa(env, alpha=0.85, gamma=0.9, epsilon=0.1, iterations=100):
    '''
    :param env: environment
    :param alpha: learning rate
    :param gamma: discount factor
    :param epsilon: epsilon value in epsilon greedy policy
    :param iterations: play times
    :return: reward_record, epoch_record

    Reference: https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter05/5.7%20Taxi%20Problem%20-%20SARSA.ipynb
    '''

    q = np.zeros((env.observation_space.n, env.action_space.n))
    epoch_record = []
    reward_record = []

    for i in range(iterations):
        state = env.reset()
        action = epsilon_greedy(q[state], epsilon)
        total_reward = 0
        epoch = 0

        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(q[next_state], epsilon)
            q[state][action] += alpha * \
                (reward + gamma * q[next_state][next_action] - q[state][action])

            action = next_action
            state = next_state
            total_reward += reward

            if done:
                break
            else:
                epoch += 1

        reward_record.append(total_reward)
        epoch_record.append(epoch)

    return reward_record, epoch_record


def q_learning(env, alpha=0.85, gamma=0.9, epsilon=0.1, iterations=100):
    q = np.zeros((env.observation_space.n, env.action_space.n))
    epoch_record = []
    penalty_record = []

    for i in range(iterations):
        # Setting the number of iterations, penalties and reward to zero,
        epochs = 0
        penalties = 0
        done = False

        state = env.reset()
        while not done:
            action = epsilon_greedy(q[state], epsilon)
            next_state, reward, done, info = env.step(action)

            q[state][action] = (1 - alpha) * q[state][action] + \
                               alpha * (reward + gamma * np.max(q[next_state]))
            state = next_state

            if reward == -10:
                penalties += 1
            epochs += 1

        epoch_record.append(epochs)
        penalty_record.append(penalties)

        print("Time steps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))

    return penalty_record, epoch_record


if __name__ == '__main__':
    env = gym.make('Taxi-v2').unwrapped
    reward_record, epoch_record = sarsa(env)
    print(reward_record)
    print(epoch_record)
