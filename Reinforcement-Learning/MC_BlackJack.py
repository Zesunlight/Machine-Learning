from collections import defaultdict
import gym


def player_policy(state):
    # return action choice
    # 1 means bid, 0 means stop
    return 1 if state[0] < 20 else 0


def play_game(env):
    # a game, return its all information
    states, rewards, actions = [], [], []
    state = env.reset()
    states.append(state)
    rewards.append(0)

    while True:
        action = player_policy(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

        if done:
            break

    return states, rewards, actions


def Monte_Carlo(env, iterations=1000, gamma=1):
    # approximate state's value
    # gamma, attenuation factor
    value = defaultdict(float)
    state_count = defaultdict(int)

    for _ in range(iterations):
        states, rewards, actions = play_game(env)
        # print(actions)
        # print(states)
        # print(rewards)
        payoff = 0

        for i in range(len(states)-1, -1, -1):
            s = states[i]
            payoff = rewards[i] + gamma * payoff

            if s not in states[i+1:]:
                state_count[s] += 1
                value[s] += (payoff - value[s]) / state_count[s]

    return value


if __name__ == '__main__':
    env = gym.make("Blackjack-v0").unwrapped
    value = Monte_Carlo(env, 500000, 0.9)
    for i in range(10):
        print(value.popitem())


# Reference
# https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python/blob/master/Chapter04/4.6%20BlackJack%20with%20First%20visit%20MC.ipynb
