import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agent import SimpleAgent, RandomAgent, QTableAgent


def main():
    env = gym.make('CartPole-v1')  # , render_mode="human")
    state, info = env.reset()
    random.seed()
    agent = QTableAgent(env, True)
    rewards = []
    total_reward = 0
    max_episodes = 4000
    for episode in range(max_episodes):
        if episode == max_episodes - 5:
            env = gym.make('CartPole-v1', render_mode="human")
            env.reset()
            env.render()

        while True:
            action = agent.start(state)
            prev_state = state
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            agent.step(state, prev_state, action, reward, terminated, truncated, episode)

            if terminated or truncated:
                print(f"Ep: {episode} Reward: {total_reward}")
                state, info = env.reset()
                rewards.append(total_reward)
                total_reward = 0
                agent.reset()
                break

    agent.finish()
    scatter_rewards = [(idx, r) for idx, r in enumerate(rewards)]
    x, y = zip(*scatter_rewards)
    plt.scatter(x, y)
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), c='r')
    plt.show()
    plt.plot(agent.epsilons)
    plt.show()


if __name__ == '__main__':
    main()
