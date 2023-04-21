import abc
import random
from abc import ABC
from collections import defaultdict
from typing import SupportsFloat

import gymnasium as gym

import gymnasium
import numpy as np


class Agent(ABC):

    def __init__(self, env: gym.Env, learn: bool):
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.learn = learn

    def start(self, state):
        pass

    def step(self, state, prev_state, last_action, reward, terminated: bool,
             truncated: bool, episode: int):
        pass

    def reset(self):
        pass


class SimpleAgent(Agent):
    def step(self, state, prev_state, last_action, reward, terminated: bool,
             truncated: bool, episode: int):
        return 1


class RandomAgent(Agent):
    def start(self, state):
        return random.randint(0, 1)

    def step(self, state, prev_state, last_action, reward, terminated: bool,
             truncated: bool, episode: int):
        return random.randint(0, 1)


class QTableAgent(Agent):
    NUMBER_OF_BINS = 40

    def __init__(self, env: gym.Env, learn: bool):
        super().__init__(env, learn)
        self.Q = np.random.uniform(low=-1,high=1,size=([self.NUMBER_OF_BINS] * self.state_space.shape[0] + [self.action_space.n]))
        self.bins = [np.linspace(max(low, -5), min(high, 5), self.NUMBER_OF_BINS - 1) for low, high in
                     zip(env.observation_space.low, env.observation_space.high)]
        self.epsilon = 1
        self.epsilon_min = 0.0
        self.epsilon_max = 1
        self.epsilon_decay = 0.002

        self.discount = 0.995  # discount factor
        self.lr = 0.3  # learning rate
        self.total_reward = 0
        self.epsilons = []

    def start(self, state):
        return self.act(state)

    def act(self, state):
        explore = self.epsilon > random.uniform(0, 1)
        if explore:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[*self.digitize(state), :])

    def step(self, state, prev_state, last_action, reward, terminated: bool,
             truncated: bool, episode: int):
        digi_state = self.digitize(state)
        digi_prev_state = self.digitize(prev_state)

        max_future_Q = np.max(self.Q[*digi_state])
        old_Q = self.Q[*digi_prev_state, last_action]
        new_Q = (1 - self.lr) * old_Q + self.lr * (
                    reward + self.discount * max_future_Q)
        self.Q[*digi_prev_state, last_action] = new_Q


        self.total_reward += reward
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.epsilon_decay * episode)
        self.epsilons.append(self.epsilon)

    def digitize(self, state):
        result = np.zeros(state.shape, dtype=int)
        for idx, (observation, bin) in enumerate(zip(state, self.bins)):
            result[idx] = np.digitize(observation, bin, right=True)
        return result

    def reset(self):
        self.total_reward = 0

    def finish(self):
        pass
