import logging
import random
from collections import deque

import numpy as np
import torch

from .game import SnakeGameAI, Direction, Point


LOG = logging.getLogger(__name__)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        # TODO: model, trainer

    def get_state(self, game):
        ...

    def remember(self, state, action, reward, next_state, done):
        ...

    def train_long_memory(self):
        ...

    def train_short_memory(self, state, action, reward, next_state, done):
        ...

    def get_action(self, state):
        ...


def train():
    """
    Training function.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        # Perform action (move) and get the new state:
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory:
        agent.train_short_memory(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )
        agent.remember(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )  # 00:52:53


if __name__ == '__main__':
    train()
