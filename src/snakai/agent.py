import logging
import random
from collections import deque
from typing import final

import numpy as np
import torch

from .utils import plot
from .game import SnakeGameAI, Direction, Point
from .model import LinearQNet, QTrainer


LOG = logging.getLogger(__name__)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer.get_new_instance(self.model, self.gamma, LR)

    def get_state(self, game):  # noqa
        """
        :param game: The instance of the game environment.
        :type game: `SnakeGameAI`
        :rtype: `np.array`
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight:
            (dir_r and game.is_collision(point_r)) or \
            (dir_l and game.is_collision(point_l)) or \
            (dir_u and game.is_collision(point_u)) or \
            (dir_d and game.is_collision(point_d)),

            # Danger right:
            (dir_r and game.is_collision(point_d)) or \
            (dir_l and game.is_collision(point_u)) or \
            (dir_u and game.is_collision(point_r)) or \
            (dir_d and game.is_collision(point_l)),

            # Danger left:
            (dir_r and game.is_collision(point_u)) or \
            (dir_l and game.is_collision(point_d)) or \
            (dir_u and game.is_collision(point_l)) or \
            (dir_d and game.is_collision(point_r)),

            # Move direction:
            dir_r,
            dir_l,
            dir_u,
            dir_d,

            # Food location:
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        state = np.array(state, dtype=int)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # It will popleft when the MAX_MEMORY will reach.

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # mini_sample is `list` of `tuple`
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 200 - self.n_games
        final_move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            move_index = random.randint(0, 2)
            final_move[move_index] = 1
        else:
            # For example:
            #   state0      -> [0., 0., 1., 0., ..., 0., 1., 1., 0., 1.]
            #   predictions -> [4.3, 5.2, 7.8]
            #   move_index  -> 2
            #   final_move  -> [0, 0, 1]
            state0 = torch.tensor(state, dtype=torch.float)
            predictions = self.model(state0)
            move_index = torch.argmax(predictions).item()
            final_move[move_index] = 1

        return final_move


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
        )

        if done:
            # The game is over, or done, or terminal of game is shutdown,
            # in this case, the game is stopped, so the agent should train
            # with the data stored in long memory. It should learn
            # from its mistakes.
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Saving of the best score registered:
            if score > record:
                record = score
                agent.epsilon = 1 - (record / 100)
                agent.model.save()
                LOG.info("epsilon: " + str(agent.epsilon))
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            total_score += score
            mean_score = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
