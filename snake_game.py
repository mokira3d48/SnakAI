"""
Snake Game Environment for Reinforcement Learning.
Implements the classic Snake game with RL-friendly interface.
"""

import pygame
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import sys


class SnakeGame:
    """Snake game environment compatible with RL training."""

    # Direction constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # Action mapping
    ACTION_TO_DIRECTION = {
        0: UP,     # Move up
        1: RIGHT,  # Move right
        2: DOWN,   # Move down
        3: LEFT,   # Move left
    }

    def __init__(self, grid_width: int = 10, grid_height: int = 10,
                 cell_size: int = 40, render: bool = True):
        """
        Initialize Snake game environment.

        Args:
            grid_width: Number of cells horizontally
            grid_height: Number of cells vertically
            cell_size: Pixel size of each cell
            render: Whether to render game visually
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.render = render

        # Game state
        self.snake = []
        self.food = (0, 0)
        self.direction = self.RIGHT
        self.score = 0
        self.game_over = False
        self.snake_length = 3  # Initial length

        # PyGame initialization
        if self.render:
            pygame.init()
            self.screen_width = grid_width * cell_size
            self.screen_height = grid_height * cell_size
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)

            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.GREEN = (0, 255, 0)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 120, 255)
            self.GRAY = (40, 40, 40)

        # Reset to initial state
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset game to initial state.

        Returns:
            Initial state observation
        """
        # Start snake in the middle
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        self.snake = [(start_x, start_y)]

        # Initial direction
        self.direction = self.RIGHT

        # Initial length
        self.snake_length = 3

        # Place first food
        self.place_food()

        # Reset score and game over flag
        self.score = 0
        self.game_over = False

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get current state representation for RL agent.

        Returns:
            State array with danger detection and food direction
        """
        head_x, head_y = self.snake[0]

        # Danger detection in 4 directions
        danger_left = self.is_collision((head_x - 1, head_y))
        danger_right = self.is_collision((head_x + 1, head_y))
        danger_up = self.is_collision((head_x, head_y - 1))
        danger_down = self.is_collision((head_x, head_y + 1))

        # Current direction (one-hot encoded)
        dir_left = 1 if self.direction == self.LEFT else 0
        dir_right = 1 if self.direction == self.RIGHT else 0
        dir_up = 1 if self.direction == self.UP else 0
        dir_down = 1 if self.direction == self.DOWN else 0

        # Food direction
        food_x, food_y = self.food
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0

        # Create state array
        state = np.array([
            danger_left, danger_right, danger_up, danger_down,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)

        return state

    def is_collision(self, point: Tuple[int, int] = None) -> bool:
        """
        Check if point would cause collision.

        Args:
            point: Point to check (if None, check snake head)

        Returns:
            True if collision would occur
        """
        if point is None:
            point = self.snake[0]

        x, y = point

        # Check wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True

        # Check self collision (skip head for current position check)
        if point in self.snake[1:]:
            return True

        return False

    def place_food(self) -> None:
        """Place food at random empty position."""
        empty_cells = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))

        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            # No empty cells - game won!
            self.game_over = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Args:
            action: Action to take (0: up, 1: right, 2: down, 3: left)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.game_over:
            return self.get_state(), 0, True, {"score": self.score}

        # Convert action to direction (prevent 180-degree turns)
        new_direction = self.ACTION_TO_DIRECTION[action]

        # Prevent reverse direction (snake can't turn 180 degrees)
        if not ((new_direction == self.UP and self.direction == self.DOWN) or
                (new_direction == self.DOWN and self.direction == self.UP) or
                (new_direction == self.LEFT and self.direction == self.RIGHT) or
                (new_direction == self.RIGHT and self.direction == self.LEFT)):
            self.direction = new_direction

        # Move snake
        head_x, head_y = self.snake[0]

        if self.direction == self.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == self.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == self.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)

        # Insert new head
        self.snake.insert(0, new_head)

        # Check for collisions
        if self.is_collision():
            self.game_over = True
            reward = -10.0  # Large penalty for dying
            # Trim snake to correct length
            while len(self.snake) > self.snake_length:
                self.snake.pop()
            return self.get_state(), reward, True, {"score": self.score}

        # Check if food eaten
        if new_head == self.food:
            self.score += 10
            reward = 10.0  # Reward for eating food
            self.snake_length += 1
            self.place_food()
        else:
            reward = -0.1  # Small penalty for each move (encourage efficiency)
            # Remove tail if not growing
            if len(self.snake) > self.snake_length:
                self.snake.pop()

        # Check if game won (snake fills entire grid)
        if len(self.snake) == self.grid_width * self.grid_height:
            self.game_over = True
            reward = 100.0  # Big reward for winning

        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "food_position": self.food
        }

        return self.get_state(), reward, self.game_over, info

    def render_frame(self) -> None:
        """Render current game state."""
        if not self.render:
            return

        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Clear screen
        self.screen.fill(self.BLACK)

        # Draw grid lines
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0),
                           (x, self.screen_height), 1)
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y),
                           (self.screen_width, y), 1)

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = self.GREEN if i == 0 else self.BLUE  # Head is different color
            rect = pygame.Rect(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 1)

        # Draw food
        food_x, food_y = self.food
        food_rect = pygame.Rect(
            food_x * self.cell_size + 1,
            food_y * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)
        pygame.draw.rect(self.screen, self.WHITE, food_rect, 1)

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))

        # Draw snake length
        length_text = self.font.render(
            f"Length: {len(self.snake)}", True, self.WHITE
        )
        self.screen.blit(length_text, (10, 40))

        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.SysFont('Arial', 40)
            game_over_text = game_over_font.render("GAME OVER", True, self.RED)
            text_rect = game_over_text.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2)
            )
            self.screen.blit(game_over_text, text_rect)

            restart_text = self.font.render(
                "Press R to restart", True, self.WHITE
            )
            restart_rect = restart_text.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + 50)
            )
            self.screen.blit(restart_text, restart_rect)

        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS for training

    def close(self) -> None:
        """Close game window."""
        if self.render:
            pygame.quit()

    def get_game_info(self) -> Dict[str, Any]:
        """Get current game information."""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "game_over": self.game_over,
            "food_position": self.food,
            "head_position": self.snake[0] if self.snake else (0, 0)
        }

    def play_human(self) -> None:
        """Allow human to play game."""
        if not self.render:
            print("Cannot play human mode without rendering!")
            return

        self.reset()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                    elif not self.game_over:
                        if event.key == pygame.K_UP:
                            self.direction = self.UP
                        elif event.key == pygame.K_DOWN:
                            self.direction = self.DOWN
                        elif event.key == pygame.K_LEFT:
                            self.direction = self.LEFT
                        elif event.key == pygame.K_RIGHT:
                            self.direction = self.RIGHT

            # Game step with current direction
            if not self.game_over:
                # Convert direction to action
                action = self.direction
                self.step(action)

            # Render
            self.render_frame()

        self.close()


if __name__ == "__main__":
    # Test the game with human player
    game = SnakeGame(grid_width=64, grid_height=32, cell_size=20, render=True)
    game.play_human()

