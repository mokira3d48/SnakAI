"""
Snake Game Inference Script
Loads trained Q-learning model and plays Snake autonomously.
"""

import random
import pygame
import numpy as np
import pickle
import time
import json
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

# Import the game and agent classes (simplified versions included for completeness)
class SnakeGame:
    """Simplified Snake game for inference."""
    
    # Direction constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    ACTION_TO_DIRECTION = {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
    
    def __init__(self, grid_width: int = 10, grid_height: int = 10, 
                 cell_size: int = 40, render: bool = True):
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
        self.snake_length = 3
        
        # PyGame initialization
        if self.render:
            pygame.init()
            self.screen_width = grid_width * cell_size
            self.screen_height = grid_height * cell_size + 150  # Extra space for info
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Snake AI Inference")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
            self.small_font = pygame.font.SysFont('Arial', 18)
            
            # Colors
            self.COLORS = {
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'green': (0, 255, 0),
                'red': (255, 0, 0),
                'blue': (0, 120, 255),
                'gray': (40, 40, 40),
                'yellow': (255, 255, 0),
                'purple': (180, 0, 255),
                'orange': (255, 165, 0),
                'dark_green': (0, 180, 0),
            }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state."""
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        self.snake = [(start_x, start_y)]
        self.direction = self.RIGHT
        self.snake_length = 3
        self.place_food()
        self.score = 0
        self.game_over = False
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state representation."""
        if not self.snake:
            return np.zeros(12, dtype=np.float32)
        
        head_x, head_y = self.snake[0]
        
        # Danger detection
        danger_left = 1.0 if self.is_collision((head_x - 1, head_y)) else 0.0
        danger_right = 1.0 if self.is_collision((head_x + 1, head_y)) else 0.0
        danger_up = 1.0 if self.is_collision((head_x, head_y - 1)) else 0.0
        danger_down = 1.0 if self.is_collision((head_x, head_y + 1)) else 0.0
        
        # Current direction
        dir_left = 1.0 if self.direction == self.LEFT else 0.0
        dir_right = 1.0 if self.direction == self.RIGHT else 0.0
        dir_up = 1.0 if self.direction == self.UP else 0.0
        dir_down = 1.0 if self.direction == self.DOWN else 0.0
        
        # Food direction
        food_x, food_y = self.food
        food_left = 1.0 if food_x < head_x else 0.0
        food_right = 1.0 if food_x > head_x else 0.0
        food_up = 1.0 if food_y < head_y else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        
        return np.array([
            danger_left, danger_right, danger_up, danger_down,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)
    
    def is_collision(self, point: Tuple[int, int] = None) -> bool:
        """Check if point would cause collision."""
        if point is None:
            point = self.snake[0] if self.snake else (0, 0)
        
        x, y = point
        
        # Wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        
        # Self collision
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
            self.game_over = True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one game step."""
        if self.game_over:
            return self.get_state(), 0.0, True, {"score": self.score}
        
        # Convert action to direction (prevent 180-degree turns)
        new_direction = self.ACTION_TO_DIRECTION[action]
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
        
        self.snake.insert(0, new_head)
        
        # Check collisions
        if self.is_collision():
            self.game_over = True
            reward = -10.0
            while len(self.snake) > self.snake_length:
                self.snake.pop()
            return self.get_state(), reward, True, {"score": self.score}
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 10
            reward = 10.0
            self.snake_length += 1
            self.place_food()
        else:
            reward = -0.1
            if len(self.snake) > self.snake_length:
                self.snake.pop()
        
        # Check if game won
        if len(self.snake) == self.grid_width * self.grid_height:
            self.game_over = True
            reward = 100.0
        
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "food_position": self.food,
            "head_position": self.snake[0]
        }
        
        return self.get_state(), reward, self.game_over, info
    
    def render_frame(self, q_values: Optional[np.ndarray] = None, 
                    chosen_action: Optional[int] = None,
                    episode_info: Optional[Dict] = None) -> None:
        """Render game frame with optional AI information."""
        if not self.render:
            return
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause
                    return True  # Signal to toggle pause
        
        # Clear screen
        self.screen.fill(self.COLORS['black'])
        
        # Draw game area (top part of screen)
        game_area_height = self.grid_height * self.cell_size
        
        # Draw grid
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS['gray'], 
                           (x, 0), (x, game_area_height), 1)
        for y in range(0, game_area_height, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS['gray'], 
                           (0, y), (self.screen_width, y), 1)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = self.COLORS['green'] if i == 0 else self.COLORS['blue']
            rect = pygame.Rect(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLORS['white'], rect, 1)
            
            # Draw eyes on head
            if i == 0:
                eye_size = self.cell_size // 5
                # Determine eye positions based on direction
                if self.direction == self.RIGHT:
                    left_eye = (rect.right - eye_size - 2, rect.top + eye_size)
                    right_eye = (rect.right - eye_size - 2, rect.bottom - eye_size - 2)
                elif self.direction == self.LEFT:
                    left_eye = (rect.left + 2, rect.top + eye_size)
                    right_eye = (rect.left + 2, rect.bottom - eye_size - 2)
                elif self.direction == self.UP:
                    left_eye = (rect.left + eye_size, rect.top + 2)
                    right_eye = (rect.right - eye_size - 2, rect.top + 2)
                else:  # DOWN
                    left_eye = (rect.left + eye_size, rect.bottom - eye_size - 2)
                    right_eye = (rect.right - eye_size - 2, rect.bottom - eye_size - 2)
                
                pygame.draw.circle(self.screen, self.COLORS['white'], left_eye, eye_size // 2)
                pygame.draw.circle(self.screen, self.COLORS['white'], right_eye, eye_size // 2)
        
        # Draw food
        food_x, food_y = self.food
        food_rect = pygame.Rect(
            food_x * self.cell_size + 1,
            food_y * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, self.COLORS['red'], food_rect)
        pygame.draw.rect(self.screen, self.COLORS['white'], food_rect, 1)
        
        # Draw information panel (bottom part of screen)
        info_panel_top = game_area_height
        info_panel_height = self.screen_height - game_area_height
        
        # Draw panel background
        panel_rect = pygame.Rect(0, info_panel_top, 
                               self.screen_width, info_panel_height)
        pygame.draw.rect(self.screen, self.COLORS['gray'], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS['white'], panel_rect, 2)
        
        # Draw episode info
        if episode_info:
            episode_text = self.font.render(
                f"Episode: {episode_info.get('episode', 0)} | "
                f"Score: {self.score} | "
                f"Length: {len(self.snake)} | "
                f"Steps: {episode_info.get('steps', 0)}",
                True, self.COLORS['white']
            )
            self.screen.blit(episode_text, (10, info_panel_top + 10))
        
        # Draw Q-values if provided
        if q_values is not None and chosen_action is not None:
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            
            # Draw Q-value bars
            bar_width = 60
            bar_spacing = 20
            start_x = 10
            start_y = info_panel_top + 50
            
            max_q = np.max(np.abs(q_values)) if np.any(q_values) else 1.0
            scale = 100 / max_q if max_q > 0 else 100
            
            for i, (action_name, q_val) in enumerate(zip(action_names, q_values)):
                # Bar background
                bar_bg_rect = pygame.Rect(
                    start_x + i * (bar_width + bar_spacing),
                    start_y,
                    bar_width,
                    100
                )
                pygame.draw.rect(self.screen, self.COLORS['black'], bar_bg_rect)
                pygame.draw.rect(self.screen, self.COLORS['white'], bar_bg_rect, 1)
                
                # Bar value (green for positive, red for negative)
                bar_height = min(abs(q_val * scale), 100)
                bar_color = self.COLORS['green'] if q_val >= 0 else self.COLORS['red']
                bar_rect = pygame.Rect(
                    start_x + i * (bar_width + bar_spacing),
                    start_y + 100 - bar_height,
                    bar_width,
                    bar_height
                )
                pygame.draw.rect(self.screen, bar_color, bar_rect)
                
                # Highlight chosen action
                if i == chosen_action:
                    highlight_rect = pygame.Rect(
                        start_x + i * (bar_width + bar_spacing) - 2,
                        start_y - 2,
                        bar_width + 4,
                        104
                    )
                    pygame.draw.rect(self.screen, self.COLORS['yellow'], highlight_rect, 3)
                
                # Action name
                action_text = self.small_font.render(action_name, True, self.COLORS['white'])
                text_rect = action_text.get_rect(
                    center=(start_x + i * (bar_width + bar_spacing) + bar_width // 2,
                           start_y + 110)
                )
                self.screen.blit(action_text, text_rect)
                
                # Q-value
                q_text = self.small_font.render(f"{q_val:.2f}", True, self.COLORS['white'])
                q_rect = q_text.get_rect(
                    center=(start_x + i * (bar_width + bar_spacing) + bar_width // 2,
                           start_y + 130)
                )
                self.screen.blit(q_text, q_rect)
            
            # Draw Q-values title
            q_title = self.small_font.render("Q-values for each action:", 
                                           True, self.COLORS['white'])
            self.screen.blit(q_title, (10, info_panel_top + 30))
        
        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.SysFont('Arial', 40)
            game_over_text = game_over_font.render("GAME OVER", True, self.COLORS['red'])
            text_rect = game_over_text.get_rect(
                center=(self.screen_width // 2, game_area_height // 2)
            )
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = self.font.render(
                "Press SPACE to continue", True, self.COLORS['white']
            )
            restart_rect = restart_text.get_rect(
                center=(self.screen_width // 2, game_area_height // 2 + 50)
            )
            self.screen.blit(restart_text, restart_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
        
        return False  # No pause toggle
    
    def close(self) -> None:
        """Close game window."""
        if self.render:
            pygame.quit()


class QLearningAgent:
    """Q-learning agent for inference."""
    
    def __init__(self, state_size: int = 12, action_size: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.num_states = 2 ** state_size
        self.q_table = np.zeros((self.num_states, action_size))
    
    def get_state_index(self, state: np.ndarray) -> int:
        """Convert binary state to integer index."""
        state_index = 0
        for i, value in enumerate(state):
            if value > 0.5:
                state_index += 2 ** i
        return min(state_index, self.num_states - 1)
    
    def choose_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Choose action using greedy policy.
        
        Returns:
            Tuple of (chosen_action, q_values_for_state)
        """
        state_index = self.get_state_index(state)
        q_values = self.q_table[state_index].copy()
        
        # Find best action(s)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        
        # If multiple actions have same Q-value, choose randomly
        chosen_action = np.random.choice(best_actions)
        
        return chosen_action, q_values
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = data['q_table']
            print(f"Model loaded successfully from {filepath}")
            print(f"Q-table shape: {self.q_table.shape}")
            print(f"Number of non-zero entries: {np.count_nonzero(self.q_table)}")
            
            # Print some statistics about the loaded model
            if 'scores' in data:
                print(f"Training history: {len(data['scores'])} episodes")
                if data['scores']:
                    print(f"Best training score: {np.max(data['scores'])}")
                    print(f"Mean training score (last 100): "
                          f"{np.mean(data['scores'][-100:]) if len(data['scores']) >= 100 else np.mean(data['scores']):.2f}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class SnakeInference:
    """Main class for running Snake game inference with trained model."""
    
    def __init__(self, model_path: str, grid_size: int = 10, 
                 render_delay: float = 0.1, show_q_values: bool = True):
        """
        Initialize inference system.
        
        Args:
            model_path: Path to trained model file
            grid_size: Size of game grid
            render_delay: Delay between frames (seconds)
            show_q_values: Whether to display Q-values during gameplay
        """
        self.model_path = model_path
        self.grid_size = grid_size
        self.render_delay = render_delay
        self.show_q_values = show_q_values
        
        # Initialize agent and environment
        self.agent = QLearningAgent()
        self.env = SnakeGame(grid_width=grid_size, grid_height=grid_size, 
                           cell_size=40, render=True)
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'all_scores': [],
            'all_lengths': [],
            'all_rewards': [],
            'decision_times': [],
            'start_time': None,
            'best_score': 0,
            'best_episode': 0,
            'total_food_collected': 0,
            'total_steps': 0
        }
        
        # Control flags
        self.running = False
        self.paused = False
    
    def load_model(self) -> bool:
        """Load trained model."""
        print(f"Loading model from {self.model_path}...")
        success = self.agent.load_model(self.model_path)
        
        if success:
            print("Model loaded successfully!")
            # Test the model with a quick state
            test_state = np.zeros(12, dtype=np.float32)
            action, q_values = self.agent.choose_action(test_state)
            print(f"Test inference: State {test_state} -> Action {action}, Q-values: {q_values}")
        
        return success
    
    def run_inference(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run inference for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with inference statistics
        """
        if not self.load_model():
            print("Failed to load model. Exiting.")
            return {}
        
        print(f"\nStarting inference for {num_episodes} episodes...")
        print("=" * 60)
        
        self.stats['start_time'] = time.time()
        self.running = True
        
        for episode in range(1, num_episodes + 1):
            if not self.running:
                break
            
            print(f"\nEpisode {episode}/{num_episodes}")
            print("-" * 40)
            
            episode_stats = self.run_episode(episode)
            self.collect_statistics(episode, episode_stats)
            
            # Display episode summary
            print(f"  Score: {episode_stats['score']}")
            print(f"  Steps: {episode_stats['steps']}")
            print(f"  Food collected: {episode_stats['food_collected']}")
            print(f"  Decision time avg: {episode_stats['avg_decision_time']*1000:.1f} ms")
        
        # Final statistics
        self.display_final_statistics()
        
        return self.stats
    
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            episode_num: Episode number
            
        Returns:
            Episode statistics
        """
        # Reset environment
        state = self.env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        food_collected = 0
        decision_times = []
        
        episode_info = {
            'episode': episode_num,
            'steps': 0
        }
        
        while not done and self.running:
            # Handle pause
            if self.paused:
                time.sleep(0.1)
                continue
            
            # Choose action
            start_time = time.time()
            action, q_values = self.agent.choose_action(state)
            decision_time = time.time() - start_time
            decision_times.append(decision_time)

            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Update statistics
            steps += 1
            total_reward += reward
            if reward == 10.0:  # Food collected
                food_collected += 1
            
            # Render with information
            episode_info['steps'] = steps
            if self.show_q_values:
                pause_toggle = self.env.render_frame(q_values, action, episode_info)
                if pause_toggle:
                    self.paused = not self.paused
            else:
                pause_toggle = self.env.render_frame(episode_info=episode_info)
                if pause_toggle:
                    self.paused = not self.paused
            
            # Add small delay for visualization
            time.sleep(self.render_delay)
            
            # Update state
            state = next_state
        
        # Episode completed
        episode_stats = {
            'score': info.get('score', 0),
            'steps': steps,
            'total_reward': total_reward,
            'food_collected': food_collected,
            'avg_decision_time': np.mean(decision_times) if decision_times else 0.0,
            'max_decision_time': np.max(decision_times) if decision_times else 0.0,
            'min_decision_time': np.min(decision_times) if decision_times else 0.0
        }
        
        return episode_stats
    
    def collect_statistics(self, episode_num: int, episode_stats: Dict[str, Any]) -> None:
        """Collect statistics from episode."""
        self.stats['episodes'] += 1
        self.stats['all_scores'].append(episode_stats['score'])
        self.stats['all_lengths'].append(episode_stats['steps'])
        self.stats['all_rewards'].append(episode_stats['total_reward'])
        self.stats['decision_times'].append(episode_stats['avg_decision_time'])
        self.stats['total_food_collected'] += episode_stats['food_collected']
        self.stats['total_steps'] += episode_stats['steps']
        
        # Update best score
        if episode_stats['score'] > self.stats['best_score']:
            self.stats['best_score'] = episode_stats['score']
            self.stats['best_episode'] = episode_num
    
    def display_final_statistics(self) -> None:
        """Display comprehensive inference statistics."""
        if not self.stats['all_scores']:
            print("No statistics to display.")
            return
        
        total_time = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("INFERENCE STATISTICS")
        print("=" * 60)
        print(f"Total episodes: {self.stats['episodes']}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per episode: {total_time/self.stats['episodes']:.2f} seconds")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Total food collected: {self.stats['total_food_collected']}")
        print()
        
        # Score statistics
        scores = self.stats['all_scores']
        print("SCORE STATISTICS:")
        print(f"  Best score: {self.stats['best_score']} (episode {self.stats['best_episode']})")
        print(f"  Average score: {np.mean(scores):.2f}")
        print(f"  Median score: {np.median(scores):.2f}")
        print(f"  Standard deviation: {np.std(scores):.2f}")
        print(f"  Minimum score: {np.min(scores)}")
        print(f"  Maximum score: {np.max(scores)}")
        print(f"  Score range: {np.max(scores) - np.min(scores)}")
        
        # Game length statistics
        lengths = self.stats['all_lengths']
        print("\nGAME LENGTH STATISTICS:")
        print(f"  Average game length: {np.mean(lengths):.2f} steps")
        print(f"  Median game length: {np.median(lengths):.2f} steps")
        print(f"  Longest game: {np.max(lengths)} steps")
        print(f"  Shortest game: {np.min(lengths)} steps")
        
        # Efficiency statistics
        print("\nEFFICIENCY STATISTICS:")
        if self.stats['total_steps'] > 0:
            food_per_step = self.stats['total_food_collected'] / self.stats['total_steps']
            print(f"  Food collected per step: {food_per_step:.4f}")
        
        if self.stats['total_food_collected'] > 0:
            steps_per_food = self.stats['total_steps'] / self.stats['total_food_collected']
            print(f"  Steps per food item: {steps_per_food:.2f}")
        
        # Decision time statistics
        decision_times = self.stats['decision_times']
        if decision_times:
            print("\nDECISION TIME STATISTICS:")
            print(f"  Average decision time: {np.mean(decision_times)*1000:.2f} ms")
            print(f"  Maximum decision time: {np.max(decision_times)*1000:.2f} ms")
            print(f"  Minimum decision time: {np.min(decision_times)*1000:.2f} ms")
        
        # Success rate (score > 0)
        successful_games = sum(1 for score in scores if score > 0)
        success_rate = (successful_games / len(scores)) * 100
        print(f"\nSuccess rate (score > 0): {success_rate:.1f}%")
        
        # Score distribution
        print("\nSCORE DISTRIBUTION:")
        score_bins = [0, 10, 20, 30, 40, 50, 100, float('inf')]
        bin_labels = ['0', '1-10', '11-20', '21-30', '31-40', '41-50', '51-100', '100+']
        
        for i in range(len(score_bins)-1):
            lower = score_bins[i]
            upper = score_bins[i+1]
            if upper == float('inf'):
                count = sum(1 for score in scores if score >= lower)
            else:
                count = sum(1 for score in scores if lower <= score < upper)
            
            percentage = (count / len(scores)) * 100
            print(f"  {bin_labels[i]}: {count} games ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def save_results(self, filename: str = None) -> None:
        """
        Save inference results to file.
        
        Args:
            filename: Output filename (default: inference_results_<timestamp>.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.json"
        
        # Prepare data for saving
        results = {
            'model_path': self.model_path,
            'grid_size': self.grid_size,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'episode_details': []
        }
        
        # Add episode details
        for i in range(len(self.stats['all_scores'])):
            episode_detail = {
                'episode': i + 1,
                'score': self.stats['all_scores'][i],
                'steps': self.stats['all_lengths'][i],
                'total_reward': self.stats['all_rewards'][i],
                'avg_decision_time': self.stats['decision_times'][i]
            }
            results['episode_details'].append(episode_detail)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def plot_results(self) -> None:
        """Plot inference results."""
        if not self.stats['all_scores']:
            print("No data to plot.")
            return
        
        scores = self.stats['all_scores']
        episodes = range(1, len(scores) + 1)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Snake AI Inference Results (Model: {os.path.basename(self.model_path)})', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Scores per episode
        axes[0, 0].plot(episodes, scores, 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].fill_between(episodes, 0, scores, alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Scores per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add horizontal line for average
        avg_score = np.mean(scores)
        axes[0, 0].axhline(y=avg_score, color='r', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_score:.1f}')
        axes[0, 0].legend()
        
        # Plot 2: Game length per episode
        lengths = self.stats['all_lengths']
        axes[0, 1].plot(episodes, lengths, 'g-', linewidth=2, alpha=0.7)
        axes[0, 1].fill_between(episodes, 0, lengths, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Game Length (steps)')
        axes[0, 1].set_title('Game Length per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Score distribution histogram
        axes[1, 0].hist(scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add vertical line for average
        axes[1, 0].axvline(x=avg_score, color='r', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_score:.1f}')
        axes[1, 0].legend()
        
        # Plot 4: Cumulative performance
        cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores) + 1)
        axes[1, 1].plot(episodes, cumulative_avg, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Average Score')
        axes[1, 1].set_title('Cumulative Average Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add final average line
        axes[1, 1].axhline(y=avg_score, color='r', linestyle='--', alpha=0.7, 
                          label=f'Final Average: {avg_score:.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"inference_plot_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        
        plt.show()
    
    def analyze_agent_behavior(self) -> None:
        """Analyze the agent's decision patterns."""
        print("\n" + "=" * 60)
        print("AGENT BEHAVIOR ANALYSIS")
        print("=" * 60)
        
        # Analyze Q-table
        q_table = self.agent.q_table
        
        # Count non-zero entries
        non_zero = np.count_nonzero(q_table)
        total_entries = q_table.size
        coverage = (non_zero / total_entries) * 100
        
        print(f"Q-table coverage: {coverage:.2f}% ({non_zero}/{total_entries} entries)")
        
        # Analyze Q-value statistics
        flat_q = q_table.flatten()
        flat_q_nonzero = flat_q[flat_q != 0]
        
        if len(flat_q_nonzero) > 0:
            print(f"\nQ-value statistics (non-zero only):")
            print(f"  Mean: {np.mean(flat_q_nonzero):.4f}")
            print(f"  Std: {np.std(flat_q_nonzero):.4f}")
            print(f"  Min: {np.min(flat_q_nonzero):.4f}")
            print(f"  Max: {np.max(flat_q_nonzero):.4f}")
            print(f"  Median: {np.median(flat_q_nonzero):.4f}")
        
        # Analyze action preferences
        print(f"\nAction preference analysis:")
        
        # Test with common states
        test_states = [
            np.array([0,0,0,0, 0,1,0,0, 1,0,0,0], dtype=np.float32),  # Moving right, food left
            np.array([0,0,0,0, 0,0,0,1, 0,0,1,0], dtype=np.float32),  # Moving left, food up
            np.array([1,0,0,0, 0,1,0,0, 0,1,0,0], dtype=np.float32),  # Danger left, moving right, food right
            np.array([0,0,1,0, 0,0,1,0, 0,0,0,1], dtype=np.float32),  # Danger up, moving down, food down
        ]
        
        state_descriptions = [
            "Safe, moving right, food left",
            "Safe, moving left, food up",
            "Danger left, moving right, food right",
            "Danger up, moving down, food down"
        ]
        
        for i, (state, desc) in enumerate(zip(test_states, state_descriptions)):
            action, q_values = self.agent.choose_action(state)
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            
            print(f"\n  Test state {i+1}: {desc}")
            print(f"    Chosen action: {action_names[action]}")
            print(f"    Q-values: {[f'{q:.3f}' for q in q_values]}")
            
            # Calculate action probabilities using softmax
            if np.any(q_values):
                exp_q = np.exp(q_values - np.max(q_values))  # For numerical stability
                probs = exp_q / np.sum(exp_q)
                print(f"    Action probabilities: {[f'{p:.3f}' for p in probs]}")
        
        print("=" * 60)
    
    def run_demo_mode(self) -> None:
        """Run a demo mode with enhanced visualization."""
        print("\n" + "=" * 60)
        print("DEMO MODE - Enhanced AI Visualization")
        print("=" * 60)
        print("In this mode, you'll see:")
        print("1. Real-time Q-values for each action")
        print("2. Action probabilities")
        print("3. Decision confidence")
        print("4. Performance metrics")
        print("5. Game state analysis")
        print("\nPress SPACE to pause/resume")
        print("Press ESC to exit")
        print("=" * 60)
        
        if not self.load_model():
            return
        
        # Run a single episode with enhanced visualization
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        break
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
            
            if not self.running:
                break
            
            if self.paused:
                # Display pause message
                self.env.screen.fill(self.env.COLORS['black'])
                pause_font = pygame.font.SysFont('Arial', 50)
                pause_text = pause_font.render("PAUSED", True, self.env.COLORS['yellow'])
                text_rect = pause_text.get_rect(center=(self.env.screen_width//2, 
                                                       self.env.screen_height//2))
                self.env.screen.blit(pause_text, text_rect)
                
                info_font = pygame.font.SysFont('Arial', 24)
                info_text = info_font.render("Press SPACE to resume, ESC to exit", 
                                           True, self.env.COLORS['white'])
                info_rect = info_text.get_rect(center=(self.env.screen_width//2, 
                                                      self.env.screen_height//2 + 60))
                self.env.screen.blit(info_text, info_rect)
                
                pygame.display.flip()
                time.sleep(0.1)
                continue
            
            # Get action and Q-values
            start_time = time.time()
            action, q_values = self.agent.choose_action(state)
            decision_time = time.time() - start_time
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            steps += 1
            
            # Enhanced rendering
            self.render_enhanced_frame(state, action, q_values, decision_time, steps, info)
            
            # Add delay for better visualization
            time.sleep(max(0.05, self.render_delay))
            
            # Update state
            state = next_state
        
        print(f"\nDemo completed!")
        print(f"Final score: {info.get('score', 0)}")
        print(f"Total steps: {steps}")
        
        # Wait for key press before closing
        print("\nPress any key to exit demo mode...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False
            time.sleep(0.1)
    
    def render_enhanced_frame(self, state: np.ndarray, action: int, 
                             q_values: np.ndarray, decision_time: float,
                             steps: int, info: Dict[str, Any]) -> None:
        """Render frame with enhanced information."""
        # First, render the normal game frame
        episode_info = {'episode': 1, 'steps': steps}
        self.env.render_frame(q_values, action, episode_info)
        
        # Now add enhanced information on top
        screen = self.env.screen
        font = self.env.font
        small_font = self.env.small_font
        colors = self.env.COLORS
        
        # Draw enhanced info panel at the bottom
        game_area_height = self.env.grid_height * self.env.cell_size
        info_top = game_area_height + 150  # Below the Q-value display
        
        # Draw state information
        state_labels = [
            "Danger Left", "Danger Right", "Danger Up", "Danger Down",
            "Dir Left", "Dir Right", "Dir Up", "Dir Down",
            "Food Left", "Food Right", "Food Up", "Food Down"
        ]
        
        # Draw state values
        state_text = "State: "
        for i, (label, value) in enumerate(zip(state_labels, state)):
            if value > 0.5:
                state_text += f"{label.split()[0][0]}{label.split()[1][0]} "
        
        state_surface = small_font.render(state_text, True, colors['white'])
        screen.blit(state_surface, (10, info_top))
        
        # Draw decision information
        decision_text = f"Decision time: {decision_time*1000:.1f} ms"
        decision_surface = small_font.render(decision_text, True, colors['white'])
        screen.blit(decision_surface, (10, info_top + 25))
        
        # Draw action confidence
        if np.any(q_values):
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            confidence = (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-10)
            confidence_text = f"Decision confidence: {confidence:.3f}"
            confidence_surface = small_font.render(confidence_text, True, colors['white'])
            screen.blit(confidence_surface, (10, info_top + 50))
        
        # Draw current game info
        game_info = f"Score: {info.get('score', 0)} | Length: {info.get('snake_length', 0)} | Food: {info.get('food_position', (0,0))}"
        game_surface = small_font.render(game_info, True, colors['white'])
        screen.blit(game_surface, (10, info_top + 75))
        
        pygame.display.flip()


def main():
    """Main function for Snake inference."""
    parser = argparse.ArgumentParser(description='Run inference with trained Snake AI')
    parser.add_argument('--model', type=str, default='models/snake_q_learning.pkl',
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--grid_size', type=int, default=20,
                       help='Grid size for Snake game')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between frames (seconds)')
    parser.add_argument('--no_q_values', action='store_true',
                       help='Hide Q-value display during gameplay')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with enhanced visualization')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze agent behavior without playing')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--plot', action='store_true',
                       help='Plot results after inference')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train a model first or specify the correct path.")
        return
    
    # Create inference system
    inference = SnakeInference(
        model_path=args.model,
        grid_size=args.grid_size,
        render_delay=args.delay,
        show_q_values=not args.no_q_values
    )
    
    try:
        if args.analyze:
            # Just analyze the agent without playing
            inference.load_model()
            inference.analyze_agent_behavior()
        
        elif args.demo:
            # Run demo mode
            inference.run_demo_mode()
        
        else:
            # Run normal inference
            stats = inference.run_inference(num_episodes=args.episodes)
            
            # Save results if requested
            if args.save and stats:
                inference.save_results()
            
            # Plot results if requested
            if args.plot and stats:
                inference.plot_results()
            
            # Analyze agent behavior
            inference.analyze_agent_behavior()
    
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
    
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        inference.env.close()
        print("\nInference completed. Goodbye!")


if __name__ == "__main__":
    main()

