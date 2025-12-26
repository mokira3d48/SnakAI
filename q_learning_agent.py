"""
Q-learning Agent for Snake Game.
Implements tabular Q-learning with epsilon-greedy exploration.
"""

import os
import pickle
from typing import Tuple, List, Optional
import numpy as np


class QLearningAgent:
    """Q-learning agent for playing Snake."""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01):
        """
        Initialize Q-learning agent.
        
        Args:
            state_size: Size of state representation
            action_size: Number of possible actions
            learning_rate: Alpha, how much to update Q-values
            discount_factor: Gamma, importance of future rewards
            exploration_rate: Epsilon, probability of random action
            exploration_decay: Rate at which epsilon decays
            min_exploration_rate: Minimum value for epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        # State is represented as binary vector, convert to integer index
        self.num_states = 2 ** state_size  # Maximum possible states
        self.q_table = np.zeros((self.num_states, action_size))
        
        # Training statistics
        self.total_rewards = []
        self.scores = []
        self.episode_lengths = []
    
    def get_state_index(self, state: np.ndarray) -> int:
        """
        Convert binary state vector to integer index for Q-table.
        
        Args:
            state: Binary state vector
            
        Returns:
            Integer index for Q-table
        """
        # Convert binary array to integer
        state_index = 0
        for i, value in enumerate(state):
            if value > 0.5:  # Treat as binary
                state_index += 2 ** i
        return min(state_index, self.num_states - 1)
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Chosen action
        """
        state_index = self.get_state_index(state)
        
        # Exploration: choose random action
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        
        # Exploitation: choose best action from Q-table
        # If multiple actions have same Q-value, choose randomly among them
        q_values = self.q_table[state_index]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-table using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_index, action]
        
        if done:
            # If episode is done, there's no next state
            target_q = reward
        else:
            # Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
            max_next_q = np.max(self.q_table[next_state_index])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_index, action] = current_q + self.learning_rate * (
            target_q - current_q
        )
    
    def decay_exploration(self) -> None:
        """Decay exploration rate (epsilon)."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def update_statistics(self, total_reward: float, score: int, 
                          episode_length: int) -> None:
        """
        Update training statistics.
        
        Args:
            total_reward: Total reward for episode
            score: Final score for episode
            episode_length: Number of steps in episode
        """
        self.total_rewards.append(total_reward)
        self.scores.append(score)
        self.episode_lengths.append(episode_length)
    
    def get_statistics(self) -> dict:
        """Get training statistics."""
        if not self.scores:
            return {}
        
        return {
            "mean_score": np.mean(self.scores[-100:]) if len(self.scores) >= 100 
                        else np.mean(self.scores),
            "max_score": np.max(self.scores),
            "mean_reward": np.mean(self.total_rewards[-100:]) 
                          if len(self.total_rewards) >= 100 
                          else np.mean(self.total_rewards),
            "exploration_rate": self.exploration_rate,
            "episodes": len(self.scores)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save Q-table and agent parameters to file.
        
        Args:
            filepath: Path to save file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration_rate': self.min_exploration_rate,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'total_rewards': self.total_rewards,
            'scores': self.scores,
            'episode_lengths': self.episode_lengths
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Load Q-table and agent parameters from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = data['q_table']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.exploration_rate = data['exploration_rate']
            self.exploration_decay = data['exploration_decay']
            self.min_exploration_rate = data['min_exploration_rate']
            self.state_size = data['state_size']
            self.action_size = data['action_size']
            self.total_rewards = data['total_rewards']
            self.scores = data['scores']
            self.episode_lengths = data['episode_lengths']
            
            print(f"Model loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {filepath} not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
        """
        state_index = self.get_state_index(state)
        return self.q_table[state_index].copy()


class ImprovedQLearningAgent(QLearningAgent):
    """Enhanced Q-learning agent with additional features."""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01, 
                 learning_rate_decay: float = 0.9999):
        """
        Initialize improved Q-learning agent.
        
        Args:
            learning_rate_decay: Decay rate for learning rate
        """
        super().__init__(state_size, action_size, learning_rate, 
                        discount_factor, exploration_rate, 
                        exploration_decay, min_exploration_rate)
        
        self.initial_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        
        # For tracking state-action visits
        self.state_action_visits = np.zeros((self.num_states, action_size))
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-table with adaptive learning rate.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        
        # Update visit count
        self.state_action_visits[state_index, action] += 1
        
        # Adaptive learning rate: decrease as we visit state-action more
        visits = self.state_action_visits[state_index, action]
        adaptive_alpha = self.learning_rate / (1 + 0.01 * visits)
        
        # Current Q-value
        current_q = self.q_table[state_index, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_index])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value with adaptive learning rate
        self.q_table[state_index, action] = current_q + adaptive_alpha * (
            target_q - current_q
        )
    
    def decay_learning_rate(self) -> None:
        """Decay learning rate over time."""
        self.learning_rate = max(
            0.01,  # Minimum learning rate
            self.learning_rate * self.learning_rate_decay
        )


if __name__ == "__main__":
    # Test the agent
    agent = QLearningAgent(state_size=12, action_size=4)
    print(f"Agent initialized with {agent.num_states} possible states")
    print(f"Q-table shape: {agent.q_table.shape}")
    
    # Test state conversion
    test_state = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    state_index = agent.get_state_index(test_state)
    print(f"Test state index: {state_index}")
    
    # Test action selection
    action = agent.choose_action(test_state, training=True)
    print(f"Chosen action: {action}")
