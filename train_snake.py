"""
Training script for Snake game with Q-learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import time
import argparse
import os
from snake_game import SnakeGame
from q_learning_agent import QLearningAgent, ImprovedQLearningAgent


def train_agent(agent: QLearningAgent, env: SnakeGame, 
                episodes: int = 1000, render_every: int = 100,
                save_every: int = 100, model_path: str = "models/snake_q_learning.pkl") -> Dict[str, List]:
    """
    Train Q-learning agent on Snake game.
    
    Args:
        agent: Q-learning agent
        env: Snake game environment
        episodes: Number of training episodes
        render_every: Render game every N episodes
        save_every: Save model every N episodes
        model_path: Path to save model
        
    Returns:
        Training statistics
    """
    print(f"Starting training for {episodes} episodes...")
    print(f"Initial exploration rate: {agent.exploration_rate}")
    
    # Training statistics
    all_scores = []
    all_rewards = []
    all_lengths = []
    episode_times = []
    
    # Create models directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    for episode in range(1, episodes + 1):
        episode_start_time = time.time()
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Determine if we should render this episode
        render_this_episode = (episode % render_every == 0) and env.render
        
        while not done:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Render if needed
            if render_this_episode:
                env.render_frame()
                time.sleep(0.05)  # Slow down for visualization
        
        # Update agent exploration rate
        agent.decay_exploration()
        
        # Update statistics
        score = info.get("score", 0)
        agent.update_statistics(total_reward, score, steps)
        
        all_scores.append(score)
        all_rewards.append(total_reward)
        all_lengths.append(steps)
        episode_times.append(time.time() - episode_start_time)
        
        # Print progress
        if episode % 10 == 0:
            stats = agent.get_statistics()
            print(f"Episode {episode:4d} | "
                  f"Score: {score:4.0f} | "
                  f"Total Reward: {total_reward:6.2f} | "
                  f"Steps: {steps:4d} | "
                  f"Epsilon: {agent.exploration_rate:.4f} | "
                  f"Mean Score (last 100): {stats['mean_score']:.2f}")
        
        # Save model periodically
        if episode % save_every == 0:
            agent.save_model(model_path.replace(".pkl", f"_ep{episode}.pkl"))
    
    # Save final model
    agent.save_model(model_path)
    
    print(f"\nTraining completed!")
    print(f"Final mean score (last 100 episodes): {np.mean(all_scores[-100:]):.2f}")
    print(f"Best score: {np.max(all_scores)}")
    print(f"Average episode time: {np.mean(episode_times):.2f} seconds")
    
    return {
        "scores": all_scores,
        "rewards": all_rewards,
        "lengths": all_lengths,
        "times": episode_times
    }


def test_agent(agent: QLearningAgent, env: SnakeGame, 
               episodes: int = 10, render: bool = True) -> Dict[str, Any]:
    """
    Test trained agent on Snake game.
    
    Args:
        agent: Trained Q-learning agent
        env: Snake game environment
        episodes: Number of test episodes
        render: Whether to render game
        
    Returns:
        Test statistics
    """
    print(f"\nTesting agent for {episodes} episodes...")
    
    test_scores = []
    test_lengths = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Choose action (no exploration during testing)
            action = agent.choose_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Render if needed
            if render:
                env.render_frame()
                time.sleep(0.1)  # Slow down for visualization
        
        score = info.get("score", 0)
        test_scores.append(score)
        test_lengths.append(steps)
        
        print(f"Test Episode {episode:2d} | "
              f"Score: {score:4.0f} | "
              f"Steps: {steps:4d}")
    
    # Calculate statistics
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    max_score = np.max(test_scores)
    
    print(f"\nTest Results:")
    print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Average Game Length: {np.mean(test_lengths):.2f} steps")
    
    return {
        "scores": test_scores,
        "lengths": test_lengths,
        "mean_score": mean_score,
        "std_score": std_score,
        "max_score": max_score
    }


def plot_training_results(training_stats: Dict[str, List], 
                         window_size: int = 100) -> None:
    """
    Plot training results.
    
    Args:
        training_stats: Training statistics
        window_size: Window size for moving average
    """
    scores = training_stats["scores"]
    rewards = training_stats["rewards"]
    lengths = training_stats["lengths"]
    
    episodes = range(1, len(scores) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Raw scores
    axes[0, 0].plot(episodes, scores, 'b.', alpha=0.3, markersize=2, label='Raw scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Raw Scores per Episode')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average of scores
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, 
                       label=f'{window_size}-episode moving average')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title(f'Moving Average of Scores (window={window_size})')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot 3: Total rewards
    axes[1, 0].plot(episodes, rewards, 'g.', alpha=0.3, markersize=2, label='Total rewards')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].set_title('Total Reward per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Game length
    axes[1, 1].plot(episodes, lengths, 'm.', alpha=0.3, markersize=2, label='Game length')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].set_title('Game Length (Steps) per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional plot: Score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('score_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_agents(env: SnakeGame, episodes: int = 500) -> None:
    """
    Compare different agent configurations.
    
    Args:
        env: Snake game environment
        episodes: Number of episodes per agent
    """
    print("Comparing different agent configurations...")
    
    # Agent configurations to compare
    agent_configs = [
        {
            "name": "Basic Q-learning",
            "agent_class": QLearningAgent,
            "params": {
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 1.0,
                "exploration_decay": 0.995,
                "min_exploration_rate": 0.01
            }
        },
        {
            "name": "High Exploration",
            "agent_class": QLearningAgent,
            "params": {
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 1.0,
                "exploration_decay": 0.999,
                "min_exploration_rate": 0.1  # Higher minimum exploration
            }
        },
        {
            "name": "Low Learning Rate",
            "agent_class": QLearningAgent,
            "params": {
                "learning_rate": 0.01,  # Lower learning rate
                "discount_factor": 0.9,
                "exploration_rate": 1.0,
                "exploration_decay": 0.995,
                "min_exploration_rate": 0.01
            }
        },
        {
            "name": "Improved Q-learning",
            "agent_class": ImprovedQLearningAgent,
            "params": {
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "exploration_rate": 1.0,
                "exploration_decay": 0.995,
                "min_exploration_rate": 0.01,
                "learning_rate_decay": 0.9999
            }
        }
    ]
    
    results = {}
    
    for config in agent_configs:
        print(f"\nTraining {config['name']}...")
        
        # Create agent
        agent = config["agent_class"](
            state_size=12,
            action_size=4,
            **config["params"]
        )
        
        # Train agent
        stats = train_agent(
            agent, env, episodes=episodes, 
            render_every=1000,  # Don't render during comparison
            save_every=episodes + 1  # Don't save during comparison
        )
        
        # Store results
        results[config["name"]] = {
            "scores": stats["scores"],
            "final_mean_score": np.mean(stats["scores"][-100:]),
            "max_score": np.max(stats["scores"])
        }
        
        print(f"{config['name']}: Final mean score = {results[config['name']]['final_mean_score']:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        scores = result["scores"]
        episodes_range = range(1, len(scores) + 1)
        
        # Plot moving average
        window = 50
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(episodes_range[window-1:], moving_avg, label=name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel(f'Score ({window}-episode moving average)')
    plt.title('Comparison of Different Agent Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('agent_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*60)
    print("AGENT COMPARISON SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name:20s} | Final Mean Score: {result['final_mean_score']:6.2f} | "
              f"Max Score: {result['max_score']:4.0f}")
    print("="*60)


def main():
    """Main function for training and testing Snake RL agent."""
    parser = argparse.ArgumentParser(description='Train Q-learning agent for Snake game')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'human', 'compare', 'plot'],
                       help='Mode: train, test, human play, compare agents, or plot results')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--grid_size', type=int, default=24,
                       help='Grid size for Snake game')
    parser.add_argument('--render', action='store_true',
                       help='Render game during training/testing')
    parser.add_argument('--model_path', type=str, default='models/snake_q_learning.pkl',
                       help='Path to save/load model')
    parser.add_argument('--agent_type', type=str, default='basic',
                       choices=['basic', 'improved'],
                       help='Type of Q-learning agent')
    
    args = parser.parse_args()
    
    # Create game environment
    env = SnakeGame(
        grid_width=args.grid_size,
        grid_height=args.grid_size,
        cell_size=40,
        render=args.render
    )
    
    if args.mode == 'human':
        # Human play mode
        print("Starting human play mode...")
        print("Controls: Arrow keys to move, R to restart, ESC to quit")
        env.play_human()
        
    elif args.mode == 'train':
        # Training mode
        if args.agent_type == 'basic':
            agent = QLearningAgent(
                state_size=12,  # From our state representation
                action_size=4,   # Up, Right, Down, Left
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=1.0,
                exploration_decay=0.995,
                min_exploration_rate=0.01
            )
        else:  # improved
            agent = ImprovedQLearningAgent(
                state_size=12,
                action_size=4,
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=1.0,
                exploration_decay=0.995,
                min_exploration_rate=0.01,
                learning_rate_decay=0.9999
            )
        
        # Train agent
        training_stats = train_agent(
            agent, env, 
            episodes=args.episodes,
            render_every=100 if args.render else 1000,  # Render less frequently if not specified
            save_every=100,
            model_path=args.model_path
        )
        
        # Plot results
        plot_training_results(training_stats)
        
    elif args.mode == 'test':
        # Testing mode
        # Load trained agent
        if args.agent_type == 'basic':
            agent = QLearningAgent(state_size=12, action_size=4)
        else:
            agent = ImprovedQLearningAgent(state_size=12, action_size=4)
        
        if agent.load_model(args.model_path):
            # Test agent
            test_stats = test_agent(agent, env, episodes=10, render=args.render)
        else:
            print(f"Could not load model from {args.model_path}")
            print("Please train a model first using --mode train")
    
    elif args.mode == 'compare':
        # Compare different agents
        compare_agents(env, episodes=min(args.episodes, 500))
    
    elif args.mode == 'plot':
        # Plot existing results
        # This would require loading saved statistics
        print("Plot mode requires saved training data.")
        print("Please run training first to generate data.")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
