<div align="center">

# SnakAI Project
![](https://img.shields.io/badge/Python-3.10.12-blue)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/lastest-2024--07--13-green)
![](https://img.shields.io/badge/contact-dr.mokira%40gmail.com-blueviolet)

</div>

I'll guide you through creating a complete Snake game implementation and then training a Reinforcement Learning (RL) algorithm to play it. We'll start with the game implementation using PyGame, then implement Q-learning to train an AI agent to play Snake autonomously. The approach will be step-by-step, ensuring you understand both game development and RL concepts.

**Keywords**: Snake game, Reinforcement Learning, Q-learning, PyGame, Markov Decision Process, Bellman equation, epsilon-greedy, state representation, reward function, neural networks, deep Q-learning.

# Introduction

Creating a Snake game and training an RL agent to play it is an excellent project that combines game development with artificial intelligence. This project will teach you fundamental concepts in both domains: game loops, collision detection, state representation, reward design, and RL algorithms. By the end, you'll have a fully functional Snake game and an AI that learns to play it through trial and error, just like humans do!

The beauty of this project is that it demonstrates how machines can learn complex behaviors through simple reward mechanisms. The Snake game provides a perfect environment for RL because it has clear rules, discrete states (in our simplified version), and immediate feedback through rewards (eating food) and penalties (dying).

# Literature Review

## Game Development Fundamentals

Before implementing RL, we need a solid game environment. The Snake game follows these basic principles:

1. **Game Loop**: Continuous cycle of processing input, updating game state, and rendering graphics
2. **Collision Detection**: Checking if the snake hits walls, itself, or food
3. **State Management**: Tracking snake position, direction, food location, and score
4. **Rendering**: Displaying game elements on screen

Reference: Rogers, D. (2010). *Mathematics for Game Developers*. Course Technology Press.

## Reinforcement Learning Basics

Reinforcement Learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment:

1. **Markov Decision Process (MDP)**: Formal framework for RL with states (S), actions (A), transitions (P), and rewards (R)
2. **Q-learning**: Model-free RL algorithm that learns action-value function Q(s,a)
3. **Bellman Equation**: Foundation for temporal difference learning: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
4. **Exploration vs Exploitation**: Balancing trying new actions (exploration) with using known good actions (exploitation)

References:
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

## State Representation for Snake

For RL to work effectively, we need to represent the game state in a way the agent can understand:

1. **Grid-based representation**: Divide game area into discrete cells
2. **Feature extraction**: Extract relevant information about danger, food direction, etc.
3. **Image-based representation**: Use raw pixels as input (more advanced)

Reference: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

# Methods for Implementing Snake Game with RL

## Method 1: Tabular Q-learning with Simple State Representation
- **Approach**: Discretize game state into finite grid positions, use Q-table
- **Pros**: Simple to implement, easy to understand, guaranteed convergence
- **Cons**: State space explosion for larger grids, doesn't scale well

## Method 2: Deep Q-Network (DQN) with Neural Network
- **Approach**: Use neural network to approximate Q-function from game pixels or features
- **Pros**: Handles high-dimensional state spaces, can learn complex patterns
- **Cons**: Requires more computational resources, can be unstable during training

## Method 3: Policy Gradient Methods (REINFORCE)
- **Approach**: Directly learn policy without value function
- **Pros**: Better for continuous action spaces, can learn stochastic policies
- **Cons**: Higher variance, slower learning

## Method 4: Double DQN with Experience Replay
- **Approach**: Advanced DQN with separate target network and memory buffer
- **Pros**: More stable training, reduces overestimation bias
- **Cons**: More complex implementation

## Comparative Analysis and Optimal Method Selection

For beginners, I recommend starting with **Method 1: Tabular Q-learning** because:
1. It's the most educational - you can see exactly how Q-values update
2. It works well for small grid sizes
3. It demonstrates core RL concepts without neural network complexities
4. Once mastered, you can graduate to more advanced methods

However, for better performance and scalability, **Method 2: Deep Q-Network** is superior for larger games. We'll implement the simpler Q-learning first for understanding, then discuss how to extend it to DQN.

# Detailed Explanation of Optimal Method: Tabular Q-learning

## Mathematical Foundation

The core of Q-learning is the Bellman equation update:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

Let's break down each term:

1. **$Q(s_t, a_t)$**: Current Q-value for state $s_t$ and action $a_t$
   - Represents expected cumulative reward from taking action $a$ in state $s$
   
2. **$\alpha$**: Learning rate ($0 < \alpha \leq 1$)
   - Controls how much new information overrides old information
   - Example: $\alpha = 0.1$ means we update Q-value by 10% toward new estimate
   
3. **$r_{t+1}$**: Immediate reward received after taking action $a_t$
   - In Snake: +10 for eating food, -10 for dying, -0.1 for each move to encourage efficiency
   
4. **$\gamma$**: Discount factor ($0 \leq \gamma < 1$)
   - Determines importance of future rewards
   - Example: $\gamma = 0.9$ means future rewards are worth 90% of immediate rewards
   
5. **$\max_{a} Q(s_{t+1}, a)$**: Maximum Q-value for next state $s_{t+1}$
   - Estimates best possible future reward from next state
   
6. **$r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)$**: Temporal difference error
   - Difference between estimated value and current Q-value

## Algorithm Steps

1. Initialize Q-table with zeros for all state-action pairs
2. For each episode:
   - Initialize game state
   - While game not over:
     - Choose action using epsilon-greedy policy
     - Execute action, observe reward and next state
     - Update Q-table using Bellman equation
     - Update current state
3. Repeat for many episodes until convergence

## Example Application

Let's walk through a simple example with a 2x2 grid:

- States: 4 possible positions (0,0), (0,1), (1,0), (1,1)
- Actions: Up, Down, Left, Right
- Initial Q-table: All zeros
- Parameters: $\alpha = 0.1$, $\gamma = 0.9$, $\epsilon = 0.1$

If in state (0,0) we take Right, get reward +1, and move to (0,1):
- Old Q((0,0), Right) = 0
- max Q for (0,1) = 0 (initially)
- New Q = 0 + 0.1 * [1 + 0.9*0 - 0] = 0.1

# Implementation Plan

## Step 1: Define and Explain the Problem

We need to create two main components:

1. **Snake Game Environment**: A fully playable Snake game with:
   - Snake that grows when eating food
   - Collision detection with walls and itself
   - Score tracking
   - Visual display
   
2. **RL Agent**: An AI that learns to play Snake by:
   - Observing game state
   - Choosing actions (up, down, left, right)
   - Receiving rewards
   - Updating its policy based on experience

The challenge is designing a state representation that captures enough information for learning while keeping the state space manageable.

## Step 2: Present Methods (Already Done Above)

We've already compared methods and selected tabular Q-learning for initial implementation.

## Step 3: Propose Algorithm

### Main Training Algorithm
```
1. Initialize Q-table with dimensions [num_states, num_actions]
2. Set hyperparameters: alpha, gamma, epsilon, epsilon_decay, min_epsilon
3. For episode = 1 to max_episodes:
   4. Reset game environment
   5. Get initial state s
   6. While game not over:
       7. With probability epsilon: choose random action a
          Else: choose a = argmax(Q[s, :])
       8. Execute action a, get reward r and next state s'
       9. Update Q[s, a] = Q[s, a] + alpha * (r + gamma * max(Q[s', :]) - Q[s, a])
       10. s = s'
       11. Decrease epsilon
   12. Log episode score and length
```

### Game Loop Algorithm
```
1. Initialize pygame, clock, display
2. Create snake with initial position and direction
3. Place food at random position
4. While game running:
   5. Handle user input (for manual play)
   6. Update snake position based on direction
   7. Check collisions:
       - If snake hits wall or itself: game over
       - If snake head at food position: grow snake, increase score, place new food
   8. Draw all game elements
   9. Update display, control frame rate
```

## Step 4: Execute Algorithm Manually

Let's trace through a simplified example:

**Initial Setup**:
- Grid: 3x3
- Snake: [(1,1)] (head at center)
- Food: (0,0)
- Q-table: All zeros
- State: Represented as danger directions and food direction

**Step 1**: State s = [danger_left=0, danger_right=0, danger_up=0, danger_down=0, food_left=1, food_up=1]
- Food is up and left from snake

**Step 2**: Choose action (epsilon-greedy with epsilon=0.1)
- 90% chance: max Q (all zeros, so random)
- Let's say we choose Up

**Step 3**: Execute Up
- Snake moves to (1,0)
- No food eaten, so reward = -0.1
- New state s' = [danger_left=0, danger_right=0, danger_up=1 (wall), danger_down=0, food_left=1, food_up=0]

**Step 4**: Update Q-value
- Old Q[s, Up] = 0
- max Q[s', :] = 0
- New Q = 0 + 0.1 * (-0.1 + 0.9*0 - 0) = -0.01

This shows how the agent learns that moving Up in the initial state gives a small negative reward.

## Step 5: Enumerate Variables

### Game Variables:
- `grid_size`: Tuple (width, height) in cells
- `cell_size`: Pixel size of each cell
- `snake`: List of (x,y) positions
- `food`: (x,y) position of food
- `direction`: Current moving direction (0: up, 1: right, 2: down, 3: left)
- `score`: Current score
- `game_over`: Boolean flag
- `clock`: PyGame clock for frame rate control
- `screen`: PyGame display surface

### RL Agent Variables:
- `q_table`: Numpy array of shape [num_states, num_actions]
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `epsilon_decay`: Rate at which epsilon decreases
- `min_epsilon`: Minimum exploration rate
- `state`: Current state representation
- `action_space`: List of possible actions
- `total_rewards`: List of rewards per episode
- `scores`: List of scores per episode

## Step 6: Specify Classes, Attributes, and Methods

### Class: `SnakeGame`
**Role**: Manages the game environment

**Attributes**:
- `grid_width`, `grid_height`: Grid dimensions
- `cell_size`: Size of each cell in pixels
- `snake`: List of (x,y) positions
- `food`: (x,y) position
- `direction`: Current direction (0-3)
- `score`: Current score
- `game_over`: Boolean
- `screen`: PyGame display
- `clock`: PyGame clock

**Methods**:
- `__init__(width, height, cell_size)`: Initialize game
- `reset()`: Reset game to initial state
- `get_state()`: Return current state as RL-friendly representation
- `step(action)`: Execute action, return (next_state, reward, done)
- `move()`: Update snake position
- `check_collision()`: Check wall/self collisions
- `check_food()`: Check if food eaten
- `place_food()`: Place food at random empty position
- `draw()`: Render game to screen
- `get_game_info()`: Return score and snake length

### Class: `QLearningAgent`
**Role**: Learns to play Snake using Q-learning

**Attributes**:
- `q_table`: Q-value table
- `alpha`: Learning rate
- `gamma`: Discount factor
- `epsilon`: Exploration rate
- `epsilon_decay`: Decay rate
- `min_epsilon`: Minimum epsilon
- `action_space`: Possible actions
- `state_size`: Number of possible states
- `action_size`: Number of possible actions

**Methods**:
- `__init__(state_size, action_size, alpha, gamma, epsilon)`: Initialize agent
- `get_state_index(state)`: Convert state to table index
- `choose_action(state)`: Epsilon-greedy action selection
- `learn(state, action, reward, next_state, done)`: Update Q-table
- `decay_epsilon()`: Reduce exploration rate
- `save_model(path)`: Save Q-table to file
- `load_model(path)`: Load Q-table from file

### Functions:
- `train_agent(agent, env, episodes)`: Train agent for specified episodes
- `test_agent(agent, env, episodes)`: Test trained agent
- `plot_training_results(scores, rewards)`: Visualize training progress
- `play_human(env)`: Allow human to play game

## Step 7: Specify Libraries and Dependencies

### Required Libraries:
1. **PyGame**: Game development library
   - Role: Handles graphics, input, and game loop
   - Utility: Creates game window, draws shapes, handles keyboard input

2. **NumPy**: Numerical computing
   - Role: Efficient array operations
   - Utility: Q-table storage and manipulation, state representation

3. **Matplotlib**: Plotting library
   - Role: Data visualization
   - Utility: Plot training progress, scores, and rewards

4. **Random**: Python standard library
   - Role: Random number generation
   - Utility: Food placement, epsilon-greedy exploration

5. **Time**: Python standard library
   - Role: Time measurement
   - Utility: Frame rate control, training duration measurement
