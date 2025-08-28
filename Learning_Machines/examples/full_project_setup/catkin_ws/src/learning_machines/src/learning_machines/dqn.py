from robobo_interface import (
    IRobobo,
    SimulationRobobo
)
import random
from collections import defaultdict
import numpy as np
import pickle
import os
from datetime import datetime

ACTIONS = ["move_forward", "big_left", "small_left", "big_right", "small_right", "backward"]
Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

# Training parameters
EPSILON = 0.2  # exploration probability
LR = 0.1  # learning rate
GAMMA = 0.99  # discount factor
MOVE_WEIGHT = 0.5  # weight for movement speed in reward
COLLISION_THRESHOLD = 130  # threshold for collision detection

# Episode parameters
MAX_STEPS = 5  # maximum steps per episode
MIN_STEPS = 200  # minimum steps before early stopping
# Putting patience = max_steps so i dont have to remove the early stopping condition later.
PATIENCE = 1_0000  # steps without improvement before early stopping
EPISODES = 1  # number of episodes to train


def get_state(rob: IRobobo):
    """Get the current state of the Robobo simulation based on ONLY IR sensor readings.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
    Returns:
        A tuple representing the normalized IR sensor readings used as the state.

    After checking a bit, we don't even use this bitch (during the simulation). We might in the future tho, so I'll leave it here.
    For now it will return the same irs reading as we use in the simulation. This should improve behaviour.
    """
    irs = rob.read_irs()
    return tuple(round(ir_reading, 2) for ir_reading in irs)

    # Normalize IR values to [0,1] range for better state representation
    normalized_irs = [min(ir / 100, 1.0) for ir in irs]
    print(f"Raw IRs: {irs}")
    print(f"Normalized IRs: {normalized_irs}")

    return tuple(round(normalized_irs[i], 2) for i in range(len(normalized_irs)))


def perform_action(rob, action):
    """Perform the given action on the Robobo instance.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
        action: The action to perform, one of the predefined ACTIONS.
    """
    if action == "move_forward":
        rob.move_blocking(60, 60, 100)
    elif action == "big_left":
        rob.move_blocking(-60, 60, 100)
    elif action == "small_left":
        rob.move_blocking(-30, 30, 100)
    elif action == "big_right":
        rob.move_blocking(60, -60, 100)
    elif action == "small_right":
        rob.move_blocking(30, -30, 100)
    # elif action == "stop":
    #     rob.move_blocking(0, 0, 100)
    elif action == "backward":
        rob.move_blocking(-60, -60, 100)


def choose_action(state):
    """Choose an action based on the current state using epsilon-greedy strategy.
    Args:
        state: The current state of the Robobo, represented as a tuple of IR sensor readings.
    Returns:
        A string representing the chosen action.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        # Add small random noise to break ties
        q_values = Q[state]
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)


def update_q(state, action, reward, next_state):
    """Update the Q-value for the given state-action pair using the Q-learning update rule.
    Args:
        state: The current state of the Robobo, represented as a tuple of IR sensor readings.
        action: The action taken in the current state.
        reward: The reward received after taking the action.
        next_state: The state reached after taking the action.
    """
    max_next = max(Q[next_state].values())
    Q[state][action] += LR * (reward + GAMMA * max_next - Q[state][action])


def compute_reward(rob, prev_position, current_position, irs, action):
    """Compute the reward based on the Robobo's movement and sensor readings.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
        prev_position: The previous position of the Robobo.
        current_position: The current position of the Robobo.
        irs: The IR sensor readings from the Robobo.
    Returns:
        A float representing the computed reward.
    """
    reward = 0

    # Movement reward: reward based on distance moved
    dx = current_position.x - prev_position.x
    dy = current_position.y - prev_position.y
    distance_moved = (dx ** 2 + dy ** 2) ** 0.5

    # Reward for movement with diminishing returns
    if distance_moved > 0:
        # maybe try exponential
        reward += 5 + (distance_moved * MOVE_WEIGHT)
    else:
        reward -= 1

    if action == "move_forward":
        reward += 1

        # Collision penalty: check front sensors for obstacles
    # front_sensors = [irs[i] for i in [3, 4, 5, 6, 7]]
    if any(sensor > COLLISION_THRESHOLD for sensor in irs):
        reward -= 5  # Significant penalty for collisions

    return reward


import os
from collections import defaultdict

def save_q_table(q_table, filename="models/qtable3.txt"):
    """
    Serialize your Q (a defaultdict) to a human-readable .txt file.
    By default it writes to ./models/qtable.txt relative to cwd.
    """
    # 1) ensure the folder exists
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # 2) convert to a plain dict (so repr(json) is clean)
    plain = {tuple(state): actions for state, actions in q_table.items()}

    # 3) write the Python dict literal to the file
    with open(filename, "w") as f:
        f.write(repr(plain))

    print(f"Q-table saved (text) to {filename}")

def run_all_actions(rob: IRobobo, load_model=None, num_episodes=EPISODES):
    """Run the Q-learning algorithm on the Robobo.

    Args:
        rob: The Robobo instance
        load_model: Optional filename of a saved model to load
        num_episodes: Number of episodes to train
    """
    global Q

    # if load_model:
    #     Q = load_q_table(load_model)

    best_episode_reward = float('-inf')
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()

        # start_position = rob.get_position()
        state = get_state(rob)
        episode_reward = 0
        steps_without_progress = 0
        last_best_reward = float('-inf')

        for step in range(MAX_STEPS):
            start_position = rob.get_position()
            action = choose_action(state)
            perform_action(rob, action)

            next_state = get_state(rob)
            irs = rob.read_irs()
            new_position = rob.get_position()

            reward = compute_reward(rob, start_position, new_position, irs, action)
            update_q(state, action, reward, next_state)

            episode_reward += reward

            # Check for learning progress
            if episode_reward > last_best_reward:
                last_best_reward = episode_reward
                steps_without_progress = 0
            else:
                steps_without_progress += 1

            # Early stopping if no progress for too long and minimum steps reached
            if steps_without_progress > PATIENCE and step >= MIN_STEPS:
                print(f"Early stopping at step {step} due to lack of progress")
                break

            state = next_state
            start_position = new_position

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: Action {action}, Reward {reward:.2f}, Total Reward {episode_reward:.2f}")

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

        episode_rewards.append(episode_reward)

        # Save if this is the best episode so far
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            # save_q_table(Q, f"best_model_episode_{episode + 1}.pkl")
            print(f"New best model saved with reward: {best_episode_reward:.2f}")

        print(f"Episode {episode + 1} completed with reward: {episode_reward:.2f}")

    # Save the final Q-table
    SAVE_DIR = os.environ.get("SAVE_DIR", "/root/results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_q_table(Q, os.path.join(SAVE_DIR, "final_model.pkl"))

    # Print training summary
    print("\nTraining Summary:")
    print(f"Best episode reward: {best_episode_reward:.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward std dev: {np.std(episode_rewards):.2f}")
    # at the very end, after Q is fully trained:




    return best_episode_reward