import os
import math
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from robobo_interface import IRobobo, SimulationRobobo
import pickle

# --- Hyperparameters ---
ACTIONS = ["move_forward", "big_left", "small_left", "big_right", "small_right", "backward"]
STATE_DIM = 8               # number of IR sensors
ACTION_DIM = len(ACTIONS)
HIDDEN_SIZE = 64            # network hidden layer size
BUFFER_CAPACITY = 10000     # replay buffer capacity
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3                   # optimizer learning rate
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000           # decay rate for epsilon
TARGET_UPDATE = 1000        # steps between target network updates
MAX_STEPS = 100       # max steps per episode
EPISODES = 100              # episodes to train
COLLISION_THRESHOLD = 130
MOVE_WEIGHT = 0.5
TIME_PENALTY = 0.1

# --- Neural network definitions ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Replay buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# --- Utility functions ---
def get_state(rob: IRobobo):
    irs = rob.read_irs()
    return np.array([ir / 1000.0 for ir in irs], dtype=np.float32)  # normalize to [0,1]


def compute_reward2(prev_pos, cur_pos, irs, action, last_action=None):
    # forward progress
    dx = cur_pos.x - prev_pos.x
    reward = 0.0
    if action == "move_forward":
        reward += 10 + dx * MOVE_WEIGHT
    else:
        reward += dx * 0.5
    # time penalty
    reward -= TIME_PENALTY
    # collision penalty
    min_ir = min(irs)
    if min_ir > COLLISION_THRESHOLD:
        reward -= 5 + (min_ir - COLLISION_THRESHOLD) / 100.0
    # smoothness
    if last_action is not None:
        reward += 0.5 if action == last_action else -0.2
    # clip
    return float(np.tanh(reward / 10.0))

def compute_reward(irs, action_idx, action_history, left_count, right_count, forward_count, COLLISION_THRESHOLD=130):
    reward = 0.0

    # Add current action to history
    action_history.append(action_idx)

    # --- Collision penalty ---
    collision = any(sensor > COLLISION_THRESHOLD for sensor in irs)
    if collision:
        reward -= 10.0

    # --- Small forward reward ---
    if action_idx == 0:  # move_forward
        reward += 1.0  # small incentive
        forward_count += 1
    else:
        forward_count = 0

    # --- Bonus for 4 forward moves in a row ---
    if forward_count == 4:
        reward += 5.0  # bigger reward for consistency
        forward_count = 0

    # --- No turning penalty (if no left/right in last 10 steps) ---
    if len(action_history) == action_history.maxlen:
        if not any(a in (1, 2, 3, 4) for a in action_history):
            reward -= 5.0

    # --- Left turn streak penalty ---
    if action_idx in (1, 2):  # big_left, small_left
        left_count += 1
    else:
        left_count = 0
    if left_count == 4:
        reward -= 5.0
        left_count = 0

    # --- Right turn streak penalty ---
    if action_idx in (3, 4):  # big_right, small_right
        right_count += 1
    else:
        right_count = 0
    if right_count == 4:
        reward -= 5.0
        right_count = 0

    return reward, left_count, right_count, forward_count



def select_action(policy_net, state, steps_done):
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        qvals = policy_net(torch.tensor(state).unsqueeze(0))
    return int(qvals.argmax(dim=1).item())

# --- Exposed function ---
__all__ = ("run_all_actions", "load_model")

def run_all_actions(rob: IRobobo, num_episodes=EPISODES):
    """
    Train and run a DQN on the given Robobo instance for num_episodes.
    Returns list of episode rewards.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(device)
    target_net = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    steps_done = 0
    rewards_history = []

    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        state = get_state(rob)
        last_action = None
        action_history = deque(maxlen=10)
        left_count = 0
        right_count = 0
        episode_reward = 0.0
        forward_count = 0


        for t in range(MAX_STEPS):
            print(t)
            action_idx = select_action(policy_net, state, steps_done)
            action = ACTIONS[action_idx]

            prev_pos = rob.get_position()
            perform_args = {
                "move_forward": (60, 60, 100),
                "big_left":    (-60, 60, 100),
                "small_left":  (-30, 30, 100),
                "big_right":   (60, -60, 100),
                "small_right": (30, -30, 100),
                "backward":    (-60, -60, 100)
            }[action]
            rob.move_blocking(*perform_args)

            next_state = get_state(rob)
            irs = rob.read_irs()
            cur_pos = rob.get_position()
            done = False

            reward, left_count, right_count, forward_count = compute_reward(irs, action_idx, action_history, left_count, right_count, forward_count)
            print(reward)
            episode_reward += reward

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            last_action = action
            steps_done += 1

            # optimization
            if len(replay_buffer) >= BATCH_SIZE:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(BATCH_SIZE)
                s_b, a_b, r_b, ns_b, d_b = [x.to(device) for x in (s_b, a_b, r_b, ns_b, d_b)]

                q_values = policy_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ns_b).max(1)[0]
                    q_target = r_b + GAMMA * q_next * (1 - d_b)

                loss = nn.functional.smooth_l1_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes} reward: {episode_reward:.2f}")

    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Get the path relative to this script
    save_dir = os.environ.get("SAVE_DIR", "/root/results")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"dqn_model_{timestamp}.pth")
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps_done': steps_done,
        'rewards_history': rewards_history,
    }, model_path)
    print(f"✅ Model saved to: {model_path}")

    return rewards_history


def load_model(model_path):
    """
    Load a saved DQN model and its training state.
    Args:
        model_path: Path to the saved model file
    Returns:
        policy_net: Loaded policy network
        target_net: Loaded target network
        optimizer: Loaded optimizer
        steps_done: Number of steps done in training
        rewards_history: History of rewards
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved state
    checkpoint = torch.load(model_path, map_location=device)

    # Create new networks and optimizer
    policy_net = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(device)
    target_net = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    # Load the saved states
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (
        policy_net,
        target_net,
        optimizer,
        checkpoint['steps_done'],
        checkpoint['rewards_history']
    )


