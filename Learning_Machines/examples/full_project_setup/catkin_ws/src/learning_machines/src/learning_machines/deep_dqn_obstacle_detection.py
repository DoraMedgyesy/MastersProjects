import cv2

from robobo_interface import (
    IRobobo,
    SimulationRobobo
)
import random
import numpy as np
import pickle
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

# Training parameters
EPSILON = 1  # exploration probability
LR = 0.001  # learning rate
GAMMA = 0.99  # discount factor
MOVE_WEIGHT = 1.0  # weight for movement speed in reward
COLLISION_THRESHOLD = 130  # threshold for collision detection
BATCH_SIZE = 50
MEMORY_SIZE = 10_000
TARGET_UPDATE = 10

# Episode parameters
MAX_STEPS = 5  # maximum steps per episode
MIN_STEPS = 5  # minimum steps before early stopping
PATIENCE = 1000  # steps without improvement before early stopping
EPISODES = 5  # number of episodes to train
THRESHOLD = 90  # threshold for collision detection

# ACTIONS = ["forward", "forward_right", "forward_left", "left", "slight_left", "right", "slight_right", "backward"]
ACTIONS = ["forward", "left", "right", "backward"]
ACT_TO_MOTOR = {
    "forward": (100, 100),  # 1
    # "forward_right": (100, 60), # 2
    # "forward_left": (60, 100), # 3
    "left": (-60, 60),  # 4
    # "slight_left": (-30, 30), # 5
    "right": (60, -60),  # 6
    # "slight_right": (30, -30), # 7
    "backward": (-100, -100)  # 8
}
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output (Q-values can be negative)
        return x


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def isolate_green(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green color
    lower_green = (40, 40, 40)
    upper_green = (80, 255, 255)

    # Create mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply mask to original image
    result = cv2.bitwise_and(frame, frame, mask=green_mask)
    return result, green_mask


def analyze_sections(green_mask):
    height, width = green_mask.shape
    section_width = width // 3

    # Split into three sections
    left_section = green_mask[:, :section_width]
    middle_section = green_mask[:, section_width:2 * section_width]
    right_section = green_mask[:, 2 * section_width:]

    # Calculate percentage of green pixels in each section
    def get_green_percentage(section):
        total_pixels = section.size
        green_pixels = np.count_nonzero(section)
        return (green_pixels / total_pixels) * 100

    left_percent = get_green_percentage(left_section)
    middle_percent = get_green_percentage(middle_section)
    right_percent = get_green_percentage(right_section)

    return left_percent, middle_percent, right_percent


def get_state(rob: IRobobo, action, img):
    """Get the current state of the Robobo simulation based on ONLY IR sensor readings."""
    readings = rob.read_irs()
    state = [ir_reading for idx, ir_reading in enumerate(readings) if idx in [2, 3, 4, 5, 7]]

    state.extend(list(ACT_TO_MOTOR[action]))
    state.extend(img)
    return torch.FloatTensor(state)


def perform_action(rob, action):
    """Perform the given action on the Robobo instance."""
    actl, actr = ACT_TO_MOTOR[action]
    rob.move_blocking(actl, actr, 100)


def choose_action(state, policy_net, epsilon=EPSILON):
    """Choose an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    with torch.no_grad():
        q_values = policy_net(state)
        return ACTIONS[q_values.argmax().item()]


def optimize_model(policy_net, target_net, optimizer, memory):
    """Perform a single step of optimization."""
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.stack(batch[0])
    action_batch = torch.tensor([ACTION_TO_IDX[a] for a in batch[1]], device=device)
    reward_batch = torch.tensor(batch[2], device=device)
    next_state_batch = torch.stack(batch[3])
    done_batch = torch.tensor(batch[4], device=device)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
        next_state_values[done_batch] = 0.0

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def _check_food_collision(rob):
    """Check if robot has collected food"""
    try:
        current_food = rob.get_nr_food_collected()
        if not hasattr(_check_food_collision, 'prev_food'):
            _check_food_collision.prev_food = 0

        food_collected = current_food > _check_food_collision.prev_food
        _check_food_collision.prev_food = current_food
        return food_collected
    except:
        return False


def _is_colliding_with_food(camera_frame, left_green, middle_green, right_green):
    """Determine if collision is with food based on green pixels"""
    if camera_frame is None:
        return False

    # If we see significant green pixels, we're likely near/colliding with food
    total_green = left_green + middle_green + right_green
    return total_green > 10  # Threshold for food collision detection


def _calculate_reward(rob, action, left_green, middle_green, right_green,
                      action_history, ir_values):
    """
    Advanced reward function based on the provided implementation
    """
    reward = 0.0
    info = {}

    # Convert action string to index for consistency
    action_to_idx = {"backward": 0, "left": 1, "right": 2, "forward": 3}
    action_idx = action_to_idx.get(action, 3)

    # 1. FOOD COLLECTION - Highest Priority
    food_collision_detected = _check_food_collision(rob)
    if food_collision_detected:
        reward += 100  # Dominant reward
        info['food_collected'] = True
        print(f"🎉 FOOD COLLECTED! Reward: +100")
        return reward, info  # Early return

    # 2. FORWARD MOVEMENT REWARD
    if action == "forward":
        reward += 1  # Simple forward reward
        info['forward_bonus'] = 1

    # 3. VISION-BASED REWARDS
    # REWARD: Keep food in center section
    if middle_green > left_green and middle_green > right_green:
        center_bonus = 6  # Base bonus for centered food
        reward += center_bonus
        info['center_bonus'] = center_bonus
        info['green_centering'] = 'centered'

    # REWARD: Maximize green pixels (core CV_DQN.py formula)
    green_pixel_reward = (middle_green * 1.5) + left_green + right_green
    reward += green_pixel_reward
    info['green_pixels_reward'] = green_pixel_reward
    info['green_distribution'] = [left_green, middle_green, right_green]


    # 4. INTELLIGENT COLLISION DETECTION (food vs obstacle)
    # Check if we're colliding with something (convert IR readings to distances)
    # Assuming higher IR values = closer objects (adjust if needed)
    min_distance = min([ir / 1000.0 for ir in ir_values]) if ir_values else 1.0
    collision_detected = any(sensor > COLLISION_THRESHOLD for sensor in ir_values)

    if collision_detected:
        # Determine if it's food or obstacle collision
        is_food_collision = _is_colliding_with_food(None, left_green, middle_green, right_green)

        if is_food_collision:
            # Colliding with food = good! (but food collection reward handles this)
            info['food_collision'] = True
            info['collision_type'] = 'food'
        else:
            # Colliding with obstacle = bad
            reward -= 10  # Penalty for obstacle collision
            info['obstacle_collision'] = True
            info['collision_penalty'] = -10
            info['collision_type'] = 'obstacle'

            # Additional penalties for pushing into walls
            if action == "forward":
                reward -= 5  # Extra penalty for pushing forward into obstacles


    # 5. REPETITIVE ACTION PENALTIES (prevent circling/backing)
    action_history.append(action_idx)

    # Check for repetitive patterns in recent actions
    if len(action_history) >= 4:
        recent_actions = list(action_history)[-4:]  # Last 4 actions

        # Count occurrences of current action in recent history
        current_action_count = recent_actions.count(action_idx)

        # Penalize continuous backward movement (3+ in last 4)
        if action_idx == 0 and current_action_count >= 3:  # backward
            backward_penalty = 10
            reward -= backward_penalty
            info['backward_spam_penalty'] = -backward_penalty
            print(f"🔄 BACKWARD SPAM: -{backward_penalty}")

        # Penalize continuous turning (3+ same turns in last 4)
        elif action_idx in [1, 2] and current_action_count >= 3:  # left, right
            turn_penalty = 5
            reward -= turn_penalty
            info['turn_spam_penalty'] = -turn_penalty
            print(f"🔄 TURN SPAM: -{turn_penalty}")

    # Store state info
    info['ir_sensors'] = ir_values
    info['action_taken'] = action_idx
    info['min_distance'] = min_distance

    return reward, info


def save_model(model, filename=None):
    """Save the PyTorch model (always as dqn_model_{timestamp}.pth)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.environ.get("SAVE_DIR", "/root/results")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"dqn_model_{timestamp}.pth"  # always use this pattern
    model_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), model_path)
    print(f" Model saved to: {model_path}")
    return model_path


def load_model(model, filename):
    """Load a PyTorch model."""
    filepath = os.path.join("/root/results", "models", filename)
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model


def get_input_size(rob: IRobobo):
    """Get the input size for the neural network."""
    irs = rob.read_irs()
    ir_state = len([ir_reading for idx, ir_reading in enumerate(irs) if idx in [2, 3, 4, 5, 7]])
    num_obs = ir_state + 3 + 2  # 3 from the images and 2 from the action
    return 10


def run_all_actions(rob: IRobobo, load_model_path=None, num_episodes=EPISODES):
    """Run the DQN algorithm on the Robobo."""
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    print("Training DQN Agent...")

    # Get input size after simulation is started
    input_size = get_input_size(rob)
    print(f"Input size: {input_size}, Action size: {NUM_ACTIONS}")

    policy_net = DQNNetwork(state_size=input_size, action_size=NUM_ACTIONS).to(device)
    target_net = DQNNetwork(state_size=input_size, action_size=NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    if load_model_path:
        policy_net = load_model(policy_net, load_model_path)
        target_net.load_state_dict(policy_net.state_dict())

    best_episode_reward = float('-inf')
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")

        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()

        rob.set_phone_tilt(100, 100)

        # Random initial action
        action = np.random.choice(ACTIONS)
        rob.move_blocking(ACT_TO_MOTOR[action][0], ACT_TO_MOTOR[action][1], 100)

        frame = rob.read_image_front()
        green_only, green_mask = isolate_green(frame)
        left, middle, right = analyze_sections(green_mask)

        state = get_state(rob, action, [left, middle, right])
        episode_reward = 0
        steps_without_progress = 0
        last_best_reward = float('-inf')

        # Episode state tracking
        action_history = deque(maxlen=10)

        for step in range(MAX_STEPS):
            action = choose_action(state, policy_net)
            perform_action(rob, action)

            frame = rob.read_image_front()
            green_only, green_mask = isolate_green(frame)
            left, middle, right = analyze_sections(green_mask)

            next_state = get_state(rob, action, [left, middle, right])

            irs = rob.read_irs()
            selected_irs = [ir_reading for idx, ir_reading in enumerate(irs) if idx in [2, 3, 4, 5, 7]]

            # Use the new advanced reward function
            reward, reward_info = _calculate_reward(
                rob, action, left, middle, right,
                action_history, selected_irs
            )

            done = step == MAX_STEPS - 1

            memory.push(state, action, reward, next_state, done)
            state = next_state

            loss = optimize_model(policy_net, target_net, optimizer, memory)

            episode_reward += reward

            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if episode_reward > last_best_reward:
                last_best_reward = episode_reward
                steps_without_progress = 0
            else:
                steps_without_progress += 1

            if steps_without_progress > PATIENCE and step >= MIN_STEPS:
                print(f"Early stopping at step {step} due to lack of progress")
                break

            if step % 50 == 0:  # Reduced logging frequency
                collision_type = reward_info.get('collision_type', 'none')
                food_status = "FOOD!" if reward_info.get('food_collected', False) else "searching"
                print(
                    f"Step {step}: Action {action}, Reward {reward:.2f}, Total {episode_reward:.2f}, Status: {food_status}, Collision: {collision_type}")

                # Print detailed reward breakdown occasionally
                if step % 200 == 0 and reward_info:
                    print(f"  Reward breakdown: {reward_info}")

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

        episode_rewards.append(episode_reward)

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            save_model(policy_net)
            print(f"New best model saved with reward: {best_episode_reward:.2f}")

        print(f"Episode {episode + 1} completed with reward: {episode_reward:.2f}")

    save_model(policy_net, "final_model.pth")
    print("\nTraining Summary:")
    print(f"Best episode reward: {best_episode_reward:.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward std dev: {np.std(episode_rewards):.2f}")

    return best_episode_reward