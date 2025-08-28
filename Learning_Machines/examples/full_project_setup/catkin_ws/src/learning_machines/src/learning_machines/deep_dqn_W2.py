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
EPSILON = 0.1  # exploration probability
LR = 0.001    # learning rate
GAMMA = 0.99    # discount factor
MOVE_WEIGHT = 1.0  # weight for movement speed in reward
COLLISION_THRESHOLD = 130  # threshold for collision detection
BATCH_SIZE = 50
MEMORY_SIZE = 10_000
TARGET_UPDATE = 10

# Episode parameters
MAX_STEPS = 500  # maximum steps per episode
MIN_STEPS = 500   # minimum steps before early stopping
PATIENCE = 1000    # steps without improvement before early stopping
EPISODES = 500   # number of episodes to train
THRESHOLD = 90  # threshold for collision detection

# TODO: Add a reward for moving forward
# ACTIONS = ["forward", "forward_right", "forward_left", "left", "slight_left", "right", "slight_right", "backward"]
ACTIONS = ["forward", "left", "right", "backward"]
ACT_TO_MOTOR = {
    "forward": (100, 100), # 1
    # "forward_right": (100, 60), # 2
    # "forward_left": (60, 100), # 3
    "left": (-60, 60), # 4
    # "slight_left": (-30, 30), # 5
    "right": (60, -60), # 6
    # "slight_right": (30, -30), # 7
    "backward": (-100, -100) # 8
}
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size )
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
    middle_section = green_mask[:, section_width:2*section_width]
    right_section = green_mask[:, 2*section_width:]
    
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
    """Get the current state of the Robobo simulation based on ONLY IR sensor readings.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
    Returns:
        A tuple representing the normalized IR sensor readings used as the state.
        
    After checking a bit, we don't even use this bitch (during the simulation). We might in the future tho, so I'll leave it here.    
    For now it will return the same irs reading as we use in the simulation. This should improve behaviour.
    """
    readings = rob.read_irs()
    state = [ir_reading for idx, ir_reading in enumerate(readings) if idx in [2,3,4,5,7]]
    # print(state)

    state.extend(list(ACT_TO_MOTOR[action]))
    # print(state)
    state.extend(img)
    return torch.FloatTensor(state)


def perform_action(rob, action):
    """Perform the given action on the Robobo instance.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
        action: The action to perform, one of the predefined ACTIONS.
    """
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

def compute_reward(rob, irs, action, left, middle, right):
    """Compute the reward based on the Robobo's movement and sensor readings.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
        irs: The IR sensor readings from the Robobo.
    Returns:
        A float representing the computed reward.
    """
    reward = 0

    food = middle+left+right
    if food > 5:
        reward += 1

    else:
        collision = any(sensor > COLLISION_THRESHOLD for sensor in irs)
        if collision:
            reward -= 10.0


    if action == "forward":
        reward += 2

    food_collected = rob.get_nr_food_collected()
    reward += food_collected*5

    #
    # # Encourage the agent to keep the food in the middle of the screen
    # if middle > left or middle > right:
    #     reward += 2
    #
    # # or Just maximise the amount of green in the picture from the camera
    # reward += (middle * 1.5) + left + right

    return reward

def compute_reward2(rob, irs, action, left, middle, right, right_count, left_count, action_history):
    """Compute the reward based on the Robobo's movement and sensor readings.
    Args:
        rob: The Robobo instance (either SimulationRobobo or HardwareRobobo)
        irs: The IR sensor readings from the Robobo.
    Returns:
        A float representing the computed reward.
    """
    reward = 0
    action_history.append(action)

    food = middle+left+right
    if food > 5:
        reward += 1

    else:
        collision = any(sensor > COLLISION_THRESHOLD for sensor in irs)
        if collision:
            reward -= 10.0
            if action == "forward":
                reward -= 3.0  # discourage pushing into wall
        else:
            if action == "forward":
                reward += 2


    if action == "left":
        left_count += 1
    else:
        left_count = 0

    if left_count == 4:
        reward -= 5
        left_count = 0  # reset after penalty

        # Track right streak
    if action == "right":
        right_count += 1
    else:
        right_count = 0

    if right_count == 4:
        reward -= 5
        right_count = 0  # reset after penalty

    if len(action_history) == 10:
        if not any(a in ("left", "right") for a in action_history):
            reward -= 5.0

    food_collected = rob.get_nr_food_collected()
    reward += food_collected*5

    #
    # # Encourage the agent to keep the food in the middle of the screen
    # if middle > left or middle > right:
    #     reward += 2
    #
    # # or Just maximise the amount of green in the picture from the camera
    # reward += (middle * 1.5) + left + right

    return reward, left_count, right_count


def compute_reward3(rob, irs, action, left, middle, right, right_count, left_count, action_history):
    reward = 0
    info = {}

    # --- Update action history ---
    action_history.append(action)
    recent_actions = list(action_history)

    # === 1. FOOD COLLECTION ===
    food_collected = rob.get_nr_food_collected()
    if food_collected:
        reward += 100
        info['food_collected'] = True
        return reward, left_count, right_count  # Early return on major event

    # === 2. FORWARD MOVEMENT REWARD ===
    if action == "forward":
        reward += 1
        info['forward_bonus'] = 1

    # === 3. VISION-BASED REWARDS ===
    # Reward centered green more than sides
    if middle > left and middle > right:
        reward += 6
        info['center_bonus'] = 6

    green_pixel_reward = (middle * 1.5) + left + right
    reward += green_pixel_reward
    info['green_pixel_bonus'] = green_pixel_reward

    # === 4. COLLISION DETECTION ===
    collision = any(sensor > COLLISION_THRESHOLD for sensor in irs)
    if collision:
        reward -= 10
        info['collision_penalty'] = -10
        if action == "forward":
            reward -= 5
            info['forward_crash_penalty'] = -5
        else:
            reward -= 2
            info['nonforward_crash_penalty'] = -2

    # === 5. REPETITION PENALTIES ===
    if action == "left":
        left_count += 1
    else:
        left_count = 0

    if left_count >= 4:
        reward -= 5
        info['left_streak_penalty'] = -5
        left_count = 0

    if action == "right":
        right_count += 1
    else:
        right_count = 0

    if right_count >= 4:
        reward -= 5
        info['right_streak_penalty'] = -5
        right_count = 0

    if len(recent_actions) == 10 and not any(a in ["left", "right"] for a in recent_actions):
        reward -= 5
        info['no_turn_penalty'] = -5

    # === 6. Final food collection bonus (not early return) ===
    reward += food_collected * 5
    info['food_bonus'] = food_collected * 5

    return reward, left_count, right_count

def compute_reward4(
    rob, irs, action, left, middle, right,
    right_count, left_count, action_history,
    recently_collided, prev_action
):
    """Compute reward including collision escape behavior."""
    reward = 0
    action_history.append(action)

    # Detect green (food) presence
    food = middle + left + right
    if food > 5:
        reward += 1
    else:
        collision = any(sensor > COLLISION_THRESHOLD for sensor in irs)
        if collision:
            reward -= 10.0
            recently_collided = True  # mark that we collided
            if action == "forward":
                reward -= 3.0  # discourage pushing into wall
        else:
            if action == "forward":
                reward += 2

            # Reward escape after a collision
            if recently_collided and prev_action == "backward" and action in ("left", "right"):
                reward += 10
                recently_collided = False

    # --- Left turn penalty ---
    if action == "left":
        left_count += 1
    else:
        left_count = 0
    if left_count == 4:
        reward -= 5
        left_count = 0

    # --- Right turn penalty ---
    if action == "right":
        right_count += 1
    else:
        right_count = 0
    if right_count == 4:
        reward -= 5
        right_count = 0

    # --- No turning penalty ---
    if len(action_history) == 10 and not any(a in ("left", "right") for a in action_history):
        reward -= 5.0

    # --- Reward collected food ---
    food_collected = rob.get_nr_food_collected()
    reward += food_collected * 5

    return reward, left_count, right_count, recently_collided

def save_model(model, filename=None):
    """Save the PyTorch model (always as dqn_model_{timestamp}.pth)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.environ.get("SAVE_DIR", "/root/results")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"dqn_model_{timestamp}.pth"  # always use this pattern
    model_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    return model_path

def load_model(model, filename):
    """Load a PyTorch model."""
    filepath = os.path.join("/root/results", "models", filename)
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model

def get_input_size(rob:IRobobo):
    """
    I just made this function to make it more mainstream. So if we change it we can add it here.
    """
    irs = rob.read_irs()
    ir_state : int= len([ir_reading for idx, ir_reading in enumerate(irs) if idx in [2,3,4,5,7]])
    num_obs = ir_state + 3 + 2 # 3 from the images and 2 from the action
    return 10

def run_all_actions(rob: IRobobo, load_model=None, num_episodes=EPISODES):
    """Run the DQN algorithm on the Robobo."""
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    # forward_streak = 0

    print("Training DQN Agent...")
    
    # Get input size after simulation is started
    input_size = get_input_size(rob) # update according to the new observations
    print(f"Input size: {input_size}, Action size: {NUM_ACTIONS}")

    policy_net = DQNNetwork(state_size=input_size, action_size=NUM_ACTIONS).to(device)
    target_net = DQNNetwork(state_size=input_size, action_size=NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    if load_model:
        policy_net = load_model(policy_net, load_model)
        target_net.load_state_dict(policy_net.state_dict())
    
    best_episode_reward = float('-inf')
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        
        rob.set_phone_tilt(100, 100)
        # In evry episode, turn the robot and random to change the initial state.
        action = np.random.choice(ACTIONS)
        rob.move_blocking(ACT_TO_MOTOR[action][0], ACT_TO_MOTOR[action][1], 100)
        
        frame = rob.read_image_front()
        green_only, green_mask = isolate_green(frame)
        left, middle, right = analyze_sections(green_mask)
        
        state = get_state(rob, action, [left, middle, right])
        episode_reward = 0
        steps_without_progress = 0
        last_best_reward = float('-inf')

        action_history = deque(maxlen=10)
        left_count = 0
        right_count = 0
        prev_action = None
        recently_collided = False
        
        for step in range(MAX_STEPS):
            start_position = rob.get_position()
            action = choose_action(state, policy_net)
            # print(action)
            perform_action(rob, action)
            

            frame = rob.read_image_front()
            green_only, green_mask = isolate_green(frame)
            left, middle, right = analyze_sections(green_mask)

            next_state = get_state(rob, action, [left, middle, right])

            irs = rob.read_irs()
            # reward, left_count, right_count = compute_reward2(rob, [ir_reading for idx, ir_reading in enumerate(irs) if idx in [2, 3, 4, 5, 7]],
            #                                                   action, left, middle, right, left_count, right_count, action_history)
            reward, left_count, right_count, recently_collided = compute_reward4(rob, [ir_reading for idx, ir_reading in enumerate(irs) if idx in [2, 3, 4, 5, 7]],
                                                              action, left, middle, right, left_count, right_count, action_history, recently_collided, prev_action)

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
            
            if step % 1 == 0:
                print(f"Step {step}: Action {action}, Reward {reward:.2f}, Total Reward {episode_reward:.2f}")
        
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