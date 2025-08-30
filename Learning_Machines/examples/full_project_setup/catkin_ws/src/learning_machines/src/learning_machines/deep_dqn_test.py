import os
import torch
import numpy as np
from .deep_dqn_obstacle_detection import DQNNetwork # Adjust if needed
from robobo_interface import IRobobo, SimulationRobobo

STATE_DIM = 10   # must match training: 5 IR + 2 motors + 3 image sections
ACTION_DIM = 4   # forward, left, right, backward
HIDDEN_SIZE = 128

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    model_path = os.environ.get("MODEL_PATH", "/root/results/dqn_model_20250618_214149.pth")

    # Build the correct model
    model = DQNNetwork(state_size=STATE_DIM, action_size=ACTION_DIM, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Read Robobo sensors and image
    irs = rob.read_irs()
    ir_subset = [irs[i] for i in [2, 3, 4, 5, 7]]
    frame = rob.read_image_front()

    from .deep_dqn_obstacle_detection import isolate_green, analyze_sections, ACT_TO_MOTOR
    _, mask = isolate_green(frame)
    left, middle, right = analyze_sections(mask)

    action = "forward"
    motor_vals = list(ACT_TO_MOTOR[action])
    state = ir_subset + motor_vals + [left, middle, right]

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = model(state_tensor)
    best_action_idx = torch.argmax(q_values).item()

    print("Q-values:", q_values)
    print("Best action index:", best_action_idx)