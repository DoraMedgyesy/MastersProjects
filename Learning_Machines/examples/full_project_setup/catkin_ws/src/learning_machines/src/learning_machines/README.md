# Deep Q-Learning for Robobo Robot Control

This folder contains custom implementations of Deep Q-Networks (DQN) to control the Robobo robot in two tasks:
1. **Obstacle Avoidance** using infrared (IR) sensor input.
2. **Green Object Tracking** using image-based segmentation and target-following behavior.

These scripts were developed as extensions to the Learning Machines Robobo framework.

---

##  Contents

- `deep_dqn_obstacle_avoidance.py`: Obstacle avoidance agent using IR sensor input only. 
- `deep_dqn_obstacle_detection.py`: Vision-guided agent trained to track and approach green-colored targets.
- `deep_dqn_test.py`: Script to load and test a saved DQN model with Robobo.

---

##  Obstacle Avoidance Agent

**Location**: `deep_dqn_obstacle_avoidance.py`

**Description**:  
Trains a DQN agent to explore the environment and avoid obstacles based solely on IR sensor input.  
The reward function penalizes collisions and repetitive turning, and rewards smooth forward progress.

**States
**: 8 normalized IR values  
**Actions**: `["move_forward", "big_left", "small_left", "big_right", "small_right", "backward"]`

During training, the script automatically saves the trained neural network as a `.pth` file in the `/examples/full_project_setup/results/` directory.  
Saved models follow the format:  
```bash
dqn_model_YYYYMMDD_HHMMSS.pth
```
---

##  Obstacle Detection Agent

**Location**: `deep_dqn_obstacle_detection.py`

**Description**:  
Trains a  more advanced DQN agent that learns to follow and collect green targets using image-based input. 
When the agent detects an object it moves towards it and touches it, resembling food being collected.
The state vector combines IR sensors, last motor command, and green pixel distribution across the camera frame.  
The reward function prioritizes:
- Food collection (via `get_nr_food_collected()`)
- Centered vision of the target
- Collision avoidance
- Penalizing repetitive actions

**States**: `[5 IR values] + [2 motor values] + [3 green pixel distribution] = 10 dimensions`  
**Actions**: `["forward", "left", "right", "backward"]`

During training, the script automatically saves the trained neural network as a `.pth` file in the `/root/results/` directory.  
Saved models follow the format:  
```bash
dqn_model_YYYYMMDD_HHMMSS.pth
```
---

##  Neural Network Architecture

- Multi-layer fully connected networks (PyTorch)
- Hidden layers with ReLU activation
- Output: Q-value for each discrete action
- Includes experience replay and a target network

---
##  Selecting which model will be used in testing

The script `deep_dqn_test.py` loads a trained model based on a path.  
Inside the script, this is specified by:

```python
model_path = os.environ.get("MODEL_PATH", "/root/results/dqn_model_20250618_131556.pth")
```
To change the trained model used for testing, in the script edit the model_path variable to point to your desired .pth file.

##  How to Run

### 1. Expose the correct agent in `__init__.py`
Open init.py in this folder.
Uncomment one of the following blocks:
```python
# To train Green Target Tracking:
from .deep_dqn_W2_2 import run_all_actions
__all__ = ("run_all_actions",)

# To train Obstacle Avoidance:
from .deep_dqn_W1 import run_all_actions
__all__ = ("run_all_actions",)

# To test a trained model:
from .deep_dqn_test import run_all_actions
__all__ = ("run_all_actions",)
```
### 2. Run code as described in external readME
To run the code follow the steps described in [`examples/full_project_setup/README.md`](../examples/full_project_setup/README.md).

##  Project Report

A written report describing the full project (including architecture, training approach, and results) is included at the root of this repository:

- [`Robobo_DQN_Project_Report.pdf`](./Robobo_DQN_Project_Report.pdf)