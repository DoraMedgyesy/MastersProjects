# Evolutionary Controller for Evoman

This project implements a custom **Evolutionary Algorithm (EA)** for training neural network controllers in the [Evoman Framework](https://github.com/karinemiras/evoman_framework.git).
The goal is to evolve neural network agents that can successfully defeat a range of enemies using different evolutionary strategies.
## Project Structure

## Contents

- `Code/algorithm.py`: Implements the Evolutionary Algorithm (EA) for training and testing specialist agents.
- `Code/main.py`: Entry point to run experiments for different enemies and selection strategies.
- `Code/train/`: Auto-generated directory containing results, logs, and saved models per enemy and run.

---

## Algorithm

The `Code/algorithm.py` file performs the following key operations:
- Initializes the Evoman environment and neural controller with 10 hidden neurons
- Defines and runs the genetic algorithm for a fixed number of generations (default: 20)
- Implements three selection strategies (Roulette Wheel, Elitism, Ranked)
- Applies mutation and crossover to generate offspring
- Logs results per generation to `results.txt` and saves best solutions to `best.txt`
- Supports checkpointing and resuming via the Evoman solution state
- Uses a "doomsday" mechanism to inject diversity if no progress is made for several generations

When called in training mode (`run_mode='train'`), it creates a full experiment directory and evolves the population. 
When called in testing mode (`run_mode='test'`), it loads the best saved solution and plays it against the selected enemy to compute a final performance gain.

The evolutionary loop is designed to be used via `main.py`, where multiple runs across enemies can be managed more easily.

##  How to Run

###  Prerequisites

- Python 3.8+
- Dependencies listed by the Evoman framework

To train or test the model, uncomment one of the following at the end of Code/main.py file:

```python
#Uncomment to train with elitist selection
train('Elitism')

#Uncomment to train with rank based selection
train('Ranked')

#Uncomment to test with elitist selection
test('Elitism')

#Uncomment to test with rank based selection
test('Ranked')
```
