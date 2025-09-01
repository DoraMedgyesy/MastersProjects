#  Neural Dependency Parser 

This project implements a **feed-forward neural network** for **transition-based dependency parsing**, built using PyTorch. 
It forms part of an NLP assignment and supports training, validation, and testing on preprocessed dependency parsing datasets.

The parser learns to predict transitions (Shift, Left-Arc, Right-Arc) for a given partial parse configuration using dense word embeddings and two hidden layers.

---

## Project Structure

- `Dependency_parsing/DepParsing/results`: Output logs and model weights
- `Dependency_parsing/DepParsing/data`: Preprocessed dataset (inputs, labels, etc.)
- `Dependency_parsing/DepParsing/parser_model.py`: Neural network model definition
- `Dependency_parsing/DepParsing/parser_transitions.py`: Transition logic and minibatch parser
- `Dependency_parsing/DepParsing/run.py`: Main training/testing script

## Environment Setup

Follow the instructions described in `Dependency_parsing/DepParsing/README.txt`.

## How to Run

Make sure you are in the folder `Dependency_parsing/DepParsing/` (where `run.py` is located). 
Then run the following in your terminal:

```bash
python run.py
```

To run a faster debug version (small dataset + fewer epochs):
```bash
python run.py --debug
```

## Acknowledgements

This project builds on starter code and materials provided by the professor as part of the **Natural Language Processing** course at **Vrije Universiteit Amsterdam (VU)**, 2025.  
The assignment structure, some utility functions, and datasets were supplied by the instructor. All additional implementation, analysis, and documentation were done independently.