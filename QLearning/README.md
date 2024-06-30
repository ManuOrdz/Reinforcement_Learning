# Q-learning Algorithm

This repository contains an implementation of the Q-learning algorithm, a reinforcement learning method used to find an optimal policy in decision-making problems.

## Table of Contents

- [Introduction](#introduction)
- [Projet Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm](#algorithm)

## Introduction

Q-Learning is a reinforcement learning algorithm that aims to learn an action-value function ***Q(s, a)*** which estimates the expected reward of taking action ***a*** in state ***s***. The goal is to find a policy that maximizes the total expected long-term reward.

## Project Structure

- `main.py`: Main implementation of the Q-Learnig algorithm.
- `Agent.py`: Definition of the agent tha uses Q-Learning to learn the optimal policy.
- `Parameters.py`: Defention of the parameters that uses Q-Learning algorithm

- `README.md`: Project documentation.

- `requeriments.txt`: Dependencies required to run the code.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requeriments.txt
```

## Usage

To run the Q-Learning algorithm, use the following command:

```bash
python main.py
```

## Algorithm
### Initialization


- Initialize the Q(s,a) arbitrarily for all states $s$ and actions $a$.

    $\\
    Q(s,a) \leftarrow 
    \begin{bmatrix}
    0 & .. & 0 \\
    : &  & : \\
    0 & .. & n
    \end{bmatrix}
    $

- Set the hyperparameters: learning rate \($\alpha$\), discount factor \($\gamma$\), and exploration parameter \($\epsilon$\). *example:*\
    $\\
    \alpha = 0.01 \\
    \gamma = 0.95 \\
    \epsilon = 0.1 \\
    $

### Training Process

1. Choose Action: Select an action $a$ and using an action selection policy \($e.g.,\epsilon\ -greedy$\)

2. Take Action: Execute acion $a$ and observe the reward $R(s,a)$ and next state $s'$.

3. Update Q-Value:
$\\
Q(s,a) \leftarrow\ Q(s,a) +\alpha\ [R(s,a) + \gamma *max_{a'}(Q(s',a')-Q(s,a))]
$

4. Update State: Set the new state $s$ to the observed next state $s'$

5. Repeat: Continue until the termination condition is met (e.g., a specified number of iteration or convergence).

### Action Selection Policy

- $\epsilon -greedy:$ With probability \($\epsilon$\), choose a random action \( *exploration*\), and with probability \( $1-\epsilon$\), choose the action that maximizes $Q(s,a)$ \( exploitation\).

## Contributing


## License