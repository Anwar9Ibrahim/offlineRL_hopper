
# Offline Reinforcement Learning on Hopper Environment with IQL

This project implements and evaluates an offline reinforcement learning (RL) algorithm using the Hopper environment from the MuJoCo suite [Gym MuJoCo documentation](https://www.gymlibrary.dev/environments/mujoco/hopper/). The environment is part of the [D4RL library](https://github.com/Farama-Foundation/D4RL), which provides a collection of benchmark datasets designed for offline RL.

## Table of Contents

- [Project Overview](#project-overview)
- [Usage](#usage)
- [Algorithms Overview](#algorithms-overview)
- [Experiment Tracking](#experiment-tracking)
- [Results](#results)

## Project Overview

This project focuses on implementing the [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) algorithm in PyTorch and testing it on the Hopper environment. The primary goal is to evaluate IQL's performance using the `hopper-expert-v2` dataset from D4RL.

![hopper](https://github.com/user-attachments/assets/03a47f98-74f3-4dfa-a4a1-b7bd5df73fa4)

### Why IQL?

IQL was chosen for this project because it tends to be less conservative than other offline RL methods, allowing it to leverage high-quality expert data more effectively. This makes it particularly suitable for the `hopper-expert-v2` dataset, where the data quality is high, and actions are close to optimal.

## Usage

The code for this project was written on the Kaggle platform, and everything can be run directly from there. You can access the Kaggle notebook and run the experiments without any additional setup.

To execute the code:

1. **Open the Kaggle Notebook**:
   - Navigate to the Kaggle platform and open the notebook associated with this project.

2. **Run the Cells**:
   - Simply run the cells in the notebook to execute the training process for the IQL algorithm on the Hopper environment.

3. **Monitor the training progress**:
   - Training logs and metrics will be automatically synced to [Weights & Biases](https://wandb.ai/anwar96ibrahim-student/offline-RL?nw=nwuseranwar96ibrahim).

## Algorithms Overview

This project compares the performance of different offline RL algorithms, including:

1. **EDAC (Ensemble Diversification in Offline Reinforcement Learning)**: Utilizes ensemble methods to handle out-of-distribution actions.
2. **AWAC (Advantage Weighted Actor-Critic)**: Adapts the actor-critic approach for offline settings.
3. **IQL (Implicit Q-Learning)**: A simpler, less conservative approach that excels with high-quality datasets.

### Decision Guidelines

- **IQL**: Best for high-quality expert datasets like `hopper-expert-v2`.
- **EDAC**: Suitable for environments with a high likelihood of out-of-distribution actions.
- **AWAC**: Ideal if using an actor-critic approach, especially with mixed-quality datasets.

## Experiment Tracking

All experiments are tracked using Weights & Biases. You can access the project dashboard [here](https://wandb.ai/anwar96ibrahim-student/offline-RL?nw=nwuseranwar96ibrahim).

## Results

The performance of IQL on the Hopper environment has shown strong results, leveraging the high-quality expert data to achieve optimal control policies.

---

This README provides a clear and concise overview of your project, tailored for execution on the Kaggle platform.
