
# Offline Reinforcement Learning on Hopper Environment with IQL

This project implements and evaluates an offline reinforcement learning (RL) algorithm using the Hopper environment from the MuJoCo suite [Gym MuJoCo documentation](https://www.gymlibrary.dev/environments/mujoco/hopper/). The environment is part of the [D4RL library](https://github.com/Farama-Foundation/D4RL), which provides a collection of benchmark datasets designed for offline RL.

## Table of Contents

- [Project Overview](#project-overview)
- [Usage](#usage)
- [Algorithms Overview](#algorithms-overview)
- [Experiment Tracking](#experiment-tracking)
- [My contributions](#my-contributions)
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

### Explanation of the Solution

The solution is structured around preparing the MuJoCo Hopper environment and creating an agent using the Implicit Q-Learning (IQL) algorithm. The core components include:

1. **Environment Setup**: The MuJoCo Hopper environment is initialized to simulate a robotic hopper's movement. The environment's state, action space, and other dynamics are crucial for training the RL model.

2. **Custom Dataset Class (`CustomDataset`)**: 
   - This class is designed to handle the dataset in a way that is compatible with PyTorch's data loading utilities.
   - **Initialization (`__init__`)**: The constructor takes observations, actions, next observations, rewards, and done flags, converting them into PyTorch tensors for efficient processing during model training.
   - **Length (`__len__`)**: This method returns the total number of samples in the dataset, making it easier to iterate over the entire dataset during training.
   - **Get Item (`__getitem__`)**: This method allows access to individual samples in the dataset, retrieving the observation, action, next observation, reward, and done flag for a given index. This is essential for batching data during model training.

3. **Agent Architecture**: The agent is built using the IQL framework, which is specifically designed for offline RL tasks. The agent includes:
   - **Actor**: The actor network is responsible for generating actions, i.e., the policy that determines what actions to take given the current state.
   - **Two Critics**: The critics evaluate the actions taken by the policy. They estimate the Q-values, which reflect the expected cumulative reward of taking a certain action from a given state.
   - **Two Target Critics**: These are used to stabilize training by providing a stable target for the critic networks. They are essentially delayed versions of the critics, updated less frequently to reduce variance.
   - **Value Network**: This network estimates the value of being in a particular state, helping in the decision-making process by providing a baseline against which the benefits of different actions can be compared.

4. **Experiment Tracking and Saving Results**: 
   - The entire training process, including model checkpoints, results, and metrics, is logged and saved using Weights & Biases (W&B). This allows for detailed tracking of the agent's performance over time and easy visualization of the results.

## Experiment Tracking

All experiments are tracked using Weights & Biases. You can access the project dashboard [here](https://wandb.ai/anwar96ibrahim-student/offline-RL?nw=nwuseranwar96ibrahim).

## My contributions

I used in this [code](https://www.kaggle.com/code/mmdalix/openai-gym-mujoco-env-setup-and-training-2022) to set up the library.
for the algorithm I read the [paper](https://arxiv.org/pdf/2110.06169), and checked multiple online resources, but in general it was all updated to suite this problem.

## Results

to consider the environment solved it should achive in the testing phase a reward>= 3000.
The performance of IQL on the Hopper environment has shown strong results in the testing block "the last block in the code" you can see that it solved the "hopper-expert-v2" environment, ours is "3608.019141577519"  leveraging the high-quality expert data to achieve optimal control policies.

an Episode of our trained agent, is in the "hopper_episode.gif" file.





