# Deep Reinforcement Learning:

[![PythonVersion](https://img.shields.io/badge/Python-3.6.9-green.svg)](https://www.python.org/downloads/release/python-369/)
[![PythonVersion](https://img.shields.io/badge/gym_super_mario_bros-7.3.2-blue.svg)](https://pypi.org/project/gym-super-mario-bros/7.3.2/)
[![PythonVersion](https://img.shields.io/badge/TensorFlow-1.15(Cuda:10.2)-red.svg)](https://www.tensorflow.org/install/gpu)
[![PythonVersion](https://img.shields.io/badge/PyTorch-1.5.1(Cuda:10.1)-red.svg)](https://pytorch.org/get-started/previous-versions/)
[![License][pypi-license]](LICENSE)

[pypi-license]: https://img.shields.io/pypi/l/gym-super-mario-bros.svg

## Project Goal
The purpose of the project is to implement and create documentation for Deep Reinforcement learning Algorithm like DDQN, policy gradient, actor-critic 
_This Work is done for Course CSYE 7370 Deep Reinforcement Learning uder guidance of Prof. Nicholas Brown_

## Abstract
Reinforcement learning is the family of learning algorithms in which an agent learns from its environment by interacting with it. What does it learn? Informally, an agent learns to take actions that bring it from its current state to the best (optimal) reachable state.

![Reinforcement Learning](/Images/RL_arch.png)
[Image credit](https://miro.medium.com/max/1250/0*WC4l7u90TsKs_eXj.png)

Reinforcement Learning has four essential elements:
1. `Agent`: The program you train, with the aim of doing a job you specify.
2. `Environment`: The world, real or virtual, in which the agent performs actions.
3. `Action`: A move made by the agent, which causes a status change in the environment.
4. `Rewards`: The evaluation of an action, which can be positive or negative.

## Methods Used: OpenAI Gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents in everything from walking to playing games like Pong or Pinball.
OpenAI Gym gives us game environments in which our programs can take actions. Each environment has an initial status. After your agent takes an action, the status is updated.
When your agent observes the change, it uses the new status together with its policy to decide what move to make next. The policy is key: it is the essential element for your program to keep working on. The better the policy your agent learns, the better the performance you get out of it.

Here I am implementing following algorithm using Super Mario Bros Gym Environment:
### 1. [`DDQN (Deep Double Q learning)`](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/DoubleDQN)
### 2. [`Policy Gradient`](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/Policy_Gradient)
### 3. [`Actor-Critic`](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/Actor_Critic)

## Needs of this project:
#### The project need good understanding of:
- Deep Learning, 
- Probability Distributions, 
- Basic Reinforcement Learning,
- Python 3
- Pycharm
- Jupyter Notebook and 
- Knack for Data Science

## Getting Started
1. Clone this [repo](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym.git) (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Since I am doing this project solely to learn I have implemented it with both `Tensorflow 1.5.2` and `Pytorch`:
3. Since there are two different framework I have created two python environment and created `.yml` file, clone of my both environment for easier project setup.
4. The Environment link:
    - [DDQN: Tensorflow 1.15](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/blob/main/DoubleDQN/RL_GPU_TF15.yml)
    - [Policy Gradient/Actor-Critic: PyTorch](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/blob/main/Actor_Critic/RL_GPU_TORCH.yml)
5. Command to copy Conda envs: [How to create conda environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)
    - `conda create -n new_env -f=\path\to\RL_GPU_TORCH.yml` 
        <br />OR
    - `conda create --name new_env --file \path\to\RL_GPU_TORCH.yml` 
## How to Load and Run:
### 1. PyCharm [Install Guid](https://www.jetbrains.com/help/pycharm/installation-guide.html):
1. Clone and open project in pycharm
2. Check Available [GPU](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/blob/main/check_available_gpu.py)
3. Each Reinforcement Learning method has separate folder where you can find and run the main file.
### 2. Jupyter:
1. Each Reinforcement Learning method has Jupyter Notebook for better understanding of project pipeline

## Results:
Super Mario game trained on GTX 1050Ti using TensorFlow 1.5.2(TF2.x Multi GPU parallel version coming soon!!) and Pytorch.
CUDA Version: 10.2
<!-- ![Training](/Images/Mario_Training.png)![Training](/Images/Mario_Training1.png)![Training](/Images/Mario_Training2.png)-->
|Algorithm Name     |  Training Video  |  Training Reward History |
|---------|-----------------|-----------------------------------|
| DDQN      | ![](/Images/DDQN_train.gif)| ![](/Images/DDQN_reward.png) |
| Actor-Critic | ![](/Images/Actor_critic_train.gif) | ![](/Images/Actor_Critic_reward.png) |
| Policy Gradient | ![](/Images/policy_gradient_train.gif) | ![](/Images/policy_gradient_reward.png) |

## Future Work:
Implementation of Asynchronous Advantage Actor-Critic (A3C), synchronous, deterministic A3C (A2C), Deterministic policy gradient (DPG), Twin Delayed Deep Deterministic(TD3)

## Citations
- https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
- https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
- https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
- https://github.com/D0miH/deep-learning-super-mario#actor-critic-approach
- https://www.statworx.com/de/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/
- https://mc.ai/introduction-to-double-deep-q-learning-ddqn/
- https://stats.stackexchange.com/questions/326788/when-to-choose-sarsa-vs-q-learning
- https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

## License
[MIT License](/LICENSE)


