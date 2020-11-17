# Deep Reinforcement Learning:
### Implementation of Deep Reinforcement learning Algorithm like DDQN, policy gradient, actor critic
Reinforcement learning is the family of learning algorithms in which an agent learns from its environment by interacting with it. What does it learn? Informally, an agent learns to take actions that bring it from its current state to the best (optimal) reachable state.

In reinforcement learning we often use a learning concept called Q-learning. Q-learning is based on so called Q-values, that help the agent determining the optimal action, given the current state of the environment. Q-values are „discounted“ future rewards, that our agent collects during training by taking actions and moving through the different states of the environment. Q-values themselves are tried to be approximated during training, either by simple exploration of the environment or by using a function approximator, such as a deep neural network (as in our case here). Mostly, we select in each state the action that has the highest Q-value, i.e. the highest discounuted future reward, givent the current state of the environment.

![GitHub Logo](/Images/DQN.png)
[Image credit:!(https://mc.ai/introduction-to-double-deep-q-learning-ddqn/)]

In Double Deep Q Learning, the agent uses two neural networks to learn and predict what action to take at every step. One network, referred to as the Q network or the online network, is used to predict what to do when the agent encounters a new state. It takes in the state as input and outputs Q values for the possible actions that could be taken. 
