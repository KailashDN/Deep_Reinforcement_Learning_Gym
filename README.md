# Deep Reinforcement Learning:
### The purpose of the project is to implement and create documentation for Deep Reinforcement learning Algorithm like DDQN, policy gradient, actor-critic 

[![PackageVersion][pypi-version]][pypi-home]
[![PythonVersion][python-version]][python-home]
[![License][pypi-license]](LICENSE)

[pypi-version]: https://badge.fury.io/py/gym-super-mario-bros.svg
[pypi-home]: https://badge.fury.io/py/gym-super-mario-bros
[python-version]: https://img.shields.io/pypi/pyversions/gym-super-mario-bros.svg
[python-home]: https://python.org
[pypi-license]: https://img.shields.io/pypi/l/gym-super-mario-bros.svg
    
## Abstract
Reinforcement learning is the family of learning algorithms in which an agent learns from its environment by interacting with it. What does it learn? Informally, an agent learns to take actions that bring it from its current state to the best (optimal) reachable state.

![DQN](/Images/RL_arch.png)
[Image credit:!(https://mc.ai/introduction-to-double-deep-q-learning-ddqn/)]

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
1. [`DDQN (Deep Double Q learning):`](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/DoubleDQN)
2. [`Policy Gradient`:](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/Policy_Gradient)
3. [`Actor-Critic`:](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/tree/main/Actor_Critic)

### 1. `DeepQ_Network.py`: 
Convolutional Neural Network Model to learn and predict what action to take. Our model architecture is the same as in the deep mind atari paper "Playing Atari with Deep        Reinforcement Learning" It takes two inputs:
#### 
    1. :param frame_dim: The dimension of the given frames
    2. :param num_actions: The number of possible actions

### 2. `wrappers.py`:
Wrappers will allow us to add functionality to environments, such as modifying observations and rewards to be fed to our agent.
#### Functionality:
1. `MaxAndSkipEnv(env, skip=frame_skip)`: In super mario environments, we also apply frame-skipping internally before doing any state-processing. Effectively, we are concatenating 4 frames selected from the span of 16 raw frames.

2. `WarpFrame(env, width=frame_dim[0], height=frame_dim[1])`: Warp frames to 84x84 as done in the Nature paper and later work. If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which observation should be warped.

3. `LazyFrames(list(self.frames))`: This object ensures that common frames between the observations are only stored once. It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers. This object should only be converted to a numpy array before being passed to the model.

4. `FrameStack(env, frame_dim[2])`: Stack 'k' (frame_dim[2]) last frames. Returns lazy array, which is much more memory efficient.
    
### 3. `replay_from_memory.py`:
Stores (state, action, reward, next_state, done) transition in memory. The policy net train on batch sampled from replay memory and sample transitions at random.

### 4. `main_Mario.py`:
Run main_Mario.py to train super mario. 
Steps:
1. Create super mario environment(*gym_super_mario_bros.make("SuperMarioBros-v0")*)
2. apply *JoypadSpace(env, ACTION_SPACE)* environment wrapper to convert binary to discrete action space (ACTION_SPACE = RIGHT_ONLY) Apply wrapper class to modify frames
3. Create 2 policy network since we are implementing Double DQN
    - 3.1 `policy_net`
    - 3.2 `target_net`
4. Create replay memory of predefined capacity of 100_000 transitions.
5. Set Hyperparamets like *total_steps*, *reward history*, *exploration(**α**)* which will be decay so that Q learning explore more in initial stage and exploitation increase with decaying **α** 
6. Iterate the model for *NUM_EPISODES*(10_000):
    - 6.1 At each step we render the environment
    - 6.2 Get next **action** using random value. If random value is smaller than current **α** then random set of *RIGHT_ONLY* action returned else *policy_net* predict and update the weights(max *Q value* ) of the target model if necessary 
    - 6.3 Perform the action using environment step function(defined in wrapper class) which returns *next_state*, *reward*, *done*, *info* 
    - 6.4 If we *done* or if mario *life* is less than 2 the end the step loop here and start next episode. Else,
    - 6.5 Add the transition to the replay memory using *push* function
    - 6.6 Increment the current reward and total steps
    - 6.7 Trains the policy net on a batch from the replay memory
    - 6.5 Update the weights of the target model if necessary
    - 6.6 update the exploration rate **α**

## Results:
Super Mario game trained on GTX 1050Ti using TensorFlow 1.5.2(TF2.x Multi GPU parallel version coming soon!!)
![Training](/Images/Mario_Training.png)![Training](/Images/Mario_Training1.png)![Training](/Images/Mario_Training2.png)

## Citations
- https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
- https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
- https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
- https://www.statworx.com/de/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/
- https://mc.ai/introduction-to-double-deep-q-learning-ddqn/
- https://stats.stackexchange.com/questions/326788/when-to-choose-sarsa-vs-q-learning
- https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

## License
[MIT License](/LICENSE)


