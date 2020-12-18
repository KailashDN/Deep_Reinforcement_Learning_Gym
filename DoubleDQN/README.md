# Deep Double Q-Learning:
### Implementation of Deep Reinforcement learning Algorithm: DDQN
    
## Abstract
In reinforcement learning, we often use a learning concept called Q-learning. Q-learning is based on so-called Q-values, that help the agent determining the optimal action, given the current state of the environment. Q-values are „discounted“ future rewards, that our agent collects during training by taking actions and moving through the different states of the environment. Q-values themselves are tried to be approximated during training, either by a simple exploration of the environment or by using a function approximator, such as a deep neural network (as in our case here). Mostly, we select in each state the action that has the highest Q-value, i.e. the highest discounted future reward, given the current state of the environment.

![DQN](/Images/DQN.png)
[Image credit](https://mc.ai/introduction-to-double-deep-q-learning-ddqn/)
<br />
#### Deep Q-Network:
![DQN](/Images/DeepDQN.png)
<br />
[Image credit](https://www.novatec-gmbh.de/wp-content/uploads/reinforcement_learning_loop-650x294.png)

In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. 
#### Steps:
    1. All the past experience is stored by the user in memory
    2. The next action is determined by the maximum output of the Q-network
    3. The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*. 
        This is basically a regression problem. However, we do not know the target or actual value here 
        as we are dealing with a reinforcement learning problem. 
    4. Going back to the Q-value update equation derived fromthe Bellman equation. 

## How to run:
python3 main_Mario.py

## Implementation: 
### Here I have explained some important code and working
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

![](/Images/DDQN_reward.png)

![](/Images/DDQN_train.gif)



