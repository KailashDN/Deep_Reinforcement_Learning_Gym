# Actor Critic Learning:
### Implementation of Deep Reinforcement learning Algorithm: Actor Critic   
    
## Abstract
Actor-critic learning is a reinforcement-learning technique in which you simultaneously learn a policy function and a value function. The policy function tells you how to make decisions, and the value function helps improve the training process for the value function.

Actor-critic methods are Temporal difference(TD) methods that have a separate memory structure to explicitly represent the policy independent of the value function. The policy structure is known as the actor, because it is used to select actions, and the estimated value function is known as the critic, because it criticizes the actions made by the actor. Learning is always on-policy: the critic must learn about and critique whatever policy is currently being followed by the actor. The critique takes the form of a TD error. This scalar signal is the sole output of the critic and drives all learning in both actor and critic

![ActorCritic](/Images/Actor_Critic.png)
[Image credit](https://theaisummer.com/assets/img/posts/Actor_critics/ac.jpg)


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

![](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/blob/main/Images/DDQN_reward.png)

![](https://github.com/KailashDN/Deep_Reinforcement_Learning_Gym/blob/main/Images/DDQN_train.gif)



