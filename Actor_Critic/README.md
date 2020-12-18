# Actor Critic Learning:
### Implementation of Deep Reinforcement learning Algorithm: Actor Critic   
    
## Abstract
Actor-critic learning is a reinforcement-learning technique in which you simultaneously learn a policy function and a value function. The policy function tells you how to make decisions, and the value function helps improve the training process for the value function.

Actor-critic methods are Temporal difference(TD) methods that have a separate memory structure to explicitly represent the policy independent of the value function. The policy structure is known as the actor, because it is used to select actions, and the estimated value function is known as the critic, because it criticizes the actions made by the actor. Learning is always on-policy: the critic must learn about and critique whatever policy is currently being followed by the actor. The critique takes the form of a TD error. This scalar signal is the sole output of the critic and drives all learning in both actor and critic

![ActorCritic](/Images/Actor_Critic.png)
[Image credit](https://theaisummer.com/assets/img/posts/Actor_critics/ac.jpg)


## Implementation: 
### Here I have explained some important code and working

### 1. `Policy_Model.py`: 
Convolutional Neural Network Model to learn and predict what action to take. 
It takes two inputs:
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
    
### 3. `policy_agent.py`:
- Select action based on current state of agent
- Returns the sampled action and the log of the probability density.
- Here we calculate the probability distribution of the prediction of actions using policy model
- Once we have probability distribution of action predicted we sample action
- Return log probability of action
- We also update log probabilities of actions and reward
- As name suggested the save_model function save the model and Load_model load the pretrained saved model

### 4. `policy_main.py`:
#### Run policy_main.py to train super mario. 
Steps:
1. Environment Setting:
    - Set mario world, stage, Level Name("SuperMarioBros-{}-{}-v0".format(WORLD, STAGE))
    - ACTION_SPACE: Marion movement(SIMPLE_MOVEMENT, RIGHT_ONLY, or COMPLEX_MOVEMENT)
2. Set Hyperparameters:
    - LEARNING_RATE, NUM_EPOCHS, GAMMA etc.
3. Create Environment:
    - Create mario environment for specified level
    - Apply *JoypadSpace(env, ACTION_SPACE)* wrapper to convert binary to discrete action space (ACTION_SPACE = SIMPLE_MOVEMENT)
    - Apply wrapper class to modify frames
4. Create Mario agent
5. Train The Mario Model for *NUM_EPOCHS(1001)*:
    At each epoch check the cuda memory, reset the state and last reward
    For each step:
    5.1 Perform action:
        - Sample action and log probability of action from probability distributions of agents action predictions
    - 5.2 Delete the last state to prevent memory overflow
    - 5.3 Calculate the `state`, `reward`, and `metadata` for current action
    - 5.4 If agent died the reward will be less than zero, update the reward history
    - 5.5 If Mario solved the current level the highest possible reward of 15 is awarded( level finish)
    - 



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



