# Actor Critic Learning:
### Implementation of Deep Reinforcement learning Algorithm: Actor Critic   
    
## Abstract
Actor-critic learning is a reinforcement-learning technique in which you simultaneously learn a policy function and a value function. The policy function tells you how to make decisions, and the value function helps improve the training process for the value function.

Actor-critic methods are Temporal difference(TD) methods that have a separate memory structure to explicitly represent the policy independent of the value function. The policy structure is known as the actor, because it is used to select actions, and the estimated value function is known as the critic, because it criticizes the actions made by the actor. Learning is always on-policy: the critic must learn about and critique whatever policy is currently being followed by the actor. The critique takes the form of a TD error. This scalar signal is the sole output of the critic and drives all learning in both actor and critic

![ActorCritic](/Images/Actor_Critic.png)
[Image credit](https://theaisummer.com/assets/img/posts/Actor_critics/ac.jpg)

## How to run:
    python3 main.py

## Implementation: 
### Here I have explained some important code and working

### 1. `model.py`: 
We have Three Convolutional Neural Network Model to learn and predict what action to take.
This 3 Neural network are implemtation of two type:
    1. ActorCriticNet: Two Head Agent
    2. ActorNet: Two Net Agent
    3. CriticNet: Two Net Agent
Each Network class takes two inputs:
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

### `custom_reward_wrpper.py`: 
    **Takes the mario gym environment and applies a custom reward function**
    **Give Position Rewars**: Rewards mario for going right and punishes him for going left
    **Time Penalty**: Punishes mario by giving negative reward for doing nothing based on predefined time 
    **Death Penalty**: Punishes mario by giving negative reward for dying or not reaching goal within allocated time
    **score reward**: Rewards mario for increasing the ingame score.
    **status_reward**: 
        - Rewards mario for collecting a mushroom and getting tall or collecting a fire flower.
        - Mario gets punished for loosing the fire flower or getting small again.


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
5. Train The Mario Model for *NUM_EPOCHS(1001)*:<br />
    - At each epoch check the cuda memory, reset the state and last reward<br />
    - For each step (Iterate actions untile level cleared or marion died):
        - 5.1 Perform action:
            - Sample action and log probability of action from probability distributions of agents action predictions
        - 5.2 Delete the last state to prevent memory overflow
        - 5.3 Calculate the `state`, `reward`, and `metadata` for current action
        - 5.4 If agent died the reward will be less than zero, update the reward history
        - 5.5 If Mario solved the current level the highest possible reward of 15 is awarded( level finish)
        - 5.6 Convert the frame to tensor(height x width x depth) for pytorch
        - 5.7 Update Reward History
    - 5.8 For Each Episode calculate the loss
    - 5.9 Iterate next episode

## Results:
Super Mario game trained on GTX 1050Ti using Pytorch GPU
#### Training reward history:
![](/Images/Actor_Critic_reward.png)

#### Training Example([More](/Actor_Critic/videos)):
![](/Images//Actor_critic_train.gif)



