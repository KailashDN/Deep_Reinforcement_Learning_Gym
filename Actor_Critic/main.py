from itertools import count
import torch
import numpy as np
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym.wrappers import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import wrapper
from actor_critic.agent import TwoNetAgent, TwoHeadAgent

WORLD = 1
STAGE = 2
LEVEL_NAME = "SuperMarioBros-{1}-{1}-v1".format(WORLD, STAGE)
ACTION_SPACE = COMPLEX_MOVEMENT
FRAME_DIM = (84, 84, 4)
FRAME_SKIP = 4
NUM_EPISODES = 20_000
# LEARNING_RATE = 0.00003
ACTOR_LEARNING_RATE = 0.00005
CRITIC_LEARNING_RATE = 0.0005
GAMMA = 0.99
ENTROPY_SCALING = 0.01

RENDER_GAME = True
PLOT_INTERVAL = 50
VIDEO_INTERVAL = 1
CHECKPOINT_INTERVAL = 100
#MODEL_PATH = "./models/actor_critic_two_head_world1-1"
ACTOR_MODEL_PATH = "./models/actor_model_world1-2"
CRITIC_MODEL_PATH = "./models/critic_model_world1-2"
LOAD_MODEL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Available GPU: {torch.cuda.get_device_name(0)}")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = wrapper(tmp_env, FRAME_DIM, FRAME_SKIP)

    return tmp_env

def plot_reward_history(reward_history, mean_reward_history):
    plt.plot(reward_history, "b-", mean_reward_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.show()
