import sys
from itertools import count

import cv2
import gym_super_mario_bros
from gym.wrappers import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import matplotlib.pyplot as plt
import numpy as np
import torch

from Policy_Gradient.policy_agent import Agent
from wrappers import wrapper

# env settings
WORLD = 1
STAGE = 1
LEVEL_NAME = "SuperMarioBros-{}-{}-v0".format(WORLD, STAGE)
FRAME_DIM = (84, 84, 4)     # (120, 128, 4)  # original image size is 240x256
ACTION_SPACE = SIMPLE_MOVEMENT
RENDER_GAME = True

MODEL_PATH = ""  # to create a new model set it to ""

# training hyperparameters
TRAIN_MODEL = True
LEARNING_RATE = 0.000007
NUM_EPOCHS = 1_000
GAMMA = 0.99

LOG_INTERVAL = 1
PLOT_INTERVAL = 10
VIDEO_INTERVAL = 50
