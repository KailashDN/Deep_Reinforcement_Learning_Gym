from math import inf

import gym


class RewardWrapper(gym.Wrapper):
    """
    Takes the mario gym environment and applies a custom reward function.
    """

    def __init__(self, env):
        self.env = env.unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.env.reward_range = (-inf, inf)

        # start value of mario is 40
        self.last_x_pos = 40
        self.max_x_pos = 40

        # start time is 400
        self.last_time_value = 400

        self.last_score = 0

        self.last_status = "small"


