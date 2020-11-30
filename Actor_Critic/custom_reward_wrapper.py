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

    def position_reward(self, x_pos):
        """
        Rewards mario for going right and punishes him for going left.
        :return:    The reward value
        """
        reward = x_pos - self.last_x_pos

        if x_pos < self.max_x_pos:
            return 0

        self.max_x_pos = x_pos

        # reduce negative reward and clip the reward to max 5
        if reward < 0:
            reward = 0
        elif reward > 1:
            reward = 1

        self.last_x_pos = x_pos

        return reward

    def time_penalty(self, time):
        """
        Punishes mario for doing nothing.
        :param time: The currently remaining time
        :return:    The negative reward value
        """
        reward = time - self.last_time_value

        self.last_time_value = time

        return reward


