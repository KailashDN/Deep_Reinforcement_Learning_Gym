import torch
from torch.distributions import Categorical

from torch import optim
from torch.functional import F

from Actor_Critic.model import ActorNet, CriticNet, ActorCriticNet


class TwoHeadAgent:

    def __init__(self, frame_dim, action_space_size, lr, gamma, entropy_scaling, device):
        self.action_space_size = action_space_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.entropy_scaling = entropy_scaling

        self.model = ActorCriticNet(action_space_size, frame_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
