import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build Deep Q Network Model

        :param state_size: (int) Dimension of state space
        :param action_size: (int) Dimension of action space
        :param seed: (int) Random seed
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc1_drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_drop = nn.Dropout(p=0.2)
        #self.fc3 = nn.Linear(32,64)
        #self.fc3_drop = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network to map state to action values

        :param state: (array_like) the current state
        :return: (array_like) action outcomes from model
        """
        x = F.relu(self.fc1_drop(self.fc1(state)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        #x = F.relu(self.fc3_drop(self.fc3(x)))
        return self.fc4(x)
