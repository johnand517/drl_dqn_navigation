import torch
from torch import nn
from torch.nn import functional as F

n1 = 256  # Number of nodes in first layer of neural network
n2 = 128  # Number of nodes in 2nd layer of neural network


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
        # First layer with dropout
        self.fc1 = nn.Linear(state_size, n1)
        self.fc1_drop = nn.Dropout(p=0.2)
        # Second layer with dropout
        self.fc2 = nn.Linear(n1, n2)
        self.fc2_drop = nn.Dropout(p=0.2)
        # Output layer
        self.fc3 = nn.Linear(n2, action_size)

    def forward(self, state):
        """Build a network to map state to action values

        :param state: (array_like) the current state
        :return: (array_like) action outcomes from model
        """
        # Use Relu activation functions for hidden layers
        x = F.relu(self.fc1_drop(self.fc1(state)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        return self.fc3(x)
