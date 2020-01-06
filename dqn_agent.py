import random
from collections import namedtuple, deque

import numpy as np

import torch
from torch.nn import functional as F
from torch import optim

from model import DQN
import hyperparameters as hp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        :param state_size: (int) dimension of state space
        :param action_size: (int) dimension of action space
        :param seed: (int) random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-network initialization
        self.dqn_local = DQN(state_size, action_size, seed).to(device)
        self.dqn_target = DQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.dqn_local.parameters(),
                                    lr=hp.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, hp.buff_size, hp.batch_size)

        # Initialize time step dictating when to pull experiences from memory
        self.time_step = 0

        # Initialize the target update counter to determine when to refresh target weights from local
        self.tu_step = 0

    def step(self, state, action, reward, next_state, done):
        """Increments

        :param state: (array_like) the current environment state
        :param action: (array_like) the determined action by our agent
        :param reward: the reward given the current state and chosen action
        :param next_state: the determined next state given our current state and action
        :param done: (bool) determines if we have reached a terminal state

        """
        # save experience to memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn from memory after number of update iterations
        self.time_step = (self.time_step+1) % hp.iter_memory_pull
        if self.time_step == 0:
            # Pull from memory when only if num experiences exceeds our batch size
            if len(self.memory) > hp.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, hp.Gamma)
                self.tu_step = (self.tu_step+1) % hp.iter_target_update

        # Do a soft update of our target network if we have achieved desired number of learning steps
        if self.tu_step == 0:
            self.soft_update(self.dqn_local, self.dqn_target, hp.tau)

    def act(self, state, epsilon=0.0):
        """Returns an action given the provided state

        :param state: (array_like) current state
        :param epsilon: (float) epsilon-greedy hyperparmeter

        :return: an action value
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        self.dqn_local.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, exps, gamma):
        """ Update Q values using experiences

        :param exps: (array_like) list of experience tuples
        :param gamma: (float) hyperparameter discount factor
        """
        states, actions, rewards, next_states, dones = exps

        # Double DQN
        # Get maximum estimated action value from local dqn + next state
        max_est_action = self.dqn_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Evaluate target Q with maximum estimated actions
        q_targets_next = self.dqn_target(next_states).gather(1, max_est_action)
        # Calculate value estimates for updated Q
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.dqn_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Updates DQN model parameters

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """A fixed size memory buffer for storing and retrieving learned experiences"""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object

        :param action_size: (int) dimension of action space
        :param buffer_size: (int) the size of our ReplayBuffer
        :param batch_size: (int) training size to pull from buffer
        :param seed: (int) random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience',
                                     field_names=['state',
                                                  'action',
                                                  'reward',
                                                  'next_state',
                                                  'done'])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory

        :param state: (array_like) the current environment state
        :param action: (array_like) the determined action by our agent
        :param reward:  the reward given the current state and chosen action
        :param next_state:  the determined next state given our current state and action
        :param done: (bool) determines if we have reached a terminal state
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """Randomly selects a sample of experiences from memory

        :return: (tuple) a previously learned experience
        """
        exps = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory.

        :return: (int) size of memory buffer
        """
        return len(self.memory)
