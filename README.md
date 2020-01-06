# Deep Reinforcement Learning - Deep Q Network - Unity Navigation Project

## Project Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course in the Artificial Intelligent program.  The problem uses [Unity's ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/) to provide an environment in which there are both yellow and blue bananas on a 2D surface in a 3D space.  The objective is to collect as many yellow bananas as possible while avoiding the blue bananas.  You interact with this environment in first person, and can move forward and backwards and turn clockwise or counter clockwise.  Through setting up a DRL agent, the objective for this code is to set up an agent that can train the computer to efficiently solve this problem through only knowledge of the state of the environment and the actions available.

## Project Objectives and Setup

We will use a Deep Q Network (DQN) to train the agent to solve the problem, starting from a randomly initialized state-action, we will attempt to maximize a reward function (Q), which is a function of both the state of the environment and and action provided for that state.

To achive this, we will require:
- A sequentional deep neural network that learns the optimal actions based on our rewards function and given our current state (local q network)
- A sequential deep nerual network that keeps its weights fixed for a determined amount of iterations, and is used to determine estimted rewards from a successive state given the current action (target q network)
- A replay buffer that stores prior experiences as our agent learns from the environment

Our program acts upon the environment by initially choosing actions for the given state more-or-less at random, and then determines the reward, and next state given the action chosen.  This "experience" is then stored in the replay buffer.  We then update the current state to the next state determined from the chosen action and repeat this process.

Learning from the environment happens by chosing a sample of experiences from the replay buffer after a determined number of steps through the environment.  From these sampled experiences, we calculate both the expected rewards given the sampled action/state from our experiences (using our local q network), as well as the estimated successive rewards from the experienced next states and possible and the best possible action for those states (using our target q network).  We determine the loss for the local q network as the difference in rewards between the expected and estimated rewards, and backpropogate and update based on this loss calculation.  After a determined number of learning updates, we then refresh our target q network weights to equal that of the current local network weights.

As we march through this process, we use a decaying greedy-epsilon parameter that is set to start by favoring randomly chosen actions during initial training, and slowly diminishing to favor the best possible action as we complete more training steps. 

We also include Double DQN as part of our learning process, which stabilizes our learning by calculating our estimated best action from the next state using the local q network, and using those actions to calculate estimated next state reward from our target q network see [ref: Double DQN Paper](https://arxiv.org/abs/1509.06461).

Hyperparameters used for this approach are provided in the hyperparameters.py file, and our neural network configuration is provided in the models.py module.

## Current results

Through applying the above learning agent, we are able to achieve a maximal score for a given run of just under 17 bananas.  The results of our scores through successive training episodes are as shown:

## Areas for improvement

- More exhaustive hyperparameter tuning
- Implementation of [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) which weights replay experiences by the loss parameter associated with them, and uses these weights to generate our sample batch while learning
- Implementation of [Dueling Q Networks](https://arxiv.org/abs/1511.06581) which learns separately on state values and action values and combines the results from learning.  This could potentially more robustly identify our reward function.



## Environment Set Up

General instructions are available on the [Udacity github page for this project](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

This implementation was validated on a x64 based Windows system using Anaconda3 to provide the hosted environment environment.

1. Setup the conda environment
```
conda create --name drlnd python=3.6 
activate drlnd
```
2. Install dependecies
```
# Install general dependencies for Udacity deep-reinforcement-learning projects
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

# Intall google OpenAI gym dependencies
pip install gym

# Install pytorch (Note this only works for Cuda 10.1 on x64 windows)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install the old version of Unity ml-agents
pip install pip install unityagents

# Install the ipython kernel package
conda install ipykernel
# Install the ipython kernel to the conda environment
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

To run the notebook, within the drlnd conda environment run `jupyter lab` and navigate to the the appropriate ipython notebook.


## Using this program

Learning and a resultant view of both a random state, and learned state are achievable by following the progression in the Navigation-checkpoint.ipynb python notebook.
