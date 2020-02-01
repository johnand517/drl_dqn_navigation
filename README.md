# Deep Reinforcement Learning - Deep Q Network - Unity Navigation Project

## Project Environment Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course in the Artificial Intelligent program by setting up a DRL agent to train the computer to maximize the score provided by the environment.  

The problem uses [Unity's ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/) to provide an environment in which there are both yellow and blue bananas on a 2D surface in a 3D space.  The objective is to collect as many yellow bananas as possible while avoiding the blue bananas before the game ends after a fixed amount of time.  You interact with this environment in first person, and have 4 possible actions (move forwards, move backwards, turn clockwise, turn counter-clockwise) and a default action of moving forward is provided if no input is given.  

## Learning Environment Set Up

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
pip install unityagents

# Install the ipython kernel package
conda install ipykernel
# Install the ipython kernel to the conda environment
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

To run the notebook, within the drlnd conda environment run `jupyter lab` and navigate to the the appropriate ipython notebook.


## Using this program

Learning and a resultant view of both a random state, and learned state are achievable by following the progression in the Navigation-checkpoint.ipynb python notebook.
