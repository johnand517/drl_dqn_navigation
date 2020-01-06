# Deep Reinforcement Learning - Deep Q Network - Unity Navigation Project

## Project Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course in the Artificial Intelligent program.  The problem uses [Unity's ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/) to provide an environment in which there are both yellow and blue bananas on a 2D surface in a 3D space.  The objective is to collect as many yellow bananas as possible while avoiding the blue bananas.  You interact with this environment in first person, and can move forward and backwards and turn clockwise or counter clockwise.  Through setting up a DRL agent, the objective for this code is to set up an agent that can train the computer to efficiently solve this problem through only knowledge of the state of the environment and the actions available.

## Project Objectives

We will use a Deep Q Network (DQN) to train the agent to solve the problem, starting from a randomly initialized state-action, we will attempt to maximize a reward function (Q), which is a function of both the state of the environment and and action provided for that state.

To achive this, we will require:
-A sequentional deep neural network that learns the optimal actions based on our rewards function and given our current state (local q network)
-A sequential deep nerual network that keeps its weights fixed for a determined amount of iterations, and is used to determine estimted rewards from a successive state given the current action (target q network)
-A replay buffer that stores prior experiences as our agent learns from the environment

Our program acts upon the environment by initially choosing actions for the given state more-or-less at random, and then determines the reward, and next state given the action chosen.  This "experience" is then stored in the replay buffer.  We then update the current state to the next state determined from the chosen action and repeat this process.

Learning from the environment happens by chosing a sample of experiences from the replay buffer after a determined number of steps through the environment.  From these sampled experiences, we calculate both the expected rewards given the sampled action/state from our experiences (using our local q network), as well as the estimated successive rewards from the experienced next states and possible and the best possible action for those states (using our target q network).  We determine the loss for the local q network as the difference in rewards between the expected and estimated rewards, and backpropogate and update based on this loss calculation.  After a determined number of learning updates, we then refresh our target q network weights to equal that of the current local network weights.

As we march through this process, we use a decaying greedy-epsilon parameter that is set to start by favoring randomly chosen actions during initial training, and slowly diminishing to favor the best possible action as we complete more training steps. 

We also include Double DQN as part of our learning process, which stabilizes our learning by calculating our estimated best action from the next state using the local q network, and using those actions to calculate estimated next state reward from our target q network see [ref: Double DQN Paper](https://arxiv.org/abs/1509.06461).

## Areas for improvement


## Environment Set Up



## Using this program
