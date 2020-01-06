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

Through applying the above learning agent, we are able to achieve a maximal score for a given run of over 17 net yellow bananas when averaged over the prior 100 episodes at a given epoch.  We converage at this result after 1250 epochs.  The results of our scores through successive training episodes are as shown:

## Areas for improvement

- More exhaustive hyperparameter tuning
- Implementation of [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) which weights replay experiences by the loss parameter associated with them, and uses these weights to generate our sample batch while learning
- Implementation of [Dueling Q Networks](https://arxiv.org/abs/1511.06581) which learns separately on state values and action values and combines the results from learning.  This could potentially more robustly identify our reward function.

