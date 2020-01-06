buff_size = int(1.0e6)  # The maximum items stored in our buffer
batch_size = 64  # The batch to sample from our buffer in a learning episode
iter_memory_pull = 4  # The number of state/action updates before learning
iter_target_update = 10  # learning iterations before syncing target q network weights to local q network weights
Gamma = 0.99  # discount factor
learning_rate = 5.0e-5  # The learning rate for our optimizer
tau = 1.0e-3  # Soft update parameter for target q network weight updates

