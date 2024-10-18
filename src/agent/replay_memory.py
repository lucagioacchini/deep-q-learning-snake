import random
import numpy as np
from .deep_q import DeepQNetwork

class ReplayMemory():
    """Replay memory used by the agent. It stores a number of experiences defined
    as (state, action, reward, next state). If the memory capacity is 
    exceeded, the older experiences are dropped

    Parameters:
        model (agent.DeepQNetwork): DQN model
        capacity (int): memory capacity
        batch_size (int): number of samples to retrieve from the memory
        gamma (float): discounting factor for the Deep Q-Learning

    Attributes:
        model (DeepQNetwork): DQN model
        capacity (int): memory capacity
        memory (list): actual memory
        push_count (int): number of performed updates
        batch_size (int): number of experiences to randomly sample 
    
    Methods:
        push(experience): Update the agent's replay memory
        sample(): Perform a random sample of the memory
        replay(stop): Predict the Q-value of the (next state, action) pairs
        exploit(): Choose the best action exploiting the trained networks
    """

    def __init__(self, model:DeepQNetwork, capacity:int, batch_size:int, gamma:float):
        self.model = model.model
        self.batch_size = batch_size
        self.capacity = capacity
        self.gamma = gamma
        self.memory = []
        self.push_count = 0

    def push(self, experience:tuple):
        """Update the agent's replay memory. If the memory capacity is 
        exceeded, the older experiences are dropped

        Parameters:
            experience (tuple): game observation. It is defined as:
                                (state, action, reward, next state)

        """
        if len(self.memory) < self.capacity:
            # Start achieving memory from experience
            self.memory.append(experience)
        else:
            # Progressively replace the acquired experience
            # with fresher one
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self):
        """Perform a random sample of the memory

        Returns:
            np.array: the batch sampled from the memory
        """
        # Randomly sample memory
        return random.sample(self.memory, self.batch_size)

    def replay(self, stop:bool):
        """Predict the Q-value of the (next state, action) pairs. Get the 
        action corresponding to the greatest Q-value. Predict the Q-value of
        the (current state, action) pairs. Replace the obtained value with the 
        discounted greatest Q-value in correspondence of the considered action.
        Train the network with the new discounted Q-values when the current 
        state is used as input. 

        Parameters:
            stop (bool): true if the snake died

        Returns:
            _type_: the training history
        """
        # Get the random sampled memory
        if len(self.memory) >= self.batch_size:
            batch = self.sample()
        else:
            batch = self.memory

        # Reshape the experience
        state, act, reward, nxt_state = batch[0]
        nxt_state = np.reshape(nxt_state, (1, 11))
        state = np.reshape(state, (1, 11))
        reward = np.asarray(reward)
        act = np.asarray(act)

        # Build the input dataset
        if len(batch) > 1:
            for state1, act1, reward1, nxt_state1 in batch[1:]:
                nxt_state1 = np.reshape(nxt_state1, (1, 11))
                nxt_state = np.vstack((nxt_state, nxt_state1))

                reward1 = np.asarray(reward1)
                reward = np.vstack((reward, reward1))

                state1 = np.reshape(state1, (1, 11))
                state = np.vstack((state, state1))

                act1 = np.asarray(act1)
                act = np.vstack((act, act1))
        q_opt = reward

        if not stop:
            # Predict the Q-value of the next state
            q_prime = self.model.predict(nxt_state)
            # Get the action providing the greatest one
            max_q_prime = np.amax(q_prime, axis=1)
            max_q_prime = np.reshape(max_q_prime, (q_prime.shape[0], 1))
            # Comput the discounted return wrt the greatest qvalue
            q_opt = np.add(reward, self.gamma * max_q_prime)
        # Predict the Q-value of the current state    
        target = self.model.predict(state)
        # Replace the current Q-value with the discounted return in 
        # correspondence of the action ensuring the greatest next Q-value
        try:
            for i in range(act.shape[0]):
                target[i, int(act[i])] = q_opt[i]
        except IndexError:
            target[0, int(act)] = q_opt
        # Train the model with the new current Q-values
        history = self.model.fit(
            state, target, epochs=1,
            verbose=0, batch_size=state.shape[0]
        )

        return history.history

    def exploit(self, state:np.array):
        """Choose the best action exploiting the trained networks

        Parameters:
            state (np.array): state vector representing the game status

        Returns:
            int: the action to perform predicted by the DQN
        """
        state = np.reshape(state, (1, 11))
        pred = self.model.predict(state)[0]
        best_act = np.argmax(pred)

        return best_act
