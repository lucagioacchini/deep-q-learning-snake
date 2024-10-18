from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import random
import math
import numpy as np
# pylint: disable=unused-wildcard-import
from lib.stats import *
from lib.config import *


class DQN():
    """
    Neural Network used in Deep-Q-Learning

    Attributes:
        model (keras.Sequential): neural network model

    """

    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        """
        Initialize the neural network

        """
        model = Sequential()
        model.add(Dense(L1, input_shape=(STATE_L,), activation='relu'))
        model.add(Dense(L2, activation='relu'))
        model.add(Dense(L3, activation='relu'))
        model.add(Dense(4))

        model.compile(loss='mse', optimizer=Adam(
            learning_rate=LR), metrics=['accuracy'])
        model.summary()

        return model


class ReplayMemory():
    """
    Replay memory used by the agent. It stores a number of experiences defined
    as (state, action, reward, next state). If the memory capacity is 
    exceeded, the older experiences are dropped

    Attributes:
        model (keras.Sequential): DQN model
        capacity (int): memory capacity
        memory (list): actual memory
        push_count (int): number of performed updates
        batch_size (int): number of experiences to randomly sample during 
                          network training

    Args:
        model (model.Sequential): DQN model

    """

    def __init__(self, model):
        self.model = model
        self.capacity = CAPACITY
        self.memory = []
        self.push_count = 0
        self.batch_size = BATCH

    def push(self, experience):
        """
        Update the agent's replay memory. If the memory capacity is 
        exceeded, the older experiences are dropped

        Args:
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
        """
        Perform a random sample of the memory

        """
        # Randomly sample memory
        return random.sample(self.memory, self.batch_size)

    def replay(self, stop):
        """
        Predict the Q-value of the (next state, action) pairs. Get the 
        action corresponding to the greatest Q-value. Predict the Q-value of
        the (current state, action) pairs. Replace the obtained value with the 
        discounted greatest Q-value in correspondence of the considered action.
        Train the network with the new discounted Q-values when the current 
        state is used as input. 

        Args:
            stop (bool): true if the snake died

        """
        # Get the random sampled memory
        if len(self.memory) >= self.batch_size:
            batch = self.sample()
        else:
            batch = self.memory

        # Reshape the experience
        state, act, reward, nxt_state = batch[0]
        nxt_state = np.reshape(nxt_state, (1, STATE_L))
        state = np.reshape(state, (1, STATE_L))
        reward = np.asarray(reward)
        act = np.asarray(act)

        # Build the input dataset
        if len(batch) > 1:
            for state1, act1, reward1, nxt_state1 in batch[1:]:
                nxt_state1 = np.reshape(nxt_state1, (1, STATE_L))
                nxt_state = np.vstack((nxt_state, nxt_state1))

                reward1 = np.asarray(reward1)
                reward = np.vstack((reward, reward1))

                state1 = np.reshape(state1, (1, STATE_L))
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
            q_opt = np.add(reward, GAMMA * max_q_prime)
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

    def exploit(self, state):
        """
        Choose the best action exploiting the trained networks

        Args:
            state (np.array): state vector representing the game status

        """
        state = np.reshape(state, (1, STATE_L))
        pred = self.model.predict(state)[0]
        best_act = np.argmax(pred)

        return best_act


class Agent():
    """
    Deep Q-Learning agent. It gets the state from the game, assigns the rewards
    and chooses the epsilon through the epsilon greedy strategy.

    Attributes:
        model (keras.Sequential): DQN model
        memory (ReplayMemory): ReplayMemory instance

    """

    def __init__(self):
        self.model = DQN().model
        self.memory = ReplayMemory(self.model)

    def getState(self, snake, food):
        """
        Get the current state of the game defined as a list with:
        1: if there is an obstacle on the right by the snake POV
        1: if there is an obstacle on the left by the snake POV
        1: if there is an obstacle forward wrt by the snake POV
        1: if there is the food on the right wrt the snake head
        1: if there is the food on the left wrt the snake head
        1: if there is the food on the top wrt the snake head
        1: if there is the food on the bottom wrt the snake head
        1: if the snake is gowing down
        1: if the snake is gowing up
        1: if the snake is gowing right
        1: if the snake is gowing left

        Args:
            snake (Snake): Snake class instance
            food (Food): Food class instance

        """
        state = np.zeros(STATE_L, dtype=int)

        # Snake goes down
        if snake.dir == 0:
            state[7] = True
            # Obstacle Right
            head = (snake.x[0]-20, snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 10:
                    state[0] = True
                    break
            # Obstacle Left
            head = (snake.x[0]+20, snake.y[0])
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[0] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > W-40:
                    state[1] = True
                    break
            # Obstacle Forward
            head = (snake.x[0], snake.y[0]+20)
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[1] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > H-40:
                    state[2] = True
                    break

        # Snake goes Up
        if snake.dir == 2:
            state[8] = True
            # Obstacle Right
            head = (snake.x[0]+20, snake.y[0])
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[0] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > W-40:
                    state[0] = True
                    break
            # Obstacle Left
            head = (snake.x[0]-20, snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 10:
                    state[1] = True
                    break
            # Obstacle Forward
            head = (snake.x[0], snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 10:
                    state[2] = True
                    break

        # Snake goes Right
        if snake.dir == 1:
            state[9] = True
            # Obstacle Right
            head = (snake.x[0], snake.y[0]+20)
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[1] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > H-40:
                    state[0] = True
                    break
            # Obstacle Left
            head = (snake.x[0], snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 10:
                    state[1] = True
                    break
            # Obstacle Forward
            head = (snake.x[0]+20, snake.y[0])
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[0] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > W-40:
                    state[2] = True
                    break

        # Snake goes Left
        if snake.dir == 3:
            state[10] = True
            # Obstacle Right
            head = (snake.x[0], snake.y[0]-20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[1] < 10:
                    state[0] = True
                    break
            # Obstacle Left
            head = (snake.x[0], snake.y[0]+20)
            for i in range(1, snake.len):
                # if head == (snake.x[i], snake.y[i]) or head[1] > 580:
                if head == (snake.x[i], snake.y[i]) or head[0] > H-40:
                    state[1] = True
                    break
            # Obstacle Forward
            head = (snake.x[0]-20, snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] < 10:
                    state[2] = True
                    break

        # Food position wrt head
        state[3] = int(snake.x[0] < food.pos[0])  # Food Right
        state[4] = int(snake.x[0] > food.pos[0])  # Food Left
        state[5] = int(snake.y[0] > food.pos[1])  # Food Up
        state[6] = int(snake.y[0] < food.pos[1])  # Food Down

        return state

    def getEpsilon(self, current_step):
        """
        Epsilon greedy strategy

        Args:
            current_step (int): number of the current step

        """
        eps = END + (START - END)*math.exp(-1*current_step*DECAY)

        return eps

    def setReward(self, died, ate, state):
        """
        Assigns the reward to the (state, action) pair

        Args:
            died (bool): true if the snake died
            ate (bool): true if the snake ate the food
            state (np.array): state describing the game status

        """
        reward = 0
        if died:
            reward = DIED
        elif ate:
            reward = ATE
        else:
            reward = -1

        return reward
