import math
import numpy as np
from .deep_q import DeepQNetwork
from .replay_memory import ReplayMemory

class Agent():
    """Deep Q-Learning agent. It gets the state from the game, assigns the rewards
    and chooses the epsilon through the epsilon greedy strategy.

    Parameters:
        screen_width (int): Width of the game screen in pixels.
        screen_height (int): Height of the game screen in pixels.
        memory_capacity (int): memory capacity
        memory_batch_size (int): number of samples to retrieve from the memory
        eps_decay (float): The epsilon decay value for the Epsilon greedy strategy
        gamma (float): discounting factor for the Deep Q-Learning

    Attributes:
        screen_width (int): Width of the game screen in pixels.
        screen_height (int): Height of the game screen in pixels.
        memory (ReplayMemory): The memory used in the Deep Q-Learning 
        eps_decay (float): The epsilon decay value for the Epsilon greedy strategy

    Methods:
        load_weights(w_path): Load the pre-trained weights
        get_state(snake, food):Get the current state of the game
        get_epsilon(current_step): Update the epsilon for the Epsilon greedy strategy
        set_reward(died, ate): Assigns the reward to the (state, action) pair
    """

    def __init__(self, screen_width:int, screen_height:int, memory_capacity:int, 
                 memory_batch_size:int, eps_decay:float, gamma:float):
        
        # Set screen size
        self.screen_width=screen_width 
        self.screen_height=screen_height
        
        # Set memory
        self.memory = ReplayMemory(
            model=DeepQNetwork(), 
            capacity=memory_capacity, 
            batch_size=memory_batch_size,
            gamma=gamma)
        self.eps_decay = eps_decay

    def load_weights(self, w_path:str):
        """Load the pre-trained weights.

        Parameters:
            w_path (str): path of the saved weights
        """
        self.memory.model.load_weights(w_path)

    def save_weights(self, w_path:str):
        """Save the trained weights.

        Parameters:
            w_path (str): path where to save the weights
        """
        self.memory.model.save_weights(w_path)

    def get_state(self, snake, food):
        """Get the current state of the game.

        Parameters:
            snake (Snake): Snake class instance
            food (Food): Food class instance

        Returns:
            np.array: the current state vector
        """
        state = np.zeros(11, dtype=int)

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
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_width-40:
                    state[1] = True
                    break
            # Obstacle Forward
            head = (snake.x[0], snake.y[0]+20)
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_height-40:
                    state[2] = True
                    break

        # Snake goes Up
        if snake.dir == 2:
            state[8] = True
            # Obstacle Right
            head = (snake.x[0]+20, snake.y[0])
            for i in range(1, snake.len):
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_width-40:
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
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_height-40:
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
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_width-40:
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
                if head == (snake.x[i], snake.y[i]) or head[0] > self.screen_height-40:
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

    def get_epsilon(self, current_step:int):
        """Update the epsilon for the Epsilon greedy strategy

        Parameters:
            current_step (int): number of the current step
        
        Returns:
            float: the current epsilon value

        """
        eps_start = 1.0
        eps_end = .03
        eps = eps_end+(eps_start-eps_end)*math.exp(-1*current_step*self.eps_decay)

        return eps

    def set_reward(self, died:bool, ate:bool):
        """Assigns the reward to the (state, action) pair

        Parameters:
            died (bool): true if the snake died
            ate (bool): true if the snake ate the food
        
        Returns:
            int: the current reward
        """
        reward = 0
        if died:
            reward = -10
        elif ate:
            reward = 10
        else:
            reward = -1

        return reward
