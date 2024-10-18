import pygame
import random
import numpy as np
from .food import Food
from .snake import Snake
from ..agent.agent import Agent
from deepqsnake.stats import Statistics


class SnakeEnvironment():
    """Deep Q Learning environment. It sets up a Snake game, initializing the 
    snake and the food objects. During training the DQL agent chooses if
    the snake moves randomly or by exploiting the learned strategy by applying
    the epsilon greedy strategy.
    After each step the agent's memory is updated and the network is trained. 
    During testing the agent exploits the network performing the action 
    returning the best discounted return.

    Parameters:
        screen_width (int): Width of the game screen in pixels.
        screen_height (int): Height of the game screen in pixels.
        stat (Statistics): An instance of the Statistics class to track game metrics.
        episode (int): The current episode number.
        agent (Agent): An instance of the Agent class representing the DQL agent.
        train (bool): Flag indicating whether the environment is in training mode.
        display (bool): Flag indicating whether to display the game visually.

    Attributes:
        width (int): Width of the game screen.
        height (int): Height of the game screen.
        stat (Statistics): Statistics object for tracking metrics.
        episode (int): Current episode number.
        agent (Agent): DQL agent object.
        train (bool): Training mode flag.
        display (bool): Display mode flag.
        state (list): Current state of the environment.
        reward (int): Current reward value.
        score (int): Current game score.
        action (int): Current action being performed.
        eps (float): Current epsilon value for epsilon-greedy strategy.
        stop (bool): Flag indicating if the game should stop.
        step_ctr (int): Counter for the number of steps taken.
        explore_ctr (int): Counter for exploration actions.
        exploit_ctr (int): Counter for exploitation actions.
        screen (pygame.Surface): Pygame screen object for display.
        snake (Snake): Snake object representing the player.
        food (Food): Food object representing the target.

    Methods:
        render(): Renders the game state on the screen.
        step(act: int, state: np.array): Performs a single step in the game.
        run(): Runs the main game loop.
        self_eat(): Checks if the snake has eaten itself.
        food_eat(): Checks if the snake has eaten the food.
        hit_border(): Checks if the snake has hit the border.
    """

    def __init__(self, screen_width: int, screen_height: int, stat: Statistics,
                 episode: int, agent: Agent, train: bool, display: bool):
        self.width = screen_width
        self.height = screen_height
        self.stat = stat
        self.episode = episode
        self.agent = agent
        self.train = train
        self.display = display

        # Initial state
        self.state = []

        # Initial reward, score and action
        self.reward = 0
        self.score = 0
        self.action = 0
        self.eps = 0
        self.stop = False

        # Counters
        self.step_ctr = 0
        self.explore_ctr = 0
        self.exploit_ctr = 0

        # Screen definition
        self.f = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()
        if self.display:
            self.screen = pygame.display.set_mode((1020, 620))
        pygame.display.set_caption('Snake')

        # Generate snake and food
        self.snake = Snake()
        self.food = Food(self.width, self.height)

    def render(self):
        """Render the pygame images displaying the game UI with additional 
        information about the DQN performances. During training the metrics
        (accuracy and loss) trends are shown. During testing a sample
        network representing the input and ouput layers is displayed.

        """
        # Refill the screen
        self.screen.fill((22, 29, 31))

        # Render the snake
        for i in range(0, self.snake.len):
            self.screen.blit(
                self.snake.img, (self.snake.x[i], self.snake.y[i]))
            self.screen.blit(self.snake.bord1,
                             (self.snake.x[i], self.snake.y[i]))
            self.screen.blit(self.snake.bord2,
                             (self.snake.x[i], self.snake.y[i]))

        # Render the food
        self.screen.blit(self.food.img, self.food.pos)

        # Render the score
        txt = f'Score: {self.score}'
        t = self.f.render(txt, True, (255, 255, 255))
        self.screen.blit(t, (630, 10))

        txt = f'Episode: {self.episode}   Survival: {self.step_ctr}'
        t = self.f.render(txt, True, (255, 255, 255))
        self.screen.blit(t, (630, 30))

        txt = (f'Epsilon: {round(self.eps, 3)}   Explore: {self.explore_ctr}'
               f'   Exploit: {self.exploit_ctr}')
        t = self.f.render(txt, True, (255, 255, 255))
        self.screen.blit(t, (630, 50))

        # Border - Left bar
        bord = pygame.Surface((10, self.height))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, 0))

        # Border -  Right bar
        bord = pygame.Surface((10, self.height))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (self.width-10, 0))

        # Border -  Up bar
        bord = pygame.Surface((self.width, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, 0))

        # Border -  Down bar
        bord = pygame.Surface((self.width, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, self.height-10))

        # Plot metrics
        if self.train:
            loss, lsize = self.stat.plotLoss()
            surf = pygame.image.fromstring(loss, lsize, 'RGB')

            self.screen.blit(surf, (630, 80))

            acc, asize = self.stat.plotAccuracy()
            surf = pygame.image.fromstring(acc, asize, 'RGB')

            self.screen.blit(surf, (630, 350))

        # Plot sample network
        else:
            for i in range(len(self.state)):
                y = self.state[i]
                color = (72*(1-y)+255*y, 156*(1-y)+255*y, 81*(1-y)+255*y)
                pygame.draw.circle(self.screen, color, (670, 120+40*i), 14)

            for i in range(12):
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (820, 100+40*i), 14)

            for i in range(4):
                if i == self.action:
                    y = 0
                else:
                    y = 1
                color = (72*(1-y)+255*y, 156*(1-y)+255*y, 81*(1-y)+255*y)
                pygame.draw.circle(self.screen, color, (970, 260+40*i), 14)

            for i in range(len(self.state)):
                for j in range(12):
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (670+15, 120+40*i), (820-15, 100+40*j), 1)
                    for k in range(4):
                        pygame.draw.line(
                            self.screen, (255, 255, 255),
                            (820+15, 100+40*j), (970-15, 260+40*k), 1
                        )

        pygame.display.update()

    def step(self, act: int, state: np.array):
        """At each game step evaluate the game status (if the snake eats itself
        or collides with the borders). Then perform the snake move and get the 
        reward.

        Arguments:
            act (int): action chosen by the agent
            state (np.array): state vector describing the game status

        """
        self.state = state
        self.reward = 0
        self.snake.ate = False
        self.snake.crashed = False
        self.clock.tick(1000)

        # Manage actions
        if act == 2 and self.snake.dir != 0:
            self.snake.dir = 2
        elif act == 0 and self.snake.dir != 2:
            self.snake.dir = 0
        elif act == 3 and self.snake.dir != 1:
            self.snake.dir = 3
        elif act == 1 and self.snake.dir != 3:
            self.snake.dir = 1

        # Perform the move
        if not self.stop:
            self.snake.move()

        # Evaluate move
        died = self.self_eat()
        ate = self.food_eat()
        if not died:
            died = self.hit_border()
        if self.hit_border() or self.self_eat():
            self.snake.crashed = True
        if self.food_eat():
            self.snake.ate = True

        # Set Reward
        self.reward = self.agent.set_reward(died, ate)

        # If die end the game
        if died:
            self.stop = True

        if self.display:
            try:
                self.render()
            except:
                pass

    def run(self):
        """Core of the snake game. At each step get the initial position, apply
        the epsilon greedy strategy, select the next action, perform the move
        and get the final state. Then update the agent's experience and train 
        the DQN.

        """
        # Get state 1
        state1 = self.agent.get_state(self.snake, self.food)
        # Choose default actin
        action = 0
        # Perform action
        self.step(action, state1)
        self.step_ctr += 1

        # Get state 2
        state2 = self.agent.get_state(self.snake, self.food)
        # Learn from experience
        experience = (state1, action, self.reward, state2)
        if self.train:
            # Update agent's memory
            self.agent.memory.push(experience)
            # Train the network and get the metrics
            history = self.agent.memory.replay(self.stop)
            self.stat.loss.append(history['loss'][0])
            self.stat.accuracy.append(history['accuracy'][0]*100)

        while not self.stop:
            self.step_ctr += 1

            # Get state 1
            state1 = self.agent.get_state(self.snake, self.food)
            # Exploration Rate and choose action
            if self.train:
                self.eps = self.agent.get_epsilon(self.step_ctr)
            else:
                self.eps = 0

            if random.random() < self.eps:
                # Explore
                self.explore_ctr += 1  # For stats
                action = random.randint(0, 3)
            else:
                # Exploit
                self.exploit_ctr += 1  # For stats
                action = self.agent.memory.exploit(state1)
                self.action = action
            # Perform action
            self.step(action, state1)

            # Get state 2
            state2 = self.agent.get_state(self.snake, self.food)  # Get state 2
            # Manage memory
            experience = (state1, action, self.reward, state2)
            if self.train:
                self.agent.memory.push(experience)
                history = self.agent.memory.replay(self.stop)
                self.stat.loss.append(history['loss'][0])
                self.stat.accuracy.append(history['accuracy'][0]*100)

    def self_eat(self):
        """Check if the snake eats itself and return the bool status.

        Returns:
            bool: True if snake ate itself, False otherwise
        """
        # Snake eats itself
        i = self.snake.len

        for j in range(1, i):
            if self.snake.x[0] == self.snake.x[j] \
                    and self.snake.y[0] == self.snake.y[j]:

                return True
        return False

    def food_eat(self):
        """Check if the snake eats the food and return the bool status.

        Returns:
            bool: True if snake ate the food, False otherwise
        """
        # Snake eats apple
        if self.snake.x[0] == self.food.pos[0] \
                and self.snake.y[0] == self.food.pos[1]:
            self.score += 1
            self.snake.x.append(700)
            self.snake.y.append(700)
            self.snake.len += 1
            self.food.pos = self.food.gen_pos()
            for i in range(self.snake.len):
                if self.food.pos[0] == self.snake.x[i] \
                        or self.food.pos[1] == self.snake.y[i]:
                    self.food.pos = self.food.gen_pos()

            return True

    def hit_border(self):
        """Check if the snake hits the border and return the bool status.

        Returns:
            bool: True if snake hit the border, False otherwise
        """
        # Snake reach the border
        if self.snake.x[0] < 10 or self.snake.x[0] > self.width-40 \
                or self.snake.y[0] < 10 or self.snake.y[0] > self.height-40:
            return True
        else:
            return False
