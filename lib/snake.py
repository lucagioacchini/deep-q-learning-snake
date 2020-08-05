import pygame
import random
import sys
import numpy as np
from lib.agent import Agent
import time
# pylint: disable=unused-wildcard-import
from lib.config import *
from pygame.locals import *
from lib.stats import *


class Environment():
    """
    Deep Q Learning enviroment. It sets up a Snake game, initializing the 
    snake and the food objects. During training the DQL agent chooses if
    the snake moves randomly or by exploiting the learned strategy by applying
    the epsilon greedy strategy.
    After each step the agent's memory is updated and the network is trained. 
    During testing the agent exploits the network performing the action 
    returning the best discounted return.

    Attributes:
        display (bool): true if the game must be displayed
        agent (Agent instance): deep-q-learning agent object
        episode (int): number of current episde
        reward (float): value of the current reward
        stop (bool): true if the game ended
        stat (Stats instance): instance of the Stats class
        train (bool): true if training the network
        score (int): value of actual score
        f (pygame.font): pygame font definition
        clock (pygame.time): pygame speed 
        step_ctr (int): number of current step
        explore_ctr (int): number of performed exploration
        exploit_ctr (int): number of performed exploitation
        eps (float): value of the current epsilon
        state (np.array): current state describing the game status
        action (int): current action chosen by the agent

    Args:
        stat (undefined): instance of the Stats class
        episode (int): number of current episode
        agent (Agent instance): deep-q-learning agent object
        train (bool): true if training the network
        display (bool): true if the game must be displayed

    """

    def __init__(self, stat, episode, agent, train, display):
        self.display = display
        self.agent = agent
        self.episode = episode
        self.reward = 0
        self.stop = False
        self.stat = stat
        self.train = train
        self.score = 0  # Initial score
        self.f = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()
        self.step_ctr = 0
        self.explore_ctr = 0
        self.exploit_ctr = 0
        self.eps = 0

        self.state = []
        self.action = 0

        # Screen definition
        # pylint: disable=undefined-variable
        if self.display:
            self.screen = pygame.display.set_mode((1020, 620), DOUBLEBUF)
        pygame.display.set_caption('Snake')

        # Generate snake and food
        self.snake = Snake()
        self.food = Food()

    def render(self):
        """
        Render the pygame images displaying the game UI with additional 
        information about the DQN performances. During training the metrics
        (accuracy and loss) trends are shown. During testing a sample
        network representing the input and ouput layers is displayed.

        """
        # Refill the screen
        self.screen.fill((22, 29, 31))

        # Render the snake
        for i in range(0, self.snake.len):
            self.screen.blit(
                self.snake.img, (
                    self.snake.x[i], self.snake.y[i]
                )
            )
            self.screen.blit(
                self.snake.bord1, (
                    self.snake.x[i], self.snake.y[i]
                )
            )
            self.screen.blit(
                self.snake.bord2, (
                    self.snake.x[i], self.snake.y[i]
                )
            )
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

        # Border
        # Left bar
        # pylint: disable=too-many-function-args
        bord = pygame.Surface((10, H))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, 0))
        # Right bar
        bord = pygame.Surface((10, H))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (W-10, 0))
        # Up bar
        bord = pygame.Surface((W, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, 0))
        # Down bar
        bord = pygame.Surface((W, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0, H-10))

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

    def step(self, act, state):
        """
        At each game step evaluate the game status (if the snake eats itself
        or collides with the borders). Then perform the snake move and get the 
        reward.

        Args:
            act (int): action chosen by the agent
            state (np.array): state vector describing the game status

        """
        self.state = state
        self.reward = 0
        self.snake.ate = False
        self.snake.crashed = False
        self.clock.tick(SPEED)

        # Get the event
        for e in pygame.event.get():
            # If press quit, game ends
            # pylint: disable=undefined-variable, no-member
            if e.type == QUIT:
                pygame.quit()
                quit()
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
        died = self.selfEat()
        ate = self.foodEat()
        if not died:
            died = self.hitBorder()

        if self.hitBorder() or self.selfEat():
            self.snake.crashed = True
        if self.foodEat():
            self.snake.ate = True

        # Set Reward
        self.reward = self.agent.setReward(died, ate, state)

        # If die end the game
        if died:
            self.stop = True

        if self.display:
            try:
                self.render()
            except:
                pass

    def run(self):
        """
        Core of the snake game. At each step get the initial position, apply
        the epsilon greedy strategy, select the next action, perform the move
        and get the final state. Then update the agent's experience and train 
        the DQN.

        """
        # Get state 1
        state1 = self.agent.getState(self.snake, self.food)
        # Choose default actin
        action = 0
        # Perform action
        self.step(action, state1)
        self.step_ctr += 1
        # Get state 2
        state2 = self.agent.getState(self.snake, self.food)
        # Learn from experience
        experience = (state1, action, self.reward, state2)
        if self.train:
            # Update agent's memory
            self.agent.memory.push(experience)
            # Train the network and get the metrics
            history = self.agent.memory.replay(self.stop)
            self.stat.loss.append(history['loss'][0])
            self.stat.accuracy.append(history['accuracy'][0]*100)

        while True:
            if self.stop:
                break
            self.step_ctr += 1

            # Get state 1
            state1 = self.agent.getState(self.snake, self.food)
            # Exploration Rate and choose action
            if self.train:
                self.eps = self.agent.getEpsilon(self.step_ctr)
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
            state2 = self.agent.getState(self.snake, self.food)  # Get state 2
            # Manage memory
            experience = (state1, action, self.reward, state2)
            if self.train:
                self.agent.memory.push(experience)
                history = self.agent.memory.replay(self.stop)
                self.stat.loss.append(history['loss'][0])
                self.stat.accuracy.append(history['accuracy'][0]*100)

    def selfEat(self):
        """
        Check if the snake eats itself and return the bool status.

        """
        # Snake eats itself
        i = self.snake.len

        for j in range(1, i):
            if self.snake.x[0] == self.snake.x[j] \
                    and self.snake.y[0] == self.snake.y[j]:

                return True
        return False

    def foodEat(self):
        """
        Check if the snake eats the food and return the bool status.

        """
        # Snake eats apple
        if self.snake.x[0] == self.food.pos[0] \
                and self.snake.y[0] == self.food.pos[1]:
            self.score += 1
            self.snake.x.append(700)
            self.snake.y.append(700)
            self.snake.len += 1
            self.food.pos = self.food.genPos()
            for i in range(self.snake.len):
                if self.food.pos[0] == self.snake.x[i] \
                or self.food.pos[1] == self.snake.y[i]:
                    self.food.pos = self.food.genPos()

            return True

    def hitBorder(self):
        """
        Check if the snake hits the border and return the bool status.

        """
        # Snake reach the border
        if self.snake.x[0] < 10 or self.snake.x[0] > W-40 \
                or self.snake.y[0] < 10 or self.snake.y[0] > H-40:
            return True
        else:
            return False


class Snake():
    """
    Generate the snake object and design the movements.

    Attributes:
        x (list): x coordinates of the snake blocks
        y (list): y coordinates of the snake blocks
        dir (int): direction taken by the snake
        ate (bool): true if the snake ate the food
        crashed (bool): true if the snake crashed with the borders
        img (pygame.Surface): snake blocks to be rendered
        bord1 (pygame.Surface): snake blocks to be rendered
        bord2 (pygame.Surface): snake blocks to be rendered
        len (int): length of the snake

    """

    def __init__(self):
        # Snake coordinates array
        self.x = [180, 180, 180, 180, 180, 180]
        self.y = [180, 160, 158, 156, 154, 152]

        self.dir = 0  # Initial direction

        self.ate = False
        self.crashed = False

        # Draw snake
        # pylint: disable=too-many-function-args
        self.img = pygame.Surface((20, 20))
        self.img.fill((255, 255, 255))
        self.bord1 = pygame.Surface((1, 20))
        self.bord1.fill((0, 0, 0))
        self.bord2 = pygame.Surface((20, 1))
        self.bord2.fill((0, 0, 0))

        # Snake length
        self.len = len(self.x)

    def move(self):
        """
        Update the snake position according to the movement and direction.

        """
        # Snake block flooding
        i = self.len-1
        while i >= 1:
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]
            i -= 1

        # Snake follows direction
        if self.dir == 0:
            self.y[0] += 20
        elif self.dir == 1:
            self.x[0] += 20
        elif self.dir == 2:
            self.y[0] -= 20
        elif self.dir == 3:
            self.x[0] -= 20


class Food():
    """
    Generate the food object and its random position in the screen

    Attributes:
        pos (tuple): x and y coordinates of the food
        img (pygame.Surface): food object to be rendered

    """

    def __init__(self):
        # Random apple position
        self.pos = self.genPos()

        # Draw apple
        # pylint: disable=too-many-function-args
        self.img = pygame.Surface((20, 20))
        self.img.fill((163, 51, 51))

    def genPos(self):
        """
        Generate the random position of the food in a predetermined grid.

        """
        randx = random.randint(20, W-40)
        randy = random.randint(20, H-40)
        if randx-randx % 20 < 20:
            randx = 10
        elif randx-randx % 20 > W-40:
            randx = W-40
        else:
            randx = randx-randx % 20

        if randy-randy % 20 < 20:
            randy = 10
        elif randy-randy % 20 > H-40:
            randy = H-40
        else:
            randy = randy-randy % 20

        return (randx, randy)
