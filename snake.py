import pygame, random, sys
from pygame.locals import *
import numpy as np
from agent import Agent
import time
from config import *
from pygame.locals import *
from stats import * 


def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
        return True
    else:
        return False

def die(score):
    pygame.time.wait(TIMEOUT)
    ##print(f'Score: {score}')
    return True
    #sys.exit(0)



class Environment():
    def __init__(self, stat, episode, agent, train, display):
        self.display = display
        self.agent = agent
        self.episode = episode
        self.reward = 0
        self.stop = False
        self.stat = stat
        self.train = train
        self.score = 0 # Initial score
        self.f = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()
        self.step_ctr = 0
        self.explore_ctr = 0
        self.exploit_ctr = 0
        self.eps = 0

        # Screen definition
        #if self.display: self.screen = pygame.display.set_mode((620, 620))
        if self.display: self.screen = pygame.display.set_mode((1020, 620), DOUBLEBUF)
        pygame.display.set_caption('Snake')
        
        # Generate snake and food
        self.snake = Snake()
        self.food = Food()
    
    def render(self):
        # Border
        #border = pygame.draw.rect(self.screen, (0, 100, 255), (600, 600, 162, 100), 3) 
        #border = pygame.rect.Rect(600, 600)
        #self.screen.blit(border)

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

        txt = f'Epsilon: {round(self.eps, 3)}   Explore: {self.explore_ctr}   Exploit: {self.exploit_ctr}'
        t = self.f.render(txt, True, (255, 255, 255))
        self.screen.blit(t, (630, 50))



        # Border
        # Left bar
        bord = pygame.Surface((10, H))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0,0))
        # Right bar
        bord = pygame.Surface((10, H))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (W-10,0))
        # Up bar
        bord = pygame.Surface((W, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0,0))
        # Down bar
        bord = pygame.Surface((W, 10))
        bord.fill((255, 255, 255))
        self.screen.blit(bord, (0,H-10))
        
        
        loss, lsize = self.stat.plotLoss()
        surf = pygame.image.fromstring(loss, lsize, 'RGB')
        
        self.screen.blit(surf, (630,80))

        acc, asize = self.stat.plotAccuracy()
        surf = pygame.image.fromstring(acc, asize, 'RGB')
        
        self.screen.blit(surf, (630, 350))
        
        pygame.display.update()
        
        
    def step(self, act, state):
        self.reward = 0
        self.snake.ate = False
        self.snake.crashed = False
        self.clock.tick(SPEED)

        # Get the event
        for e in pygame.event.get():
            # If press quit, game ends
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
        
        #self.snake.dir = act
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
            self.stop = die(self.score)
        
        if self.display:
            try:
                self.render()
            except:
                pass



    def run(self):
        # Get state 1
        state1 = self.agent.getState(self.snake, self.food)
        # Choose default actin
        action = 0
        # Perform action
        self.step(action, state1)
        self.step_ctr+=1
        # Get state 2
        state2 = self.agent.getState(self.snake, self.food)
        # Learn from experience
        experience = (state1, action, self.reward, state2)
        if self.train:
            self.agent.memory.push(experience)
            #self.agent.memory.replay(self.stop)
            history = self.agent.memory.replay(self.stop)
            self.stat.loss.append(history['loss'][0])
            self.stat.accuracy.append(history['accuracy'][0]*100)
        
        while True:
            if self.stop:
                break
            self.step_ctr+=1

            # Get state 1
            state1 = self.agent.getState(self.snake, self.food)
            #print(f'Old state:\n\t{state1}')
            # Exploration Rate and choose action
            if self.train: self.eps = self.agent.getEpsilon(self.step_ctr)
            else: self.eps = 0

            if random.random() < self.eps:
            #if random.randint(0,1) < self.agent.getEpsilon(self.step_ctr):
                # Explore
                self.explore_ctr += 1 # For stats
                action = random.randint(0,3)
            else:
                # Exploit
                self.exploit_ctr += 1 # For stats
                action = self.agent.memory.exploit(state1)
            #print(f'Choosen action:\n\t{action}')
            # Perform action
            self.step(action, state1)
            # Get state 2           
            state2 = self.agent.getState(self.snake, self.food) # Get state 2
            #print(f'New State:\n\t{state2}')
            # Manage memory
            experience = (state1, action, self.reward, state2)
            #print(f'Experience')
            if self.train:
                self.agent.memory.push(experience)
                #self.agent.memory.train_short_memory(state1, action, self.reward, state2, self.stop)
                history = self.agent.memory.replay(self.stop)
                self.stat.loss.append(history['loss'][0])
                self.stat.accuracy.append(history['accuracy'][0]*100)
                #print()
        #history = self.agent.memory.replay(self.stop)
        #self.stat.loss.append(history['loss'][0])
        #self.stat.accuracy.append(history['accuracy'][0]*100)
            
    def selfEat(self):
        # Snake eats itself
        i = self.snake.len

        for j in range(1,i):
            if self.snake.x[0] == self.snake.x[j] \
            and self.snake.y[0] == self.snake.y[j]:
                
                return True
        return False

    def foodEat(self):
        # Snake eats apple
        """
        collided = collide(
            self.snake.x[0], self.food.pos[0],
            self.snake.y[0], self.food.pos[1],
            20, 20, 20, 20
        )
        """
        if self.snake.x[0] == self.food.pos[0] \
        and self.snake.y[0] == self.food.pos[1]:
            self.score+=1
            self.snake.x.append(700)
            self.snake.y.append(700)
            self.snake.len += 1
            self.food.pos = self.food.genPos()
            for i in range(self.snake.len):
                if self.food.pos[0] == self.snake.x[i] or self.food.pos[1] == self.snake.y[i]:
                    self.food.pos = self.food.genPos()

            return True
    
    def hitBorder(self):
        # Snake reach the border
        if self.snake.x[0] < 10 or self.snake.x[0] > W-40 \
        or self.snake.y[0] < 10 or self.snake.y[0] > H-40: 
            return True
        else:
            return False

class Snake():
    def __init__(self):
        # Snake coordinates array
        self.x = [180, 180, 180, 180, 180, 180]
        self.y = [180, 160, 158, 156, 154, 152]

        self.dir = 0 # Initial direction

        self.ate = False
        self.crashed = False

        #if self.display:
        # Draw snake
        self.img = pygame.Surface((20, 20))
        self.img.fill((255, 255, 255))
        self.bord1 = pygame.Surface((1, 20))
        self.bord1.fill((0, 0, 0))
        self.bord2 = pygame.Surface((20, 1))
        self.bord2.fill((0, 0, 0))

        # Snake length
        self.len = len(self.x)
        
    def move(self):
        # Snake block flooding
        i = self.len-1
        while i >= 1:
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]
            i -= 1
        
        # Snake follows direction
        if self.dir == 0: self.y[0] += 20
        elif self.dir == 1: self.x[0] += 20
        elif self.dir == 2: self.y[0] -= 20
        elif self.dir == 3: self.x[0] -= 20


class Food():
    def __init__(self):
        # Random apple position
        self.pos = self.genPos()
        
        #if self.display:
            # Draw apple
        self.img = pygame.Surface((20, 20))
        self.img.fill((163, 51, 51))

    def genPos(self):
        randx = random.randint(20, W-40)
        randy = random.randint(20, H-40)
        if randx-randx%20 < 20:randx=10
        elif randx-randx%20 > W-40:randx=W-40
        else: randx = randx-randx%20

        if randy-randy%20 < 20:randy=10
        elif randy-randy%20 > H-40:randy=H-40
        else: randy = randy-randy%20

        return (randx, randy)

