import pygame, random, sys
from pygame.locals import *
import numpy as np
from agent import Agent
import time

SPEED = 500

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
        return True
    else:
        return False

def die(score):
    pygame.time.wait(100)
    return True
    #sys.exit(0)



class Environment():
    def __init__(self, agent):
        self.agent = agent
        self.reward = 0
        self.stop = False
        self.score = 0 # Initial score
        self.f = pygame.font.SysFont('Arial', 20)
        self.clock = pygame.time.Clock()
        self.step_ctr = 0

        # Screen definition
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption('Snake')
        
        # Generate snake and food
        self.snake = Snake()
        self.food = Food()
    
    def render(self):
        # Refill the screen
        self.screen.fill((0, 0, 0))
        # Render the snake	
        for i in range(0, self.snake.len):
            self.screen.blit(
                self.snake.img, (
                    self.snake.x[i], self.snake.y[i]
                )
            )
        # Render the food
        self.screen.blit(self.food.img, self.food.pos)
        # Render the score
        t = self.f.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(t, (10, 10))

        pygame.display.update()
    
    def step(self, act):
        self.reward = 0
        died = False
        ate = False
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
        
        # Evaluate move
        died = self.selfEat()
        ate = self.foodEat()
        if not died:
            died = self.hitBorder()
        
        # Set Reward
        self.reward = self.agent.setReward(died, ate)

        # If die end the game
        if died:
            self.stop = die(self.score)

        # Perform the move
        if not self.stop:
            self.snake.move()
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
        self.step(action)
        # Get state 2
        state2 = self.agent.getState(self.snake, self.food)
        # Learn from experience
        experience = (state1, action, self.reward, state2)
        self.agent.memory.push(experience)
        self.agent.memory.replay(self.stop)

        while True:
            if self.stop:
                break
            self.step_ctr+=1
            # Get state 1
            state1 = self.agent.getState(self.snake, self.food)
            # Exploration Rate and choose action
            if random.random() < self.agent.getEpsilon(self.step_ctr):
            #if random.randint(0,1) < self.agent.getEpsilon(self.step_ctr):
                # Explore
                action = random.randint(0,3)
            else:
                # Exploit
                action = self.agent.memory.exploit(state1)
            # Perform action
            self.step(action)
            # Get state 2            
            state2 = self.agent.getState(self.snake, self.food) # Get state 2
            
            # Manage memory
            experience = (state1, action, self.reward, state2)
            self.agent.memory.push(experience)
            self.agent.memory.replay(self.stop)
            
    def selfEat(self):
        # Snake eats itself
        i = self.snake.len
        while i >= 2:   
            collided = collide(
                self.snake.x[0], self.snake.x[i], 
                self.snake.y[0], self.snake.y[i], 
                20, 20, 20, 20
            )
            if collided: return True
            i-= 1
        
        return False
    
    def foodEat(self):
        # Snake eats apple
        collided = collide(
            self.snake.x[0], self.food.pos[0],
            self.snake.y[0], self.food.pos[1],
            20, 20, 20, 20
        )
        if collided:
            self.score+=1
            self.snake.x.append(700)
            self.snake.y.append(700)
            self.snake.len += 1
            randx = random.randint(0, 560)
            randy = random.randint(0, 560)
            self.food.pos = (randx-randx%20, randy-randy%20)

            return True
    
    def hitBorder(self):
        # Snake reach the border
        if self.snake.x[0] < 0 or self.snake.x[0] > 580 \
        or self.snake.y[0] < 0 or self.snake.y[0] > 580: 
            return True
        else:
            return False

class Snake():
    def __init__(self):
        # Snake coordinates array
        self.x = [280, 280, 280, 280, 280]
        self.y = [280, 260, 240, 220, 200]

        self.dir = 0 # Initial direction

        # Draw snake
        self.img = pygame.Surface((20, 20))
        self.img.fill((255, 255, 255))

        # Snake length
        self.len = len(self.x)-1
        
    def move(self):
        # Snake block flooding
        i = self.len
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
        randx = random.randint(0, 560)
        randy = random.randint(0, 560)
        self.pos = (randx-randx%20, randy-randy%20)
        
        # Draw apple
        self.img = pygame.Surface((20, 20))
        self.img.fill((255, 0, 0))


agent = Agent()
for episode in range(100):
    print(f'Episode {episode}')
    pygame.init()
    env = Environment(agent)
    env.run()
    del env
    pygame.quit()

    