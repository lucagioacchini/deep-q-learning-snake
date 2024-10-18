import pygame
from src.agent import Agent
from src.stats.stats import Statistics
from src.environment import SnakeEnvironment

SCREEN_WEIGHT = 620
SCREEN_HEIGHT = 620

# Initialize the agent
agent = Agent(
    screen_width=SCREEN_WEIGHT, 
    screen_height=SCREEN_HEIGHT, 
    memory_capacity=1E6, 
    memory_batch_size=5E3, 
    eps_decay=.03, 
    gamma=.9
)
agent.load_weights('weights/weights.weights.h5') # Load the pre-trained weights

# Initialize the statistics for plotting
stat = Statistics()

# Initialize a new game
pygame.init()

# Initialize the environment and run
env = SnakeEnvironment(
        screen_width=SCREEN_WEIGHT, 
        screen_height=SCREEN_HEIGHT, 
        stat = Statistics(), 
        episode=0, 
        agent=agent, 
        train=False, 
        display=True
    )
env.run()

# Ending
del env
pygame.quit()