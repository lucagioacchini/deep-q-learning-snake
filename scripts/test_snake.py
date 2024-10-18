import pygame
from deepqsnake.agent import Agent
from deepqsnake.stats.stats import Statistics
from deepqsnake.environment import SnakeEnvironment

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
# Load the pre-trained weights
agent.load_weights('weights/weights.weights.h5')

# Initialize the statistics for plotting
stat = Statistics()

# Initialize a new game
pygame.init()

# Initialize the environment and run
env = SnakeEnvironment(
    screen_width=SCREEN_WEIGHT,
    screen_height=SCREEN_HEIGHT,
    stat=Statistics(),
    episode=0,
    agent=agent,
    train=False,
    display=True
)
env.run()

# Ending
del env
pygame.quit()
