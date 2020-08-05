from lib.agent import Agent
from lib.stats import Statistics
from lib.snake import Environment
import pygame
from lib.config import EPISODE

"""
Train the agent and neural network for a number of episodes.
"""

agent = Agent()
stat = Statistics()
episode = 0

while True:
    if episode > EPISODE: break
    episode += 1
    print(f'Episode:{episode}')
    # pylint: disable=no-member
    pygame.init()
    env = Environment(stat, episode, agent, True, False)
    env.run()
    env.agent.model.save_weights('weights/weights.hdf5')
    del env
    pygame.quit()