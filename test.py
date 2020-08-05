from lib.agent import Agent
from lib.stats import Statistics
from lib.snake import Environment
import pygame

"""
Test the Snake game exploiting the trained neural network
"""

agent = Agent()
stat = Statistics()
# pylint: disable=no-member
pygame.init()
env = Environment(stat, 0, agent, False, True)
env.agent.model.load_weights('weights/weights.hdf5')
env.run()
del env
pygame.quit()