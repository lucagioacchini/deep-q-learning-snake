from agent import Agent
from stats import Statistics
from snake import Environment
import pygame

agent = Agent()
stat = Statistics()
pygame.init()
env = Environment(stat, 0, agent, False, True)
env.agent.model.load_weights('weights.hdf5')
env.run()
del env
pygame.quit()