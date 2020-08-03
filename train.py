from agent import Agent
from stats import Statistics
from snake import Environment
import pygame

agent = Agent()
stat = Statistics()
episode = 0
EP = 500

while True:
    if episode > EP: break
    episode += 1
    print(f'Episode:{episode}')
    pygame.init()
    env = Environment(stat, episode, agent, True, True)
    env.run()
    env.agent.model.save_weights('weights.hdf5')
    del env
    pygame.quit()



"""
for j in range(2):
    ##print(f'Training {j}')
    pygame.init()
    env = Environment(stat, episode, agent, False)
    env.run()
    del env
    pygame.quit()
"""