import pygame
from deepqsnake.agent import Agent
from deepqsnake.stats import Statistics
from deepqsnake.environment import SnakeEnvironment

EPISODES = 1000  # Training episodes
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 320

# Initialize the agent
agent = Agent(
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT,
    memory_capacity=1E6,
    memory_batch_size=5E3,
    eps_decay=.03,
    gamma=.9
)

# Start the training
episode = 0
while episode <= EPISODES:
    print(f'Episode:{episode}')

    # Initialize a new game
    pygame.init()

    # Initialize a new environment and run
    env = SnakeEnvironment(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        stat=Statistics(),
        episode=episode,
        agent=agent,
        train=True,
        display=True
    )
    env.run()

    # Save the trained weights
    agent.save_weights('weights/weights.weights.h5')

    # End current game
    del env
    pygame.quit()
    episode += 1
