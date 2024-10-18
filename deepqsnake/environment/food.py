import pygame
import random

class Food():
    """Generate the food object and its random position in the screen
    
    Parameters:
        screen_width (int): the width of the game screen
        screen_height (int): the height of the game screen

    Attributes:
        screen_width (int): the width of the game screen
        screen_height (int): the height of the game screen
        pos (tuple): x and y coordinates of the food
        img (pygame.Surface): food object to be rendered

    Methods:
        gen_pos(): Generate the random position of the food in a fixed grid.
    """

    def __init__(self, screen_width:int, screen_height:int):
        # Screen size
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Random apple position
        self.pos = self.gen_pos()
        
        # Draw apple
        self.img = pygame.Surface((20, 20))
        self.img.fill((163, 51, 51))

    def gen_pos(self):
        """Generate the random position of the food in a predetermined grid.

        """
        randx = random.randint(20, self.screen_width-40)
        randy = random.randint(20, self.screen_height-40)
        if randx-randx % 20 < 20:
            randx = 10
        elif randx-randx % 20 > self.screen_width-40:
            randx = self.screen_width-40
        else:
            randx = randx-randx % 20

        if randy-randy % 20 < 20:
            randy = 10
        elif randy-randy % 20 > self.screen_height-40:
            randy = self.screen_height-40
        else:
            randy = randy-randy % 20

        return (randx, randy)
