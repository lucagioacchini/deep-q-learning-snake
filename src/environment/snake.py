import pygame

class Snake():
    """Generate the snake object and design the movements.

    Attributes:
        x (list): x coordinates of the snake blocks
        y (list): y coordinates of the snake blocks
        dir (int): direction taken by the snake
        ate (bool): true if the snake ate the food
        crashed (bool): true if the snake crashed with the borders
        img (pygame.Surface): snake blocks to be rendered
        bord1 (pygame.Surface): snake blocks to be rendered
        bord2 (pygame.Surface): snake blocks to be rendered
        len (int): length of the snake

    Methods:
        move(): Update the snake position according to the movement and direction.
    """

    def __init__(self):
        # Snake coordinates array
        self.x = [180, 180, 180, 180, 180, 180]
        self.y = [180, 160, 158, 156, 154, 152]

        self.dir = 0  # Initial direction

        self.ate = False
        self.crashed = False

        # Draw snake
        # pylint: disable=too-many-function-args
        self.img = pygame.Surface((20, 20))
        self.img.fill((255, 255, 255))
        self.bord1 = pygame.Surface((1, 20))
        self.bord1.fill((0, 0, 0))
        self.bord2 = pygame.Surface((20, 1))
        self.bord2.fill((0, 0, 0))

        # Snake length
        self.len = len(self.x)

    def move(self):
        """Update the snake position according to the movement and direction.

        """
        # Snake block flooding
        i = self.len-1
        while i >= 1:
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]
            i -= 1

        # Snake follows direction
        if self.dir == 0:
            self.y[0] += 20
        elif self.dir == 1:
            self.x[0] += 20
        elif self.dir == 2:
            self.y[0] -= 20
        elif self.dir == 3:
            self.x[0] -= 20