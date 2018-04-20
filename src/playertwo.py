import cv2
import numpy as np
import pygame

class PlayerTwo(object):
    def __init__(self, args):
        self.screen_size = args.pygame_width, args.pygame_height
        self.surface = pygame.display.set_mode(self.screen_size, pygame.DOUBLEBUF)

    def get_action(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                return -1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return -1

        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 2 if not keys[pygame.K_SPACE] else 4
        elif keys[pygame.K_DOWN]:
            action = 3 if not keys[pygame.K_SPACE] else 5
        elif keys[pygame.K_SPACE]:
            action = 1
        return action

    def render_screen(self, screen):
        screen = cv2.resize(screen, self.screen_size, interpolation=cv2.INTER_NEAREST)
        screen = np.transpose(screen, (1,0,2))
        pygame.pixelcopy.array_to_surface(self.surface, screen)
        pygame.display.flip()
