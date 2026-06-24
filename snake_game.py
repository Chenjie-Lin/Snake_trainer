import pygame
import random
import numpy as np
from collections import deque
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

BLOCK = 20  # each grid cell is 20px

class SnakeGame:

    # Initialize the game
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        pygame.init()
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    # Resets the board
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = [self.w // 2, self.h // 2]
        self.snake = [
            self.head[:],
            [self.head[0] - BLOCK, self.head[1]],
            [self.head[0] - 2*BLOCK, self.head[1]],
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Place a food randomly on the board
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    # Updates the game based on action
    def step(self, action):
        # action: [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
        self.frame_iteration += 1

        # Checks to see if the user decides to close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update the Snake
        self._move(action)
        self.snake.insert(0, self.head[:])

        reward = 0
        game_over = False

        # Checks to see if the Snake dies
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Checks to see if the snake ate the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._render()
        self.clock.tick(40)  # speed — increase to train faster
        return reward, game_over, self.score

    # Checks to see if the snake collides with the wall or its body
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] >= self.w or pt[0] < 0 or pt[1] >= self.h or pt[1] < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    # Updates the position of the snake
    def _move(self, action):
        # clockwise directions
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Updates direction based on action
        if action == [1, 0, 0]:    # straight
            self.direction = clock_wise[idx]
        elif action == [0, 1, 0]:  # right turn
            self.direction = clock_wise[(idx + 1) % 4]
        else:                       # left turn
            self.direction = clock_wise[(idx - 1) % 4]

        # Updates the snake head based on the direction
        x, y = self.head
        if self.direction == Direction.RIGHT: x += BLOCK
        elif self.direction == Direction.LEFT: x -= BLOCK
        elif self.direction == Direction.DOWN: y += BLOCK
        elif self.direction == Direction.UP:  y -= BLOCK
        self.head = [x, y]

    # Renders the snake, score, and food
    def _render(self):
        self.display.fill((15, 15, 15))
        for i, pt in enumerate(self.snake):
            color = (0, 200, 100) if i == 0 else (0, 150, 70)
            pygame.draw.rect(self.display, color, (*pt, BLOCK-2, BLOCK-2))
        pygame.draw.rect(self.display, (220, 60, 60), (*self.food, BLOCK-2, BLOCK-2))
        font = pygame.font.SysFont("monospace", 16)
        self.display.blit(font.render(f"Score: {self.score}", True, (200,200,200)), (4, 4))
        pygame.display.flip()

    # Get the state of the game
    def get_state(self):
        if not self.snake:
            return None
        head = self.snake[0] # coordinate for the snake head

        # Checks to see which direction the snake is facing
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Coordinates for the blocks adjacent to the head
        point_l = [head[0] - BLOCK, head[1]]
        point_r = [head[0] + BLOCK, head[1]]
        point_u = [head[0], head[1] - BLOCK]
        point_d = [head[0], head[1] + BLOCK]

        # Checks to see if theres collision for each relative direction
        danger_straight =   (dir_r and self._is_collision(point_r)) or \
                            (dir_l and self._is_collision(point_l)) or \
                            (dir_u and self._is_collision(point_u)) or \
                            (dir_d and self._is_collision(point_d))
        
        danger_right =  (dir_r and self._is_collision(point_d)) or \
                        (dir_l and self._is_collision(point_u)) or \
                        (dir_u and self._is_collision(point_r)) or \
                        (dir_d and self._is_collision(point_l))
        
        danger_left =   (dir_r and self._is_collision(point_u)) or \
                        (dir_l and self._is_collision(point_d)) or \
                        (dir_u and self._is_collision(point_l)) or \
                        (dir_d and self._is_collision(point_r))
        
        # Checks to see if theres food in any of the directions
        food_left = (self.food[0] < head[0])
        food_right = (self.food[0] > head[0])
        food_up = (self.food[1] < head[1])
        food_down = (self.food[1] > head[1])

        # Return all states
        return np.array([
            danger_straight, danger_right, danger_left,
            dir_r, dir_l, dir_u, dir_d,
            food_left, food_right, food_up, food_down
        ], dtype=int)