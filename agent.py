import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame
from model import LinearQNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.model.load()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
def train():
    scores = []
    mean_scores = []
    total_score = 0
    agent = Agent()
    game = SnakeGame()
    high_score = 0

    while True:
        state = game.get_state()
        action = agent.get_action(state)
        reward, done, score = game.step(action)
        new_state = game.get_state()

        agent.train_short_memory(state, action, reward, new_state, done)
        agent.remember(state, action, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score
                agent.model.save()

            print(f'Game: {agent.n_games}  Score: {score}  High Score: {high_score}')

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()
