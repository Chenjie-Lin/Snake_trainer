from snake_game import SnakeGame
from agent import Agent

agent = Agent()
agent.n_games = 200
game = SnakeGame()

state = game.get_state()
while True:
    action = agent.get_action(state)
    reward, done, score = game.step(action)
    new_state = game.get_state()
    state = new_state
    
    if done:
        print(f'Final Score: {score}')
        break