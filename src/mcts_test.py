from envs.tictactoe import tictactoe
from agents.mcts.mcts import UCTAgent

env = tictactoe.env(render_mode="human")


agent1 = UCTAgent(environment=env)

agent_list = [agent1 , agent1]


for _ in range(5):
    idx = 0
    env.reset(seed=42)
    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if a == 'player_1':
                print('\nPLAYER X TURN')
            else:
                print('\nPLAYER O TURN')
            mask = observation["action_mask"]
            if agent == 'random':
                action = env.action_space(a).sample(mask)
            else:
                action = agent.step(observation)

        env.step(action)
        idx = (idx+1) % 2
    pause = input('\npress enter for new game')
env.close()
