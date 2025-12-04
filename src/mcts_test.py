from envs.tictactoe import tictactoe
from agents.mcts.mcts import UCTAgent

env = tictactoe.env(render_mode="human")
env.reset(seed=42)

agent1 = UCTAgent(environment=env)
agent_list = [agent1, agent1]

idx = 0

for a in env.agent_iter():
    agent = agent_list[idx]
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        if idx == 0:
            # action = env.action_space(a).sample(mask)
            action = agent.step(observation)
        else:
            action = env.action_space(a).sample(mask)
            # action = agent.step(observation)

    env.step(action)
    idx += 1
    if idx >= len(agent_list):
        idx = 0
pause = input('press enter')
env.close()
