from envs.tictactoe import tictactoe
from agents.mcts.mcts import UCTAgent

env = tictactoe.env(render_mode="human")


agent1 = UCTAgent(environment=env)
agent2 = UCTAgent(environment=env, p1=False)
agent_list = [agent1, agent2]


for _ in range(5):
    idx = 0
    env.reset(seed=42)
    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            if agent == 'random':
                action = env.action_space(a).sample(mask)
            else:
                action = agent.step(observation)

        env.step(action)
        idx = (idx+1) % 2
    pause = input('press enter for new game')
env.close()
