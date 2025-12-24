from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

import datetime

# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

config = {
    'batch_size': 8,
    'buffer_size': 100,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'max_iters': 1000,
    'gamma': 0.997,
    'k_unroll_steps': 5
}

agent1 = MuZeroAgent(environment=env, config=config)

agent_list = [agent1 , agent1]


start_time = datetime.datetime.now()
for ep in range(100):
    idx = 0
    env.reset(seed=42)

    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # if a == 'player_1':
            #     print('\nPLAYER X TURN')
            # else:
            #     print('\nPLAYER O TURN')
            mask = observation["action_mask"]
            if agent == 'random':
                action = int(env.action_space(a).sample(mask))
            else:
                action = agent.step(observation)

        env.step(action)
        idx = (idx+1) % 2
        
    agent1.update()
    print(f'{datetime.datetime.now()-start_time} EP={ep}')
    # pause = input('\npress enter for new game')
env.close()