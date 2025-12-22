from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

config = {
    'batch_size': 8,
    'buffer_size': 10,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'max_iters': 10,
    'gamma': 0.997,
    'k_unroll_steps': 5
}

agent1 = MuZeroAgent(environment=env, config=config)

agent_list = [agent1 , 'random']


for ep in range(5):
    print(f'EP={ep}')
    idx = 0
    p1_reward, p2_reward = 0, 0
    recorded = False
    env.reset(seed=42)
    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if not recorded:
                p1_reward, p2_reward = env.rewards['player_1'], env.rewards['player_2']
                recorded = True
                if a == 'player_1':
                    agent1.experience(observation, a, action, 
                                    p1_reward, p1_reward, termination)
                else:
                    agent1.experience(observation, a, action, 
                                    p2_reward, p2_reward, termination)
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
        if not recorded:
            if a == 'player_1':
                agent1.experience(observation, a, action, 
                                p1_reward, p1_reward, termination)
            else:
                agent1.experience(observation, a, action, 
                                p2_reward, p2_reward, termination)
        idx = (idx+1) % 2
    agent1.update()
    # pause = input('\npress enter for new game')
env.close()
