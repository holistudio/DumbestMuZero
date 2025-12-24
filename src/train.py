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
    'train_iters': 100,
    'gamma': 0.997,
    'k_unroll_steps': 5
}

agent1 = MuZeroAgent(environment=env, config=config)

agents = {
    'player_1': agent1,
    'player_2': agent1
}

start_time = datetime.datetime.now()
for ep in range(100):
    env.reset(seed=42)

    for a in env.agent_iter():
        agent = agents[a]
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
        agent.experience(observation, a, action, env.rewards[a], termination)

    agent1.update()
    if (ep+1) % 10 == 0:
        print(f'{datetime.datetime.now()-start_time} EP={ep}')
        eval(agent1,ep)
    # pause = input('\npress enter for new game')
env.close()

def eval(rl_agent, ep):
    p1_w_l_d = [0, 0, 0]
    p2_w_l_d = [0, 0, 0]

    env = tictactoe.env()
    agents = {
        'player_1': rl_agent,
        'player_2': 'random'
    }

    

    for ep in range(100):
        env.reset(seed=42)
        for a in env.agent_iter():
            agent = agents[a]
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None

            else:
                mask = observation["action_mask"]
                if agent == 'random':
                    action = int(env.action_space(a).sample(mask))
                else:
                    action = agent.act(observation)

            env.step(action)
            if env.terminations[a] == True:
                if env.rewards['player_1'] == 1:
                    p1_w_l_d[0] += 1
                elif env.rewards['player_1'] == -1:
                    p1_w_l_d[1] += 1
                else:
                    p1_w_l_d[2] += 1
    env.close()

    env = tictactoe.env()
    agents = {
        'player_1': 'random',
        'player_2': rl_agent
    }

    for ep in range(100):
        env.reset(seed=42)
        for a in env.agent_iter():
            agent = agents[a]
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None

            else:
                mask = observation["action_mask"]
                if agent == 'random':
                    action = int(env.action_space(a).sample(mask))
                else:
                    action = agent.act(observation)

            env.step(action)
            if env.terminations[a] == True:
                if env.rewards['player_2'] == 1:
                    p2_w_l_d[0] += 1
                elif env.rewards['player_2'] == -1:
                    p2_w_l_d[1] += 1
                else:
                    p2_w_l_d[2] += 1
    env.close()

    p1_w_perc = p1_w_l_d[0] * 100 / sum(p1_w_l_d)
    p2_w_perc = p2_w_l_d[0] * 100 / sum(p2_w_l_d)

    print(f'EP={ep} Agent Performance, as P1: {p1_w_perc:.2f}%, {p1_w_l_d}, as P2: {p2_w_perc:.2f}%, {p2_w_l_d}')
    pass