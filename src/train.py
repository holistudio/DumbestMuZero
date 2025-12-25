from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

import datetime
import csv
import os

TRAIN_EPS = 1000
EVAL_EPS = 100

def eval_agent(rl_agent, train_ep):
    p1_w_l_d = [0, 0, 0]
    p2_w_l_d = [0, 0, 0]

    env = tictactoe.env()
    agents = {
        'player_1': rl_agent,
        'player_2': 'random'
    }

    for ep in range(EVAL_EPS):
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
            if len(env.terminations.keys()) == 2:
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

    for ep in range(EVAL_EPS):
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
            if len(env.terminations.keys()) == 2:
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

    print(f'EP={train_ep} Agent Performance, as P1: {p1_w_perc:.2f}%, {p1_w_l_d}, as P2: {p2_w_perc:.2f}%, {p2_w_l_d}')
    
    csv_filename = 'agent_performance.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['train_ep', 'p1_win', 'p1_loss', 'p1_draw', 'p2_win', 'p2_loss', 'p2_draw'])
        writer.writerow([train_ep] + p1_w_l_d + p2_w_l_d)

# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

config = {
    'batch_size': 128,
    'buffer_size': 500,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'max_iters': 80,
    'train_iters': 100,
    'gamma': 1.0, # 0.997,
    'k_unroll_steps': 5,
    'temperature': 1.0,
}

agent1 = MuZeroAgent(environment=env, config=config)

agents = {
    'player_1': agent1,
    'player_2': agent1
}

start_time = datetime.datetime.now()
for ep in range(TRAIN_EPS):
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

        # print(a, len(env.terminations.keys()), env.terminations, env.rewards)
        if len(env.terminations.keys()) == 2:
            agent.experience(observation, a, action, env.rewards[a], env.terminations[a])

    agent1.update()
    print(f'{datetime.datetime.now()-start_time} EP={ep}')
    if ((ep+1) % 10 == 0) or ep+1 == TRAIN_EPS: 
        eval_agent(agent1, ep)
    # pause = input('\npress enter for new game')
env.close()
