from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent, set_seed
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_EPS = 10 # number of games to play against random agent
ROUNDS = 10
SEED = 42

set_seed(SEED)

def eval_agent(rl_agent):
    """
    evaluate the current agent's performance against a random agent
    when agent is playing as either player 1 or player 2
    """

    p1_w_l_d = [0, 0, 0]
    p2_w_l_d = [0, 0, 0]

    """RL agent playing as player 1 against random agent"""
    env = tictactoe.env()

    agents = {
        'player_1': rl_agent,
        'player_2': 'random'
    }

    for ep in range(EVAL_EPS):
        env.reset(seed=SEED)
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

            # when the game is terminated 
            # and two players are still present in environment's
            # logging dictionary
            if len(env.terminations.keys()) == 2:
                if env.terminations[a]:

                    # log RL agent's win loss or draw
                    if env.rewards['player_1'] == 1:
                        p1_w_l_d[0] += 1
                    elif env.rewards['player_1'] == -1:
                        p1_w_l_d[1] += 1
                    else:
                        p1_w_l_d[2] += 1
    env.close()


    """RL agent playing as player 2 against random agent"""
    env = tictactoe.env()

    agents = {
        'player_1': 'random',
        'player_2': rl_agent
    }

    for ep in range(EVAL_EPS):
        env.reset(seed=SEED)
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

            # when the game is terminated 
            # and two players are still present in environment's
            # logging dictionary
            if len(env.terminations.keys()) == 2:
                if env.terminations[a]:

                    # log RL agent's win loss or draw
                    if env.rewards['player_2'] == 1:
                        p2_w_l_d[0] += 1
                    elif env.rewards['player_2'] == -1:
                        p2_w_l_d[1] += 1
                    else:
                        p2_w_l_d[2] += 1
    env.close()

    # calculate win percentages
    p1_w_perc = p1_w_l_d[0] * 100 / sum(p1_w_l_d)
    p2_w_perc = p2_w_l_d[0] * 100 / sum(p2_w_l_d)
    p1_l_perc = p1_w_l_d[1] * 100 / sum(p1_w_l_d)
    p2_l_perc = p2_w_l_d[1] * 100 / sum(p2_w_l_d)
    p1_d_perc = p1_w_l_d[2] * 100 / sum(p1_w_l_d)
    p2_d_perc = p2_w_l_d[2] * 100 / sum(p2_w_l_d)

    return p1_w_perc, p1_l_perc, p1_d_perc, p2_w_perc, p2_l_perc, p2_d_perc

# initialize game environment
# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

# load model
# initialize MuZero agent with config
config = {
    'batch_size': 128,
    'buffer_size': 10_000,
    'min_replay_size': 1500,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_iters': 50,
    'train_iters': 2,
    'gamma': 1.0,
    'k_unroll_steps': 5,
    'temperature': 1.0,
    'temp_schedule': [(0.6*10_000, 1.0), (0.9*10_000, 0.5), (10**9, 0.25)],
    'dirichlet_alpha': 1.0,
    'root_exploration_fraction': 0.25
}

load_dir = os.path.join('results','_0')
agent1 = MuZeroAgent(environment=env, config=config, load=True, load_dir=load_dir)

data = {
    'p1_w_perc':[],
    'p1_l_perc':[],
    'p1_d_perc':[],
    'p2_w_perc':[],
    'p2_l_perc':[],
    'p2_d_perc':[],
}

for i in range(ROUNDS):
    p1_w_perc, p1_l_perc, p1_d_perc, p2_w_perc, p2_l_perc, p2_d_perc = eval_agent(agent1)

    # record win-loss-draw percentages
    data['p1_w_perc'].append(p1_w_perc)
    data['p1_l_perc'].append(p1_l_perc)
    data['p1_d_perc'].append(p1_d_perc)
    data['p2_w_perc'].append(p1_w_perc)
    data['p2_l_perc'].append(p1_l_perc)
    data['p2_d_perc'].append(p1_d_perc)

# display histogram
df = pd.DataFrame(data=data)
df['p1_w_perc'].hist(bins=np.arange(50,101,1))
df['p2_w_perc'].hist(bins=np.arange(50,101,1))
plt.show()
