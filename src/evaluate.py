from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent, set_seed
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_EPS = 10 # number of games to play against random agent
ROUNDS = 10
SEED = 42
EPISODES_PER_ROUND = 2 * EVAL_EPS # each round plays as both player 1 and player 2

set_seed(SEED)

# elapsed seconds for every evaluation episode played so far, across all rounds
episode_times = []

def _record_episode_time(elapsed):
    """record an episode's duration and, on the very first episode, use it to
    project how long the whole ROUNDS loop will take"""
    episode_times.append(elapsed)
    if len(episode_times) == 1:
        estimated_round_time = elapsed * EPISODES_PER_ROUND
        estimated_total_time = estimated_round_time * ROUNDS
        print(f"First evaluation episode took {elapsed:.3f}s")
        print(f"  -> estimated time per round: {estimated_round_time:.2f}s")
        print(f"  -> estimated total time for {ROUNDS} rounds: {timedelta(seconds=estimated_total_time)}")

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
        ep_start = time.perf_counter()
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
        _record_episode_time(time.perf_counter() - ep_start)
    env.close()


    """RL agent playing as player 2 against random agent"""
    env = tictactoe.env()

    agents = {
        'player_1': 'random',
        'player_2': rl_agent
    }

    for ep in range(EVAL_EPS):
        ep_start = time.perf_counter()
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
        _record_episode_time(time.perf_counter() - ep_start)
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
    data['p2_w_perc'].append(p2_w_perc)
    data['p2_l_perc'].append(p2_l_perc)
    data['p2_d_perc'].append(p2_d_perc)

    # updated ETA based on the average episode time observed so far
    avg_episode_time = sum(episode_times) / len(episode_times)
    remaining_episodes = (ROUNDS - (i + 1)) * EPISODES_PER_ROUND
    remaining_time = avg_episode_time * remaining_episodes
    eta = datetime.now() + timedelta(seconds=remaining_time)
    print(f"Round {i + 1}/{ROUNDS} done | avg episode time: {avg_episode_time:.3f}s | "
          f"estimated remaining: {timedelta(seconds=remaining_time)} | ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

# display histograms: rows are player 1 / player 2, columns are win / loss / draw
df = pd.DataFrame(data=data)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

cols = [
    ('p1_w_perc', 'Player 1 Win %', np.arange(50, 101, 1), '#2ca02c'),
    ('p1_l_perc', 'Player 1 Loss %', np.arange(0, 31, 1), '#d62728'),
    ('p1_d_perc', 'Player 1 Draw %', np.arange(0, 21, 1), '#7f7f7f'),
    ('p2_w_perc', 'Player 2 Win %', np.arange(50, 101, 1), '#2ca02c'),
    ('p2_l_perc', 'Player 2 Loss %', np.arange(0, 31, 1), '#d62728'),
    ('p2_d_perc', 'Player 2 Draw %', np.arange(0, 21, 1), '#7f7f7f'),
]

for ax, (col, title, bins, color) in zip(axes.flat, cols):
    df[col].hist(bins=bins, ax=ax, color=color)
    ax.set_title(title)

fig.tight_layout()
plt.show()
