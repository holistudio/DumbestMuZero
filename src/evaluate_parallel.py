"""parallel version of evaluate.py

each round is independent, so rounds are farmed out to a pool of worker
processes. MCTS here is a long chain of batch-1 forward passes through tiny
MLPs, which is latency bound rather than throughput bound, so workers run on
CPU with a single torch thread each -- that is faster per call than the GPU and
lets us use all the cores.
"""

# must happen before torch is imported anywhere, so workers never touch CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# env var overrides make it easy to do a quick smoke run without editing the file
EVAL_EPS = 100 # number of games to play against random agent
ROUNDS = 100
SEED = 42
WORKERS = 20  # 24 physical cores; leave a few for the OS
EPISODES_PER_ROUND = 2 * EVAL_EPS # each round plays as both player 1 and player 2

LOAD_DIR = os.path.join('results', '_3')

CONFIG = {
    'batch_size': 128,
    'buffer_size': 3000,
    'min_replay_size': 1500,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_iters': 200,
    'train_iters': 5,
    'gamma': 1.0,
    'k_unroll_steps': 5,
    'temperature': 1.0,
    'temp_schedule': [(0.6*10_000, 1.0), (0.9*10_000, 0.5), (10**9, 0.25)],
    'dirichlet_alpha': 1.0,
    'root_exploration_fraction': 0.4
}

# one agent per worker process, built lazily and reused across that worker's rounds
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        import torch
        from envs.tictactoe import tictactoe
        from agents.muzero.muzero import MuZeroAgent

        # one thread per process: the nets are far too small for intra-op
        # threading to help, and oversubscription would thrash the cores
        torch.set_num_threads(1)

        env = tictactoe.env()
        _agent = MuZeroAgent(environment=env, config=CONFIG, load=True, load_dir=LOAD_DIR)
    return _agent


def _play_games(rl_agent, rl_is_player_1, seed):
    """play EVAL_EPS games against a random agent, return [wins, losses, draws]"""
    from envs.tictactoe import tictactoe

    w_l_d = [0, 0, 0]
    rl_label = 'player_1' if rl_is_player_1 else 'player_2'

    env = tictactoe.env()
    agents = {
        'player_1': rl_agent if rl_is_player_1 else 'random',
        'player_2': 'random' if rl_is_player_1 else rl_agent,
    }

    for ep in range(EVAL_EPS):
        # env.reset(seed=seed + ep)
        env.reset(seed=seed)
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
                    if env.rewards[rl_label] == 1:
                        w_l_d[0] += 1
                    elif env.rewards[rl_label] == -1:
                        w_l_d[1] += 1
                    else:
                        w_l_d[2] += 1
    env.close()
    return w_l_d


def run_round(round_idx):
    """
    evaluate the current agent's performance against a random agent
    when agent is playing as either player 1 or player 2
    """
    from agents.muzero.muzero import set_seed

    # each round gets its own RNG stream so parallel rounds stay independent
    # round_seed = SEED + round_idx * 10_000
    round_seed = SEED
    set_seed(round_seed)

    agent = _get_agent()

    start = time.perf_counter()
    p1_w_l_d = _play_games(agent, True, round_seed)
    p2_w_l_d = _play_games(agent, False, round_seed)
    elapsed = time.perf_counter() - start

    # calculate win percentages
    p1 = [c * 100 / sum(p1_w_l_d) for c in p1_w_l_d]
    p2 = [c * 100 / sum(p2_w_l_d) for c in p2_w_l_d]

    return round_idx, (p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]), elapsed


def main():
    data = {
        'p1_w_perc': [None] * ROUNDS,
        'p1_l_perc': [None] * ROUNDS,
        'p1_d_perc': [None] * ROUNDS,
        'p2_w_perc': [None] * ROUNDS,
        'p2_l_perc': [None] * ROUNDS,
        'p2_d_perc': [None] * ROUNDS,
    }
    keys = list(data.keys())

    print(f"\nRunning {ROUNDS} rounds across {WORKERS} worker processes\n")
    wall_start = time.perf_counter()
    round_times = []

    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_round, i) for i in range(ROUNDS)]

        for done, future in enumerate(as_completed(futures), start=1):
            idx, percs, elapsed = future.result()
            for key, value in zip(keys, percs):
                data[key][idx] = value
            round_times.append(elapsed)

            # ETA from observed wall-clock throughput, which already accounts
            # for the pool running WORKERS rounds concurrently
            wall_elapsed = time.perf_counter() - wall_start
            remaining_time = wall_elapsed / done * (ROUNDS - done)
            eta = datetime.now() + timedelta(seconds=remaining_time)
            print(f"Round {done}/{ROUNDS} done (round {idx}) | "
                  f"avg round time: {sum(round_times) / len(round_times):.2f}s | "
                  f"estimated remaining: {timedelta(seconds=remaining_time)} | "
                  f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            print(*percs)

    print(f"\nTotal wall time: {timedelta(seconds=time.perf_counter() - wall_start)}")

    # display histograms: rows are player 1 / player 2, columns are win / loss / draw
    df = pd.DataFrame(data=data)
    eval_data_file = os.path.join('agents', 'muzero', LOAD_DIR, 'eval_results.json')
    with open(eval_data_file, 'w') as f:
        json.dump(data, f, indent=4)

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


if __name__ == '__main__':
    main()
