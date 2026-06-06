from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

import datetime
import csv
import json
import os
import random

import numpy as np
import torch

TRAIN_EPS = int(os.getenv("TRAIN_EPS", "100000"))
EVAL_EPS = int(os.getenv("EVAL_EPS", "100"))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "500"))
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "5"))
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "10"))
SEED = int(os.getenv("SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

"""TRAINING HELPER FUNCTIONS"""
def preprocess_obs(observation):
    """
    pre-process observation dictionary into a string for logging purposes
    """
    
    obs = np.zeros((3,3))

    # get current and opponent board planes
    # environment alternates board planes based on whose turn it is
    current_player_plane = np.array(observation["observation"][:, :, 0])
    opponent_plane = np.array(observation["observation"][:, :, 1])

    total_pieces = np.sum(current_player_plane) + np.sum(opponent_plane)

    # determine if it is player 1's (X) or player 2's (O) turn
    # if total pieces is even, it's player 1's turn (current player is p1)
    # if total pieces is odd, it's player 2's turn (current player is p2)
    if total_pieces % 2 == 0:
        p1_plane, p2_plane = current_player_plane, opponent_plane
    else:
        p2_plane, p1_plane = current_player_plane, opponent_plane

    # X pieces = 1
    # O pieces = 2
    # blank board spaces = 0
    p2_plane = p2_plane * 2
    obs = obs + p1_plane + p2_plane
    obs = obs.flatten()

    # convert board state into a string
    obs_str = ''
    for o in obs:
        obs_str += str(int(o))
    return obs_str

def eval_agent(rl_agent, train_ep):
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
        eval_seed = SEED + 100_000 + ep
        env.reset(seed=eval_seed)
        env.action_space('player_2').seed(eval_seed)
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
                if env.terminations[a] == True:

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
        eval_seed = SEED + 200_000 + ep
        env.reset(seed=eval_seed)
        env.action_space('player_1').seed(eval_seed)
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
                if env.terminations[a] == True:

                    # log RL agent's win loss or draw
                    if env.rewards['player_2'] == 1:
                        p2_w_l_d[0] += 1
                    elif env.rewards['player_2'] == -1:
                        p2_w_l_d[1] += 1
                    else:
                        p2_w_l_d[2] += 1
    env.close()

    total_games = sum(p1_w_l_d) + sum(p2_w_l_d)
    eval_loss = (p1_w_l_d[1] + p2_w_l_d[1]) / total_games
    eval_win_rate = (p1_w_l_d[0] + p2_w_l_d[0]) / total_games
    p1_loss_rate = p1_w_l_d[1] / sum(p1_w_l_d)
    p2_loss_rate = p2_w_l_d[1] / sum(p2_w_l_d)

    # display performance in terminal
    print(
        f"EVAL  episode={train_ep + 1}/{TRAIN_EPS} "
        f"loss_rate={eval_loss:.3f} win_rate={eval_win_rate:.3f} "
        f"P1_loss={p1_loss_rate:.3f} P1_WLD={p1_w_l_d} "
        f"P2_loss={p2_loss_rate:.3f} P2_WLD={p2_w_l_d}"
    )
    
    # log in CSV file
    csv_filename = 'agent_performance.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'train_ep',
                'p1_win', 'p1_loss', 'p1_draw',
                'p2_win', 'p2_loss', 'p2_draw',
            ])
        writer.writerow([train_ep] + p1_w_l_d + p2_w_l_d)
    return {
        "loss_rate": eval_loss,
        "win_rate": eval_win_rate,
        "p1_loss_rate": p1_loss_rate,
        "p2_loss_rate": p2_loss_rate,
        "p1_wld": p1_w_l_d,
        "p2_wld": p2_w_l_d,
    }


def format_duration(duration):
    total_seconds = max(0, int(duration.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_message(ep, elapsed, agent, config):
    completed = ep + 1
    progress = completed / TRAIN_EPS
    eta = elapsed * ((1 / progress) - 1)
    replay_size = len(agent.replay_buffer.buffer)
    warmup = min(replay_size / agent.min_replay_size, 1.0)
    loss_text = f"{agent.last_loss:.6f}" if agent.last_loss is not None else "waiting"
    return (
        f"TRAIN episode={completed}/{TRAIN_EPS} ({progress:6.2%}) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"replay={replay_size}/{config['buffer_size']} warmup={warmup:6.2%} "
        f"updates={agent.training_steps} train_loss={loss_text}"
    )

"""SELF-PLAY TRAINING"""

# initialize game environment
# env = tictactoe.env(render_mode="human")
env = tictactoe.env()
for index, agent_name in enumerate(env.possible_agents):
    env.action_space(agent_name).seed(SEED + index)

# initialize MuZero agent with config
config = {
    'batch_size': int(os.getenv("BATCH_SIZE", "128")),
    'buffer_size': int(os.getenv("BUFFER_SIZE", "20000")),
    'min_replay_size': int(os.getenv("MIN_REPLAY_SIZE", "100")),
    'state_size': 16,
    'hidden_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_iters': int(os.getenv("MAX_ITERS", "50")),
    'train_iters': int(os.getenv("TRAIN_ITERS", "2")),
    'checkpoint_interval': int(os.getenv("CHECKPOINT_INTERVAL", "100")),
    'gamma': 1.0,
    'k_unroll_steps': 5,
    'temperature': 1.0,
    # AlphaZero-style temperature schedule keyed on completed self-play games,
    # scaled to a 100k-episode run: explore at T=1 for the first 60%, then
    # sharpen (0.5) through 90%, then near-greedy (0.25) for the final 10% so
    # the policy crystallizes toward minimax as it nears convergence.
    'temp_schedule': [(60000, 1.0), (90000, 0.5), (10**9, 0.25)],
    'dirichlet_alpha': float(os.getenv("DIRICHLET_ALPHA", "0.1")),
    # categorical (distributional) value/reward representation
    'num_bins': int(os.getenv("NUM_BINS", "51")),
    'support_limit': float(os.getenv("SUPPORT_LIMIT", "1.0")),
    'value_transform': os.getenv("VALUE_TRANSFORM", "1") == "1",
    'value_loss_weight': float(os.getenv("VALUE_LOSS_WEIGHT", "0.25")),
    # Prioritized Experience Replay: sample positions ~ |value error|^alpha,
    # correcting the bias with importance-sampling weights ^beta.
    'PER': os.getenv("PER", "0") == "1",
    'PER_alpha': float(os.getenv("PER_ALPHA", "0.5")),
    'PER_beta': float(os.getenv("PER_BETA", "1.0")),
    # exponential learning-rate decay: lr0 * rate ** (step / steps)
    'lr_decay_rate': float(os.getenv("LR_DECAY_RATE", "0.1")),
    'lr_decay_steps': float(os.getenv("LR_DECAY_STEPS", "200000")),
}

agent1 = MuZeroAgent(environment=env, config=config)

# self-play
agents = {
    'player_1': agent1,
    'player_2': agent1
}

# log the cumulative frequency of board states 
# reached during self-play
every_ep_log = {}

start_time = datetime.datetime.now()
for ep in range(TRAIN_EPS):
    env.reset(seed=SEED + ep)

    for a in env.agent_iter():
        agent = agents[a]
        observation, reward, termination, truncation, info = env.last()

        board_state = preprocess_obs(observation)
        if board_state in every_ep_log.keys():
            every_ep_log[board_state] += 1
        else:
            every_ep_log[board_state] = 1

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

        # Record exactly one transition for every action that was actually
        # played. In shared-agent self-play, adding a synthetic losing-player
        # step would create a bogus action/policy pair and a second trajectory.
        agent.experience(
            observation,
            a,
            action,
            env.rewards[a],
            any(env.terminations.values()) or any(env.truncations.values()),
        )

        # check if the game has ended (termination or truncation)
        if any(env.terminations.values()) or any(env.truncations.values()):
            # break the loop to finish the episode
            break

    # agent neural network updates parameters if replay buffer is full
    if (ep + 1) % UPDATE_INTERVAL == 0:
        agent1.update()
    if (ep + 1) % LOG_INTERVAL == 0 or ep == 0 or ep + 1 == TRAIN_EPS:
        elapsed = datetime.datetime.now() - start_time
        print(progress_message(ep, elapsed, agent1, config), flush=True)
    
    if ((ep + 1) % EVAL_INTERVAL == 0) or ep + 1 == TRAIN_EPS:
        eval_agent(agent1, ep)
    
    if ((ep+1) % 500 == 0) or ep+1 == TRAIN_EPS:
        first_logged_ep = max(0, ep - 499)
        with open(f'board_states_eps{first_logged_ep}-{ep}_log.json', 'w') as f:
            json.dump(every_ep_log, f, indent=4)
    # pause = input('\npress enter for new game')
agent1.save_model()
env.close()
