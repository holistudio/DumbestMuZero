from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent, set_seed

import datetime
import csv
import json
import os
import copy

import numpy as np

TRAIN_EPS = 10_000 # number of training self-play games
EVAL_EPS = 100 # number of games to play against random agent
SEED = 42

set_seed(SEED)

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
                if env.terminations[a] == True:

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

    # display performance in terminal
    print(f'EP={train_ep} Agent Performance, as P1: {p1_w_perc:.2f}%, {p1_w_l_d}, as P2: {p2_w_perc:.2f}%, {p2_w_l_d}')
    
    # log in CSV file
    csv_filename = 'agent_performance.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['train_ep', 'p1_win', 'p1_loss', 'p1_draw', 'p2_win', 'p2_loss', 'p2_draw'])
        writer.writerow([train_ep] + p1_w_l_d + p2_w_l_d)

"""SELF-PLAY TRAINING"""

# initialize game environment
# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

# initialize MuZero agent with config
config = {
    'batch_size': 128,
    'buffer_size': TRAIN_EPS,
    'min_replay_size': 1000,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_iters': 50,
    'train_iters': 2,
    'gamma': 1.0,
    'k_unroll_steps': 5,
    'temperature': 1.0,
    'temp_schedule': [(0.6*TRAIN_EPS, 1.0), (0.9*TRAIN_EPS, 0.5), (10**9, 0.25)],
    'dirichlet_alpha': 1.0,
    'root_exploration_fraction': 0.4
}

agent1 = MuZeroAgent(environment=env, config=config)
eval_agent(agent1, 0)
print()

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
    episode_num = ep + 1  # 1-indexed episode number, used for all EP display/logging
    env.reset(seed=SEED)

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

        agent.experience(
            observation,
            a,
            action,
            env.rewards[a],
            any(env.terminations.values() or any(env.truncations.values()))
        )

        # check if the game has ended (termination or truncation)
        if any(env.terminations.values()) or any(env.truncations.values()):
            # break the loop to finish the episode
            break

    # agent neural network updates parameters if replay buffer is full
    agent1.update()

    if episode_num == 1 or episode_num % 1000 == 0:
        elapsed = datetime.datetime.now() - start_time
        avg_time_per_ep = elapsed / episode_num
        eta = avg_time_per_ep * (TRAIN_EPS - episode_num)
        eta_completion = datetime.datetime.now() + eta
        print(f'EP={episode_num}/{TRAIN_EPS} | Avg/EP: {avg_time_per_ep} | ETA: {eta} (done at {eta_completion.strftime("%Y-%m-%d %H:%M:%S")})')

    if (episode_num % 1000 == 0) or episode_num == TRAIN_EPS:
        eval_agent(agent1, episode_num)

    if (episode_num % 1000 == 0) or episode_num == TRAIN_EPS:
        with open(f'board_states_eps{episode_num-1000}-{episode_num-1}_log.json', 'w') as f:
            json.dump(every_ep_log, f, indent=4)
    # pause = input('\npress enter for new game')
agent1.save_model()
env.close()

# post-hoc timing summary
total_elapsed = datetime.datetime.now() - start_time
avg_time_per_ep = total_elapsed / TRAIN_EPS
print(f'\nDone: {TRAIN_EPS} episodes in {total_elapsed} (avg {avg_time_per_ep} per episode)')
