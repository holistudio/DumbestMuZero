from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

import datetime
import csv
import json
import os
import copy

import numpy as np

TRAIN_EPS = 500
EVAL_EPS = 100

def preprocess_obs(observation):
    # pre-process observation dictionary into tensor
    obs = np.zeros((3,3))
    current_player_plane = np.array(observation["observation"][:, :, 0])
    opponent_plane = np.array(observation["observation"][:, :, 1])
    total_pieces = np.sum(current_player_plane) + np.sum(opponent_plane)

    # If total pieces is even, it's player 1's turn (current player is p1)
    # If total pieces is odd, it's player 2's turn (current player is p2)
    if total_pieces % 2 == 0:
        p1_plane, p2_plane = current_player_plane, opponent_plane
    else:
        p2_plane, p1_plane = current_player_plane, opponent_plane

    p2_plane = p2_plane * 2
    obs = obs + p1_plane + p2_plane
    obs = obs.flatten()

    obs_str = ''
    for o in obs:
        obs_str += str(int(o))
    return obs_str

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
    'buffer_size': 400,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'max_iters': 80,
    'train_iters': 100,
    'gamma': 1.0,
    'k_unroll_steps': 5,
    'temperature': 0.5,
    'dirichlet_alpha': 0.05
}

agent1 = MuZeroAgent(environment=env, config=config)

agents = {
    'player_1': agent1,
    'player_2': agent1
}


every_ep_log = {}

start_time = datetime.datetime.now()
for ep in range(TRAIN_EPS):
    env.reset(seed=42)

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

        # Check if the game has ended (termination or truncation)
        if any(env.terminations.values()) or any(env.truncations.values()):
            # Game Over. We must record the experience for ALL agents to ensure 
            # the losing agent (who didn't get to act) receives their negative reward.
            for agent_name in agents.keys():
                p_agent = agents[agent_name]
                r = env.rewards[agent_name]
                t = env.terminations[agent_name]
                trunc = env.truncations[agent_name]
                is_terminal = t or trunc

                # Construct observation for the specific agent
                if agent_name == a:
                    # This is the agent who just acted (or caused termination)
                    curr_obs = observation
                    # Use the action they took. If None (shouldn't happen if they just acted), use 0.
                    act = action if action is not None else 0
                else:
                    # This is the other agent. We need to swap the observation perspective.
                    # TicTacToe obs is (3,3,2). Channel 0 is self, 1 is opponent.
                    curr_obs = copy.deepcopy(observation)
                    p1_plane = curr_obs["observation"][:, :, 0].copy()
                    p2_plane = curr_obs["observation"][:, :, 1].copy()
                    curr_obs["observation"][:, :, 0] = p2_plane
                    curr_obs["observation"][:, :, 1] = p1_plane
                    
                    # This agent did not act, so we pass a random placeholder action
                    # to avoid biasing the dynamics model against a specific move (like 0).
                    act = env.action_space(agent_name).sample()
                
                p_agent.experience(curr_obs, agent_name, act, r, is_terminal)
            
            # Break the loop to finish the episode
            break
        else:
            # Game continues. Record experience for the current agent only.
            agent.experience(observation, a, action, env.rewards[a], env.terminations[a])

    agent1.update()
    print(f'{datetime.datetime.now()-start_time} EP={ep}')
    if ((ep+1) % 10 == 0) or ep+1 == TRAIN_EPS: 
        eval_agent(agent1, ep)
        with open(f'board_states_eps{ep-9}-{ep}_log.json', 'w') as f:
            json.dump(every_ep_log, f, indent=4)
    # pause = input('\npress enter for new game')
env.close()
