import math
import copy
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def min_max_normalize(state):
    """
    Scale a hidden state to the [0, 1] range per-sample (MuZero Appendix G).

    Keeping the latent on a fixed scale stops the recurrent dynamics function
    from drifting to ever-larger magnitudes over the unroll, which is one of the
    main sources of training instability in MuZero.
    """
    min_s = state.min(dim=-1, keepdim=True)[0]
    max_s = state.max(dim=-1, keepdim=True)[0]
    scale = (max_s - min_s).clamp_min(1e-5)
    return (state - min_s) / scale


"""NEURAL NETS"""
class StateFunction(nn.Module):
    """
    representation function, h
    
    input: observation/state of current environment (tic-tac-toe board)
    output: hidden representation of initial observation for subsequent MCTS
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.lin5 = nn.Linear(hidden_size, output_size)
        pass

    def forward(self, obs):
        x = self.lin1(obs)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        x = self.lin3(x)
        x = F.gelu(x)
        x = self.lin4(x)
        x = F.gelu(x)
        x = self.lin5(x)
        # normalize the hidden state to [0, 1] for stable recurrent unrolling
        return min_max_normalize(x)

class DynamicsFunction(nn.Module):
    """
    dynamics function, g
    
    input: hidden state representation, s_t, and candidate action, a
    output: predict next hidden state, s_t+1, and reward, r_t+1
    """
    def __init__(self, input_size, output_size, hidden_size, reward_support):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.state_head = nn.Linear(hidden_size, output_size)
        # reward is predicted as a categorical distribution over a fixed support
        self.reward_head = nn.Linear(hidden_size, reward_support)
        pass

    def forward(self, s_prev, a):
        # concatenate along dimension 1 to support batch processing
        x = torch.cat((s_prev, a), dim=1)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        x = self.lin3(x)
        x = F.gelu(x)
        x = self.lin4(x)
        x = F.gelu(x)
        s = self.state_head(x)
        # normalize the hidden state to [0, 1] for stable recurrent unrolling
        s = min_max_normalize(s)
        r = self.reward_head(x)
        return s, r
    
class PredictionFunction(nn.Module):
    """
    prediction function, f
    
    input: hidden state representation, s_t
    output: policy logits, p_t, and value, v_t
    """
    def __init__(self, input_size, output_size, hidden_size, value_support):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        # value is predicted as a categorical distribution over a fixed support
        self.value_head = nn.Linear(hidden_size, value_support)
        pass

    def forward(self, s):
        x = self.lin1(s)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        x = self.lin3(x)
        x = F.gelu(x)
        x = self.lin4(x)
        x = F.gelu(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

"""MCTS DATA STRUCTURES"""
class Node(object):
    """
    MCTS node 
    """
    def __init__(self, prior):
        """
        prior: initial probability (from the policy network) 
        of selecting the action that leads to this node from parent node
        """
        self.state = None # hidden state representation
        self.value_sum = 0 # NOT Q-value (see mean_value)
        self.N = 0 # number of node visits
        self.P = prior # policy 
        self.R = 0 # immediate reward

        # child nodes
        self.children = {}
        pass

    def expanded(self):
        """
        check if its a leaf node with no children or not
        """
        return len(self.children.items()) > 0
    
    def mean_value(self):
        """
        return the mean value Q
        """
        if self.N > 0:
            return  self.value_sum / self.N # Q-value
        else:
            return 0

"""TRAINING DATA STRUCURES"""
class ReplayBuffer(object):
    """
    replay buffer for storing entire game trajectories
    and training batch sampling 
    """
    def __init__(self, buffer_size, batch_size, per=False, per_alpha=0.5, per_beta=1.0):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Prioritized Experience Replay (PER). When disabled, sampling is
        # uniform and IS weights are all 1.0 (original behavior preserved).
        self.per = per
        self.per_alpha = per_alpha   # priority exponent (0 = uniform)
        self.per_beta = per_beta     # importance-sampling correction exponent
        # Parallel to self.buffer: per-position priorities (np.array) and the
        # per-game priority (max of its positions). total_samples tracks the
        # number of stored positions across the whole buffer.
        self.priorities = []
        self.game_priorities = []
        self.total_samples = 0

        self.buffer = [] # store each game's trajectory

        # TRAJECTORY = zip(observations, player_turns, actions, immediate_rewards, target_policies, final_outcomes)
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.rewards = []
        self.root_values = []
        pass
    
    def reset_trajectory(self):
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.rewards = []
        self.root_values = []
        pass

    def store_step(self, obs, player_turn, action, target_action_probs, reward, root_value):
        # observation vector
        self.observations.append(obs)

        # player turn
        self.player_turns.append(player_turn)

        # action direct from what was chosen by select_action (different from child visits if sampled)
        self.actions.append(action)

        self.target_policies.append(target_action_probs)

        # get the reward from environment
        self.rewards.append(reward)

        self.root_values.append(root_value)
        pass

    def store_trajectory(self):
        trajectory = dict(obs=self.observations, 
                         turns=self.player_turns, actions=self.actions, 
                         rewards=self.rewards, # dynamics function target
                         target_policies=self.target_policies,
                         root_values=self.root_values) # prediction function targets
        
        trajectory = copy.deepcopy(trajectory)
        self.buffer.append(trajectory)

        if self.per:
            # Optimistic initialization: give every new position the current
            # maximum priority so it is sampled at least once, after which the
            # training step overwrites it with the real value-prediction error.
            n = len(trajectory['root_values'])
            max_prio = max(self.game_priorities) if self.game_priorities else 1.0
            self.priorities.append(np.full(n, max_prio, dtype=np.float32))
            self.game_priorities.append(float(max_prio))
            self.total_samples += n

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            if self.per:
                popped = self.priorities.pop(0)
                self.game_priorities.pop(0)
                self.total_samples -= len(popped)

        self.reset_trajectory()
        pass

    def update_priorities(self, sample_indices, new_priorities):
        """
        Write back per-position priorities after a training step. Indices are
        (game_index, position_index) pairs returned by sample_batch. In this
        synchronous design no trajectory is stored between sampling and update,
        so the indices remain valid for the duration of one update() call.
        """
        if not self.per:
            return
        for (ep_idx, ix), prio in zip(sample_indices, new_priorities):
            if 0 <= ep_idx < len(self.priorities):
                arr = self.priorities[ep_idx]
                if ix < len(arr):
                    arr[ix] = prio
                    self.game_priorities[ep_idx] = float(arr.max())

    def sample_batch(self, k_unroll_steps, gamma, device, td_steps=None):
        # Prepare lists to collect batch data
        obs_batch = []
        actions_batch = []
        target_rewards = []
        target_values = []
        target_policies_batch = []
        legal_masks = []

        num_eps = len(self.buffer)
        # PER bookkeeping: per-batch-element (game, position) indices and the
        # sampling probability of each, used to build importance-sampling
        # weights and to write priorities back after the training step.
        sample_indices = []
        sample_probs = []
        if self.per and self.game_priorities:
            gp = np.asarray(self.game_priorities, dtype=np.float64)
            gp_total = gp.sum()
            game_probs = gp / gp_total if gp_total > 0 else None
        else:
            game_probs = None

        for b in range(self.batch_size):
            # --- choose a game ---
            if game_probs is not None:
                ep_idx = int(np.random.choice(num_eps, p=game_probs))
            else:
                ep_idx = random.randint(0, num_eps - 1)
            random_ep = self.buffer[ep_idx]
            observations, player_turns, actions = random_ep['obs'], random_ep['turns'], random_ep['actions']
            rewards, target_policies, root_values = random_ep['rewards'], random_ep['target_policies'], random_ep['root_values']

            if td_steps is None:
                td_steps = k_unroll_steps

            # --- choose a position within the game ---
            # Every real position must be eligible as an initial inference
            # target. Recurrent targets use a fixed k-step bootstrap horizon,
            # and positions too close to the end of the game fall back to the
            # pure return when that bootstrap index runs past the trajectory.
            if game_probs is not None:
                pos_prio = self.priorities[ep_idx]
                pp_total = pos_prio.sum()
                if pp_total > 0:
                    pos_probs = pos_prio / pp_total
                    ix = int(np.random.choice(len(pos_prio), p=pos_probs))
                    sample_prob = float(game_probs[ep_idx]) * float(pos_probs[ix])
                else:
                    ix = random.randint(0, len(root_values) - 1)
                    sample_prob = float(game_probs[ep_idx]) / len(root_values)
            else:
                ix = random.randint(0, len(root_values) - 1)
                sample_prob = None
            sample_indices.append((ep_idx, ix))
            sample_probs.append(sample_prob)

            # Collect initial observation for this sequence
            obs_batch.append(observations[ix])
            
            # Collect actions for unrolling, padding with random actions if we go past the end of the game
            current_actions = []
            for k in range(k_unroll_steps):
                if ix + k < len(actions):
                    current_actions.append(actions[ix+k])
                else:
                    # Pad with a random valid action index (e.g., 0) for absorbing states
                    current_actions.append(random.randint(0, len(target_policies[0])-1))
            actions_batch.append(current_actions)

            sample_rewards = []
            sample_values = []
            sample_policies = []
            sample_masks = []

            for i in range(ix, ix+k_unroll_steps+1):
                # Pre-compute legal action mask based on recorded observation
                if i < len(observations):
                    mask = (observations[i] == 0)
                else:
                    # In absorbing state, allow all actions (or mask as needed, but target is uniform)
                    mask = torch.ones_like(observations[0], dtype=torch.bool)
                sample_masks.append(mask)

                if i < len(player_turns):
                    current_player = player_turns[i]
                else:
                    current_player = None
                
                bootstrap_ix = i + td_steps

                if bootstrap_ix < len(root_values):
                    value = root_values[bootstrap_ix] * gamma**td_steps
                    bootstrap_player = player_turns[bootstrap_ix]
                    if current_player is not None and current_player != bootstrap_player:
                        value = -value
                else:
                    value = 0
                for j, reward in enumerate(rewards[i:bootstrap_ix]):
                    if i+j < len(player_turns) and player_turns[i+j] == current_player:
                        value += reward * gamma**j
                    else:
                        value -= reward * gamma**j
                
                if i > 0 and i <= len(rewards):
                    last_reward = rewards[i-1]
                else:
                    last_reward = 0

                sample_rewards.append(last_reward)
                sample_values.append(value)
                
                # --- DEBUG DIAGNOSIS START ---
                # print(f"\n[DEBUG Step {i}] Current Player: {current_player}")
                # print(f"  Calculated Target Value: {value}")
                # print(f"  Last Reward (Raw): {last_reward}")
                # if bootstrap_ix < len(root_values):
                #     b_player = player_turns[bootstrap_ix]
                #     print(f"  Bootstrap Player (ix={bootstrap_ix}): {b_player}")
                #     if current_player is not None and b_player != current_player:
                #         print("  -> Value sign flipped from bootstrap? Yes (Correct for zero-sum)")
                # else:
                #     print(f"  Bootstrap Index (ix={bootstrap_ix}) out of bounds (Terminal State). Value set to 0.")
                # if i > 0 and i <= len(rewards):
                #     prev_player = player_turns[i-1]
                #     print(f"  Player who generated last_reward (at {i-1}): {prev_player}")
                #     if current_player is not None and prev_player != current_player:
                #         print(f"  -> Perspective Switch: Yes. Prev ({prev_player}) vs Curr ({current_player})")
                #         print(f"  -> If raw reward is {last_reward}, expected target reward for current player is {-last_reward}")
                # --- DEBUG DIAGNOSIS END ---

                if i < len(root_values):
                    sample_policies.append(target_policies[i])
                else:
                    # A zero soft target contributes no policy loss for
                    # absorbing states.
                    sample_policies.append(torch.zeros_like(target_policies[0]))
            
            target_rewards.append(sample_rewards)
            target_values.append(sample_values)
            target_policies_batch.append(torch.stack(sample_policies))
            legal_masks.append(torch.stack(sample_masks))

        # Importance-sampling weights for PER (all 1.0 when PER is disabled),
        # normalized by the max weight in the batch so they only scale down.
        if self.per and all(p is not None for p in sample_probs):
            probs = np.asarray(sample_probs, dtype=np.float64)
            weights = (self.total_samples * probs) ** (-self.per_beta)
            weights = weights / weights.max()
            is_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            is_weights = torch.ones(self.batch_size, dtype=torch.float32, device=device)

        # Stack into tensors and move to device once
        return (torch.stack(obs_batch).to(device),
                torch.tensor(actions_batch, dtype=torch.long).to(device),
                torch.tensor(target_rewards, dtype=torch.float32).to(device),
                torch.tensor(target_values, dtype=torch.float32).to(device),
                torch.stack(target_policies_batch).to(device),
                torch.stack(legal_masks).to(device),
                is_weights,
                sample_indices)

"""MUZERO"""
class MuZeroAgent(object):
    """
    MuZero agent class
    """
    def __init__(self, environment, config, load=False):
        self.env = environment
        self.observation_space = self.flatten(environment.observation_space('player_1'))
        self.obs_size = self.observation_space.shape
        self.action_space = environment.action_space('player_1')
        self.action_size = self.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # replay buffer for loading training batch data
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'],
            config['batch_size'],
            per=config.get('PER', False),
            per_alpha=config.get('PER_alpha', 0.5),
            per_beta=config.get('PER_beta', 1.0),
        )
        self.buffer_size = config['buffer_size']
        self.min_replay_size = config.get('min_replay_size', self.buffer_size)

        # categorical (distributional) value/reward representation.
        # Values/rewards are encoded as a two-hot vector over a fixed support and
        # learned via cross-entropy; an invertible R2D2 transform optionally
        # squashes the scale before encoding. Defaults keep diagnostic scripts
        # that omit these keys working out of the box.
        self.num_bins = config.get('num_bins', 51)
        self.support_limit = float(config.get('support_limit', 1.0))
        self.value_transform = config.get('value_transform', True)
        # weight on the value/reward CE terms relative to policy CE. MuZero
        # convention is 0.25; default 1.0 here preserves prior behavior.
        self.value_loss_weight = float(config.get('value_loss_weight', 1.0))
        self.support_min = -self.support_limit
        self.support_max = self.support_limit
        self.support = torch.linspace(
            self.support_min, self.support_max, self.num_bins, device=self.device
        )
        self.support_delta = (self.support_max - self.support_min) / (self.num_bins - 1)

        # neural networks
        self.state_function = StateFunction(self.obs_size[0],
                                            config['state_size'],
                                            config['hidden_size'])

        self.dynamics_function = DynamicsFunction(config['state_size'] + self.action_size,
                                                  config['state_size'],
                                                  config['hidden_size'],
                                                  self.num_bins)

        self.prediction_function = PredictionFunction(config['state_size'],
                                                      self.action_size,
                                                      config['hidden_size'],
                                                      self.num_bins)
        # move to CUDA device, if available
        self.state_function.to(self.device)
        self.dynamics_function.to(self.device)
        self.prediction_function.to(self.device)

        all_function_params = (list(self.state_function.parameters()) +
                               list(self.dynamics_function.parameters()) +
                               list(self.prediction_function.parameters()))
        
        # training optimizer with
        # L2 regularization loss term
        self.optimizer = torch.optim.AdamW(params=all_function_params,
                                           lr=config['lr'],
                                           weight_decay=config['weight_decay'])

        # exponential learning-rate decay (MuZero-style):
        #   lr(step) = lr0 * decay_rate ** (step / decay_steps)
        # Default decay_rate=1.0 is a no-op so scripts that omit these keys are
        # unaffected; train.py supplies real decay values.
        self.lr_decay_rate = float(config.get('lr_decay_rate', 1.0))
        self.lr_decay_steps = float(config.get('lr_decay_steps', 100_000))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self.lr_decay_rate ** (step / self.lr_decay_steps),
        )

        self.min_Q = float('inf')
        self.max_Q = -float('inf')
        self.max_iters = config['max_iters']
        self.train_iters = config['train_iters']
        self.checkpoint_interval = config.get('checkpoint_interval', 1)
        self.gamma = config['gamma']
        self.k_unroll_steps = config['k_unroll_steps']
        # Count of completed self-play games (drives the temperature schedule).
        self.episodes_played = 0
        # Value-target bootstrap horizon. None => tie to k_unroll_steps (current
        # behavior). A value >= max game length makes the bootstrap index always
        # run past the trajectory, collapsing the target to the pure Monte-Carlo
        # game return (the canonical MuZero board-game setting).
        self.td_steps = config.get('td_steps', None)
        self.training_steps = 0
        self.last_checkpoint_step = 0
        self.last_loss = None
        self.last_update_losses = []

        self.root_value = 0
        self.action_probs = torch.zeros(self.action_size)
        self.temperature = config['temperature']
        # Optional AlphaZero-style temperature schedule over EPISODES (completed
        # self-play games): a list of (threshold_episodes, temperature) pairs,
        # sorted by threshold. The temperature of the first threshold that
        # episodes_played is below is used; past the last threshold the final
        # temperature holds. None => constant self.temperature.
        # Episodes (not network-update steps) so the schedule maps directly onto
        # the planned training-game budget regardless of update frequency.
        self.temp_schedule = config.get('temp_schedule', None)
        self.dirichlet_alpha = config['dirichlet_alpha']
        # whether to inject Dirichlet root noise on the GREEDY (temperature=0)
        # self-play plies too. True = current behavior; False = noise only on
        # the exploratory plies (coherent: don't corrupt a search we argmax over).
        self.noise_on_greedy = config.get('noise_on_greedy', True)


        self.state_function.eval()
        self.dynamics_function.eval()
        self.prediction_function.eval()

        if load:
            self.load_model()
        pass
    
    """model utilities"""
    def save_model(self, directory=None):
        """
        save neural network and optimizer parameters
        """
        base_dir = directory or os.path.dirname(os.path.abspath(__file__))
        os.makedirs(base_dir, exist_ok=True)
        
        torch.save(self.state_function.state_dict(), 
                   os.path.join(base_dir, 'mu_state_rep_params.pth.tar'))
        torch.save(self.dynamics_function.state_dict(), 
                   os.path.join(base_dir, 'mu_dyn_func_params.pth.tar'))
        torch.save(self.prediction_function.state_dict(), 
                   os.path.join(base_dir, 'mu_pred_func_params.pth.tar'))
        torch.save(self.optimizer.state_dict(), 
                   os.path.join(base_dir, 'mu_optimizer_params.pth.tar'))
        
        # print("Models and optimizer saved.")
        pass

    def load_model(self, directory=None):
        """
        load neural network and optimizer parameters
        """

        base_dir = directory or os.path.dirname(os.path.abspath(__file__))
        
        paths = {
            'state': os.path.join(base_dir, 'mu_state_rep_params.pth.tar'),
            'dynamics': os.path.join(base_dir, 'mu_dyn_func_params.pth.tar'),
            'prediction': os.path.join(base_dir, 'mu_pred_func_params.pth.tar'),
            'optimizer': os.path.join(base_dir, 'mu_optimizer_params.pth.tar')
        }

        if os.path.exists(paths['state']):
            self.state_function.load_state_dict(torch.load(paths['state'], map_location=self.device))
        if os.path.exists(paths['dynamics']):
            try:
                self.dynamics_function.load_state_dict(torch.load(paths['dynamics'], map_location=self.device))
            except RuntimeError as error:
                raise RuntimeError(
                    "Dynamics checkpoint is incompatible with one-hot action encoding. "
                    "Start a new training run or load a checkpoint created after this change."
                ) from error
        if os.path.exists(paths['prediction']):
            self.prediction_function.load_state_dict(torch.load(paths['prediction'], map_location=self.device))
        if os.path.exists(paths['optimizer']):
            self.optimizer.load_state_dict(torch.load(paths['optimizer'], map_location=self.device))
        print("Models and optimizer loaded.")
        pass

    """game environment helper functions"""
    def flatten(self, observation_space):
        """
        convert observation space
        """
        # TODO: move back into init...

        H, W, C = observation_space['observation'].shape
        # return observation space as a 1-D vector
        return torch.zeros(H*W)

    def encode_actions(self, actions):
        """Convert categorical action indices into one-hot dynamics inputs."""
        actions = actions.to(self.device).long()
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        return F.one_hot(actions, num_classes=self.action_size).to(dtype=torch.float32)

    def preprocess_obs(self, observation):
        """
        convert environment observation dictionary
        into a canonical (mover-relative) torch tensor for neural nets

        The PettingZoo observation planes are already current-player-relative:
        plane 0 holds the player-to-move's pieces and plane 1 the opponent's.
        We encode this as a single channel where +1 marks the current player's
        pieces, -1 marks the opponent's, and 0 marks empty squares.

        Keeping the board in the mover's perspective means the networks only
        ever learn a single "side-to-move" policy/value, which matches the
        negamax value convention used in search (pUCT/backup) and in the
        training value targets. (Previously the board was de-canonicalised into
        an absolute X=1/O=2 encoding, forcing the net to learn both sides and
        infer whose turn it was from piece parity.)
        """
        current_player_plane = torch.tensor(
            observation["observation"][:, :, 0], dtype=torch.float32
        )
        opponent_plane = torch.tensor(
            observation["observation"][:, :, 1], dtype=torch.float32
        )
        obs = current_player_plane - opponent_plane
        return obs.reshape(-1)
    
    def display_board(self, obs):
        """
        display the board to the terminal
        """

        board = [
                [" "," "," "],
                [" "," "," "],
                [" "," "," "]]
        
        obs_grid = obs.reshape(3, 3)

        # Canonical board: +1 is the current player (shown as X), -1 the
        # opponent (shown as O).
        for i in range(3):
            for j in range(3):
                if obs_grid[i, j] == 1:
                    board[j][i] = "X"
                elif obs_grid[i, j] == -1:
                    board[j][i] = "O"

        print("BOARD")
        print("=====")
        for i,row in enumerate(board):
            row_disp = ("|").join(row)
            print(row_disp)
            if i < 2:
                print("-----")
        print("=====")
        print()


    """MuZero search functions"""
    def get_legal_actions(self, temp_board, history):
        """
        given a canonical (mover-relative) board and a hypothetical action
        history, return indices of available legal actions that can be taken

        The board is always from the perspective of the player to move at the
        root, so the root mover places +1 and players alternate (+1, -1, +1,
        ...) as the history is replayed. An empty list is returned once any
        line is completed (terminal node).
        """
        temp_board = temp_board.clone()
        for depth, action in enumerate(history):
            temp_board[action] = 1.0 if depth % 2 == 0 else -1.0

        winning_combinations = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for combo in winning_combinations:
            values = temp_board[list(combo)]
            if values[0] != 0 and torch.all(values == values[0]):
                return []

        return torch.where(temp_board == 0)[0].tolist()

    def update_min_max_Q(self, node_mean_value):
        """
        keep track of min and max over entire search tree
        """
        self.max_Q = max(self.max_Q, node_mean_value)
        self.min_Q = min(self.min_Q, node_mean_value)
        pass

    def expansion(self, last_node, state, reward, policy_logits, legal_actions, action_history):
        """
        expansion phase of MuZero search
        add child nodes to a given leaf node based on legal actions
        and initialize them with policy priors based on the prediction network
        """

        last_node.state = state
        last_node.R = reward

        # if legal_actions is empty to avoid a ValueError
        if not legal_actions:
            return

        # Since networks now process batches, squeeze the batch dimension for single inferences
        policy_logits = policy_logits.squeeze(0)

        # mask illegal actions and normalize the policy over legal moves
        # Use .item() to convert tensor logits to float for math.exp
        # Subtract max logit for numerical stability to prevent ZeroDivisionError
        max_logit = max(policy_logits[a].item() for a in legal_actions)
        policy = {a: math.exp(policy_logits[a].item() - max_logit) for a in legal_actions}
        policy_sum = sum(policy.values())
        for a in policy.keys():
            last_node.children[a] = Node(policy[a]/policy_sum)
        pass

    def pUCT(self, node, sum_visits):
        """
        upper confidence bound computed for a given candidate action node
        refer to Equation 2, Appendix B

        sum_visits: total number of visits to the parent node
        (same as sum of visits across all parent's child nodes)
        """
        
        c1 = 1.25
        c2 = 19652
        N = node.N
        if N > 0:
            # Child values are stored from the child player's perspective.
            # The parent is the opposing player, so negate the continuation
            # value before using it to select the parent's action.
            Q = -node.mean_value()
            if self.max_Q > self.min_Q:
                # Negating values also swaps/negates the normalization bounds.
                Q = (self.max_Q - node.mean_value()) / (self.max_Q - self.min_Q)
            Q = node.R + self.gamma * Q 
        else:
            Q = 0
        P = node.P
        N_sum = sum_visits
        return (Q + (P*math.sqrt(N_sum)/(1+N))*(c1+math.log((N_sum+c2+1)/c2)))

    def select_child(self, node):
        """
        select the next child node during simulation/search

        (NOT for selecting the next action in the game)
        """
        
        # parent node visits same as sum of visits across all parent's child nodes
        sum_visits = node.N

        best_uct = -float('inf')
        best_child = None
        best_action = None
        for a, child_node in node.children.items():
            uct = self.pUCT(child_node, sum_visits)
            if uct > best_uct:
                best_uct = uct
                best_action = a
                best_child = child_node
        return best_child, best_action

    def selection(self, node):
        """
        selection phase of MuZero search

        return:
        - unexpanded leaf node
        - latest search path and action history
        """
        search_path = [node]
        action_history = []

        while node.expanded():
            node, action = self.select_child(node)
            search_path.append(node)
            action_history.append(action)
        return node, search_path, action_history

    def backup(self, value, search_path):
        """
        backup phase of MuZero search
        update mean value based on simulated game outcomes 
        and node visit counts during simulation
        """
        # `value` starts from the leaf player's perspective. Node rewards are
        # from the perspective of the player who acted at the parent node.
        G = value
        for current_node in reversed(search_path):
            current_node.value_sum += G
            current_node.N += 1
            self.update_min_max_Q(current_node.mean_value())
            G = current_node.R - self.gamma * G
        pass

    def select_action(self, node, temperature):
        """
        select the next action to take in the game
        given tree search root node and softmax sampling temperature
        """

        # sample action based on visit counts and temperature
        sum_visits = node.N
        self.action_probs = torch.zeros(self.action_size)
        
        # account for visit counts for each action
        visits = []
        actions = []
        for a, child_node in node.children.items():
            visits.append(child_node.N)
            actions.append(a)
            self.action_probs[a] = child_node.N / sum_visits

        if temperature == 0:
            # greedy selection (argmax)
            max_visits = -1
            best_action = None
            for a, v in zip(actions, visits):
                if v > max_visits:
                    max_visits = v
                    best_action = a
            return best_action
        else:
            # softmax sampling with temperature
            # P(a) = (N(a)^(1/T)) / sum(N(b)^(1/T))
            visits_tensor = torch.tensor(visits, dtype=torch.float32)
            scaled_visits = visits_tensor.pow(1.0 / temperature)
            probs = scaled_visits / scaled_visits.sum()
            
            # sample from multinomial distribution
            action_idx = torch.multinomial(probs, 1).item()
            return actions[action_idx]
        
    def add_exploration_noise(self, node):
        """
        add Dirichlet noise to root node's child node policy priors
        to encourage exploration of different actions during MuZero search/simulation
        """
        root_exploration_fraction = 0.25
        actions = list(node.children.keys())
        if not actions:
            return
        noise = torch.distributions.Dirichlet(torch.full((len(actions),), self.dirichlet_alpha)).sample()
        for a, n in zip(actions, noise):
            node.children[a].P = node.children[a].P * (1 - root_exploration_fraction) + n * root_exploration_fraction
            
    def search(self, obs, temperature, add_exploration_noise=True):
        """
        overall MuZero search algorithm
        
        given:
        - obs: tensor representation of current game environment observation state
        - temperature: softmax sampling temperature

        return: next action to play in the game
        """
        # print('search()')

        # ensure inference mode
        with torch.no_grad():
            # Value normalization statistics are local to one search tree.
            self.min_Q = float('inf')
            self.max_Q = -float('inf')

            # initialize tree root node
            root_node = Node(0)

            # encode observation into a hidden state represnetation
            initial_state = self.state_function(obs.to(self.device).unsqueeze(0))

            # predict initial policy logits and value
            policy_logits, value = self.prediction_function(initial_state)

            # get list of current available legal actions
            action_history = []
            legal_actions = self.get_legal_actions(obs, action_history)

            # expand root node
            self.expansion(root_node, initial_state, 0, policy_logits, legal_actions, action_history)
            
            # add exploration noise to root node's children
            if add_exploration_noise:
                self.add_exploration_noise(root_node)

            for i in range(self.max_iters):
                # select leaf node
                last_node, search_path, action_history = self.selection(root_node)

                # TODO: revisit this
                # Check if node is already expanded (has state) but has no children (terminal/leaf).
                # This prevents re-expanding terminal nodes and handles the edge case where Root is terminal.
                if last_node.state is not None:
                    self.backup(0, search_path)
                    continue

                # get leaf node's parent
                parent_node = search_path[-2]

                # get latest candidate action as a tensor
                latest_action = torch.tensor([action_history[-1]], dtype=torch.long, device=self.device)

                # dynamics function predicts next state and immediate reward
                state, reward = self.dynamics_function(
                    parent_node.state,
                    self.encode_actions(latest_action),
                )

                # prediciton function estimates policy logits and value based on next state
                policy_logits, value = self.prediction_function(state)

                # get legal actions avalaible at the leaf node given previous actions
                legal_actions = self.get_legal_actions(obs, action_history)

                # expand leaf node with child nodes for each legal action
                # decode the categorical reward logits into a scalar
                self.expansion(last_node, state, self.support_to_scalar(reward).item(), policy_logits, legal_actions, action_history)
                
                # --- DEBUG START ---
                # print(f"\n[DEBUG] Simulation {i+1}/{self.max_iters}")
                # print(f"Leaf Value (NN): {value.item():.4f}")
                
                # temp_board = obs.clone()
                # # current_agent_idx = 0 if self.env.agent_selection == 'player_1' else 1

                # print("Path BEFORE Backup:")
                # for depth, node in enumerate(search_path):
                #     if depth > 0:
                #         act = action_history[depth-1]
                #         mover_idx = self.whose_turn(action_history[:depth-1])
                #         marker = 1 if mover_idx == 0 else 2
                #         temp_board[int(act)] = marker
                    
                #     print(f"  Depth {depth} | Action: {action_history[depth-1] if depth > 0 else 'Root'} | "
                #           f"N: {node.N} | V_sum: {float(node.value_sum):.4f} | Mean V: {float(node.mean_value()):.4f} | R: {float(node.R):.4f}")
                #     self.display_board(temp_board)
                # --- DEBUG END ---

                # update node mean values back up to the root node
                # A node with no legal actions is a known terminal state, so
                # its continuation value is exactly zero.
                # decode the categorical value logits into a scalar
                leaf_value = self.support_to_scalar(value).item() if legal_actions else 0
                self.backup(leaf_value, search_path)
                
                # --- DEBUG START ---
                # print("Path AFTER Backup:")
                # for depth, node in enumerate(search_path):
                #     print(f"  Depth {depth} | N: {node.N} | V_sum: {float(node.value_sum):.4f} | Mean V: {float(node.mean_value()):.4f}")
                # --- DEBUG END ---

                # pause = input('end of one search simulation\n')
            
            # store the mean value of the root node
            self.root_value = root_node.mean_value()
            # expose the root for inspection/debugging (per-child N/Q/P stats)
            self.search_root = root_node

        # return the next action based on node visits and softmax sampling temperature
        return self.select_action(root_node, temperature)
    
    """RL agent standard functions"""

    def experience(self, observation, player_label, action, reward, terminal):
        """
        update experience
        store either recent state-action-reward transition step
        or entire game trajectory if game is complete
        """
        obs = self.preprocess_obs(observation)
        player_turn = 0 if player_label == 'player_1' else 1
        self.replay_buffer.store_step(
            obs,
            player_turn,
            action,
            self.action_probs.clone(),
            reward,
            self.root_value,
        )
        if terminal:
            # once final_outcome is nonzero, label the entire trajectory with the final outcome
            # self.replay_buffer.final_outcomes = [final_outcome if i == player_turn else -final_outcome for i in self.replay_buffer.player_turns]
            self.replay_buffer.store_trajectory()
            # one completed self-play game; advances the temperature schedule
            self.episodes_played += 1
        pass

    def step(self, observation):
        """
        select next action to play in the game
        with softmax temperature sampling
        """
        obs = self.preprocess_obs(observation)

        # muzero-general convention: sample from the MCTS visit distribution at
        # the configured temperature on EVERY self-play move (no greedy
        # annealing), with root Dirichlet exploration noise. Always-stochastic
        # self-play maximizes data diversity; greedy play is used only in act()
        # (evaluation). The old move-count anneal to temperature=0 is removed.
        # Temperature may decay over training steps via temp_schedule.
        action = self.search(obs, self.current_temperature())

        return action

    def current_temperature(self):
        """
        Resolve the self-play sampling temperature for the current training
        step from the optional schedule (constant self.temperature if none).
        """
        if not self.temp_schedule:
            return self.temperature
        for threshold, temp in self.temp_schedule:
            if self.episodes_played < threshold:
                return temp
        return self.temp_schedule[-1][1]
    
    def act(self, observation):
        """
        select next action to play in the game
        with neural nets doing inference 
        """
        # TODO: review whether or not nothing else is different from step()

        # neural nets should be in evaluation mode
        self.state_function.eval()
        self.dynamics_function.eval()
        self.prediction_function.eval()
        
        obs = self.preprocess_obs(observation)
        action = self.search(obs, 0.0, add_exploration_noise=False)
        return action

    def scale_gradient(self, tensor, scale):
        """
        scale gradients for stability during recurrent network training
        """
        return tensor * scale + tensor.detach() * (1.0 - scale)

    """categorical (distributional) value/reward helpers"""
    def _h_transform(self, x):
        """Invertible R2D2 scaling: h(x) = sign(x)(sqrt(|x|+1)-1) + eps*x."""
        eps = 0.001
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

    def _h_inverse(self, x):
        """Inverse of _h_transform."""
        eps = 0.001
        return torch.sign(x) * (
            ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
        )

    def scalar_to_support(self, x):
        """
        Two-hot encode a batch of scalars onto the fixed support.

        x: tensor of shape (N,) or (N, 1). Returns (N, num_bins) probabilities.
        """
        x = x.reshape(-1).to(self.device)
        if self.value_transform:
            x = self._h_transform(x)
        x = x.clamp(self.support_min, self.support_max)
        pos = (x - self.support_min) / self.support_delta
        lower = torch.floor(pos).long()
        upper = torch.clamp(lower + 1, max=self.num_bins - 1)
        w_upper = pos - lower.to(pos.dtype)
        w_lower = 1.0 - w_upper
        target = torch.zeros(x.shape[0], self.num_bins, device=self.device)
        target.scatter_add_(1, lower.unsqueeze(1), w_lower.unsqueeze(1))
        target.scatter_add_(1, upper.unsqueeze(1), w_upper.unsqueeze(1))
        return target

    def support_to_scalar(self, logits):
        """
        Convert categorical logits back to a scalar expectation.

        logits: (N, num_bins). Returns (N, 1).
        """
        probs = F.softmax(logits, dim=-1)
        x = (probs * self.support).sum(dim=-1, keepdim=True)
        if self.value_transform:
            x = self._h_inverse(x)
        return x

    def categorical_loss(self, logits, target_scalar):
        """Cross-entropy between predicted distribution and two-hot target."""
        target = self.scalar_to_support(target_scalar)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    def policy_loss(self, policy_logits, target_policy, legal_mask):
        """
        Cross-entropy over legal actions only.

        Absorbing states use an all-zero target and contribute no policy loss.
        """
        legal_targets = target_policy.masked_fill(~legal_mask, 0)
        target_sums = legal_targets.sum(dim=1, keepdim=True)
        valid_targets = target_sums.squeeze(1) > 0
        legal_targets = torch.where(
            target_sums > 0,
            legal_targets / target_sums.clamp_min(torch.finfo(legal_targets.dtype).eps),
            legal_targets,
        )
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        per_sample_loss = -(legal_targets * F.log_softmax(masked_logits, dim=1)).sum(dim=1)
        if valid_targets.any():
            return per_sample_loss[valid_targets].mean()
        return policy_logits.sum() * 0

    def categorical_loss_elementwise(self, logits, target_scalar):
        """Per-sample categorical cross-entropy (no batch reduction). Returns (N,)."""
        target = self.scalar_to_support(target_scalar)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1)

    def policy_loss_elementwise(self, policy_logits, target_policy, legal_mask):
        """
        Per-sample policy cross-entropy over legal actions (no batch reduction).
        Absorbing states (all-zero target) contribute exactly zero. Returns (N,).
        """
        legal_targets = target_policy.masked_fill(~legal_mask, 0)
        target_sums = legal_targets.sum(dim=1, keepdim=True)
        valid = (target_sums.squeeze(1) > 0).to(policy_logits.dtype)
        legal_targets = torch.where(
            target_sums > 0,
            legal_targets / target_sums.clamp_min(torch.finfo(legal_targets.dtype).eps),
            legal_targets,
        )
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        per_sample = -(legal_targets * F.log_softmax(masked_logits, dim=1)).sum(dim=1)
        return per_sample * valid

    def update(self):
        """
        update neural network parameters
        """
        # print('update()')

        if len(self.replay_buffer.buffer) >= self.min_replay_size:
            # set neural nets to training mode
            self.state_function.train()
            self.dynamics_function.train()
            self.prediction_function.train()

            use_per = self.replay_buffer.per
            update_losses = []
            for epoch in range(self.train_iters):
                # sample batch as stacked tensors (+ IS weights and indices for PER)
                obs_batch, actions_batch, target_rewards_batch, target_values_batch, \
                    target_policies_batch, legal_masks_batch, is_weights, sample_indices = \
                    self.replay_buffer.sample_batch(self.k_unroll_steps, self.gamma, self.device, td_steps=self.td_steps)

                # process entire batch of observations at once
                state = self.state_function(obs_batch)
                policy_logits, value = self.prediction_function(state)

                # get batch targets for step 0 (scalars; categorical loss
                # encodes them onto the support internally)
                t_value = target_values_batch[:, 0]
                t_policy = target_policies_batch[:, 0]
                mask = legal_masks_batch[:, 0]

                # gradient scale for recurrent backpropagation through time stable training
                gradient_scale = 1.0 / self.k_unroll_steps

                if use_per:
                    # --- PER path: per-sample losses, IS-weighted, with priority writeback ---
                    ps_loss = self.policy_loss_elementwise(policy_logits, t_policy, mask) + \
                              self.value_loss_weight * self.categorical_loss_elementwise(value, t_value)
                    # Step-0 value-prediction error drives the new priority.
                    with torch.no_grad():
                        pred_v0 = self.support_to_scalar(value).squeeze(-1)
                        target_v0 = t_value.detach()
                    for k in range(self.k_unroll_steps):
                        action = self.encode_actions(actions_batch[:, k])
                        state, pred_reward = self.dynamics_function(state, action)
                        policy_logits, value = self.prediction_function(state)
                        state = self.scale_gradient(state, 0.5)
                        t_value = target_values_batch[:, k+1]
                        t_policy = target_policies_batch[:, k+1]
                        t_reward = target_rewards_batch[:, k+1]
                        mask = legal_masks_batch[:, k+1]
                        l = self.value_loss_weight * self.categorical_loss_elementwise(pred_reward, t_reward) + \
                            self.policy_loss_elementwise(policy_logits, t_policy, mask) + \
                            self.value_loss_weight * self.categorical_loss_elementwise(value, t_value)
                        ps_loss = ps_loss + self.scale_gradient(l, gradient_scale)
                    loss = (ps_loss * is_weights).mean()
                else:
                    # --- original uniform path (preserved exactly) ---
                    loss = self.policy_loss(policy_logits, t_policy, mask) + \
                           self.value_loss_weight * self.categorical_loss(value, t_value)
                    for k in range(self.k_unroll_steps):
                        # Convert categorical action indices to one-hot dynamics inputs.
                        action = self.encode_actions(actions_batch[:, k])
                        # dynamics and prediction function on batch
                        state, pred_reward = self.dynamics_function(state, action)
                        policy_logits, value = self.prediction_function(state)
                        # scale hidden state gradient for stability
                        state = self.scale_gradient(state, 0.5)
                        t_value = target_values_batch[:, k+1]
                        t_policy = target_policies_batch[:, k+1]
                        t_reward = target_rewards_batch[:, k+1]
                        mask = legal_masks_batch[:, k+1]
                        l = self.value_loss_weight * self.categorical_loss(pred_reward, t_reward) + \
                            self.policy_loss(policy_logits, t_policy, mask) + \
                            self.value_loss_weight * self.categorical_loss(value, t_value)
                        loss += self.scale_gradient(l, gradient_scale)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite MuZero loss at training step {self.training_steps}: {loss.item()}")

                self.optimizer.zero_grad()
                # backprop
                loss.backward()

                for parameter in (
                    list(self.state_function.parameters())
                    + list(self.dynamics_function.parameters())
                    + list(self.prediction_function.parameters())
                ):
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(
                            f"Non-finite MuZero gradient at training step {self.training_steps}"
                        )

                # NOTE: AdamW also does L2 regularization via weight decay setting, see above
                self.optimizer.step()
                self.lr_scheduler.step()
                self.training_steps += 1
                update_losses.append(loss.item())

                if use_per:
                    # New priority = |predicted_value - target_value|^alpha at the
                    # sampled (step-0) position; clamped away from zero so every
                    # position keeps a nonzero sampling probability.
                    with torch.no_grad():
                        new_prio = (pred_v0 - target_v0).abs().pow(self.replay_buffer.per_alpha)
                        new_prio = new_prio.clamp_min(1e-6).cpu().numpy()
                    self.replay_buffer.update_priorities(sample_indices, new_prio)

            self.last_update_losses = update_losses
            self.last_loss = sum(update_losses) / len(update_losses)

            # revert neural nets back to evaluation mode
            self.state_function.eval()
            self.dynamics_function.eval()
            self.prediction_function.eval()

            if self.training_steps - self.last_checkpoint_step >= self.checkpoint_interval:
                self.save_model()
                self.last_checkpoint_step = self.training_steps

            # pause=input('done update()\n')
            pass
