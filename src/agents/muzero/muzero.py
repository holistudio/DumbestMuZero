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
    min-max normalize hidden state to [0, 1] elementwise per row,
    matching the action input range (Appendix G)
    """
    s_min = state.min(dim=-1, keepdim=True)[0]
    s_max = state.max(dim=-1, keepdim=True)[0]
    scale = s_max - s_min
    scale = torch.where(scale < 1e-5, scale + 1e-5, scale)
    return (state - s_min) / scale

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.apply(self._init_weights)
        pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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
        s = min_max_normalize(x)
        return s

class DynamicsFunction(nn.Module):
    """
    dynamics function, g
    
    input: hidden state representation, s_t, and candidate action, a
    output: predict next hidden state, s_t+1, and reward, r_t+1
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.state_head = nn.Linear(hidden_size, output_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)
        torch.nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.reward_head.bias)
        pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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
        s = min_max_normalize(s)
        r = self.reward_head(x)
        return s, r
    
class PredictionFunction(nn.Module):
    """
    prediction function, f
    
    input: hidden state representation, s_t
    output: policy logits, p_t, and value, v_t
    """
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)
        torch.nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.policy_head.bias)
        torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.value_head.bias)
        pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

        # track the player whose turn it is at this node
        # ex: to_play = 0, player X's turn is at this node, player O has already made a move
        # ex: to_play = 1, player O's turn is at this node, player X has already made a move
        self.to_play = -1

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
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

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
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        self.reset_trajectory()
        pass

    def sample_batch(self, k_unroll_steps, gamma, device):
        # Prepare lists to collect batch data
        obs_batch = []
        actions_batch = []
        target_rewards = []
        target_values = []
        target_policies_batch = []
        legal_masks = []

        num_eps = len(self.buffer)
        for b in range(self.batch_size):
            random_ep = self.buffer[random.randint(0, num_eps-1)]  
            # random_ep = self.buffer[random.randint(num_eps // 2, num_eps-1)]  # TODO: look into this "front-half-most-recent games" buffer lookup...
            observations, player_turns, actions = random_ep['obs'], random_ep['turns'], random_ep['actions']
            rewards, target_policies, root_values = random_ep['rewards'], random_ep['target_policies'], random_ep['root_values']
            
            if len(root_values) - k_unroll_steps -1 > 0:
                ix = random.randint(0, len(root_values) - k_unroll_steps -1)
            else:
                ix = 0

            # last record is the terminal state; its reward is the game outcome
            # from that record's player's perspective (-1 loss, 0 draw)
            outcome = rewards[-1]
            terminal_player = player_turns[-1]

            # td_steps = len(root_values) - ix
            # td_steps = k_unroll_steps

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
                
                # board-game convention: no intermediate rewards, discount 1,
                # no bootstrapping. the terminal record carries the outcome from
                # its own player's perspective; sign it per state.
                if current_player is None:
                    value = 0.0                      # absorbing state past the terminal
                elif current_player == terminal_player:
                    value = outcome
                else:
                    value = -outcome

                # reward head is unused for board games
                sample_rewards.append(0.0)
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
                    # Absorbing state: zero soft target policy
                    sample_policies.append(torch.zeros_like(target_policies[0]))
            
            target_rewards.append(sample_rewards)
            target_values.append(sample_values)
            target_policies_batch.append(torch.stack(sample_policies))
            legal_masks.append(torch.stack(sample_masks))

        # Stack into tensors and move to device once
        return (torch.stack(obs_batch).to(device),
                torch.tensor(actions_batch, dtype=torch.long).to(device),
                torch.tensor(target_rewards, dtype=torch.float32).to(device),
                torch.tensor(target_values, dtype=torch.float32).to(device),
                torch.stack(target_policies_batch).to(device),
                torch.stack(legal_masks).to(device))

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
        self.replay_buffer = ReplayBuffer(config['buffer_size'], config['batch_size'])
        self.buffer_size = config['buffer_size']
        self.min_replay_size = config['min_replay_size']

        # neural networks
        self.state_function = StateFunction(self.obs_size[0],
                                            config['state_size'],
                                            config['hidden_size'])
        
        self.dynamics_function = DynamicsFunction(config['state_size']+self.action_size,
                                                  config['state_size'],
                                                  config['hidden_size'])
        
        self.prediction_function = PredictionFunction(config['state_size'],
                                                      self.action_size,
                                                      config['hidden_size'])
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

        self.min_Q = float('inf')
        self.max_Q = -float('inf')
        self.max_iters = config['max_iters']
        self.train_iters = config['train_iters']
        self.gamma = config['gamma']
        self.k_unroll_steps = config['k_unroll_steps']

        self.root_value = 0
        self.action_probs = torch.zeros(self.action_size)
        self.temperature = config['temperature']
        self.episodes_played = 0 # temperature schedule based on episodes
        self.temp_schedule = config.get('temp_schedule', None)
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.root_exploration_fraction = config['root_exploration_fraction']


        self.state_function.eval()
        self.dynamics_function.eval()
        self.prediction_function.eval()

        if load:
            self.load_model()
        pass
    
    """model utilities"""
    def save_model(self):
        """
        save neural network and optimizer parameters
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
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

    def load_model(self):
        """
        load neural network and optimizer parameters
        """

        # TODO: allow loading from a specified directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        paths = {
            'state': os.path.join(base_dir, 'mu_state_rep_params.pth.tar'),
            'dynamics': os.path.join(base_dir, 'mu_dyn_func_params.pth.tar'),
            'prediction': os.path.join(base_dir, 'mu_pred_func_params.pth.tar'),
            'optimizer': os.path.join(base_dir, 'mu_optimizer_params.pth.tar')
        }

        if os.path.exists(paths['state']):
            self.state_function.load_state_dict(torch.load(paths['state'], map_location=self.device))
        if os.path.exists(paths['dynamics']):
            self.dynamics_function.load_state_dict(torch.load(paths['dynamics'], map_location=self.device))
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

    def preprocess_obs(self, observation):
        """
        convert environment observation dictionary 
        into a canonical (player-relative) torch tensor for neural nets
        """
        current_player_plane = torch.tensor(observation["observation"][:, :, 0], dtype=torch.float32)
        opponent_plane = torch.tensor(observation["observation"][:, :, 1], dtype=torch.float32)
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

    def update_min_max_Q(self, node_mean_value):
        """
        keep track of min and max over entire search tree
        """
        if node_mean_value > self.max_Q:
            self.max_Q = node_mean_value
        if node_mean_value < self.min_Q:
            self.min_Q = node_mean_value
        pass

    def expansion(self, last_node, state, reward, policy_logits, actions):
        """
        expansion phase of MuZero search
        add child nodes to a given leaf node based on legal actions
        and initialize them with policy priors based on the prediction network
        """

        last_node.state = state
        last_node.R = reward

        # Since networks now process batches, squeeze the batch dimension for single inferences
        policy_logits = policy_logits.squeeze(0)

        # mask illegal actions and normalize the policy over legal moves
        # Use .item() to convert tensor logits to float for math.exp
        # Subtract max logit for numerical stability to prevent ZeroDivisionError
        max_logit = max(policy_logits[a].item() for a in actions)
        policy = {a: math.exp(policy_logits[a].item() - max_logit) for a in actions}
        policy_sum = sum(policy.values())
        for a in actions:
            child = Node(policy[a]/policy_sum)
            child.to_play = 1 - last_node.to_play
            last_node.children[a] = child
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
            # child values are stored from player's perspective
            # the parent is the opposing player, so negate value
            # before using to select parent action
            # normalize the full backed-up action value R + gamma*V,
            # not just the raw mean value (Eq. 5 / Appendix B)
            Q = node.R + self.gamma * (-node.mean_value())
            if self.max_Q > self.min_Q:
                Q = (Q - self.min_Q) / (self.max_Q - self.min_Q)
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
        to_play = search_path[-1].to_play
        G = value
        for i in range(len(search_path) - 1, -1, -1):
            current_node = search_path[i]
            current_node.value_sum += G if current_node.to_play == to_play else -G
            current_node.N += 1
            # track the range of the same backed-up quantity used in pUCT:
            # R + gamma*V, not the raw mean value
            self.update_min_max_Q(current_node.R + self.gamma * (-current_node.mean_value()))
            if i > 0:
                parent = search_path[i - 1]
                R = current_node.R
                G = (R if parent.to_play == to_play else -R) + self.gamma * G
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
        actions = list(node.children.keys())
        if not actions:
            return
        noise = torch.distributions.Dirichlet(torch.full((len(actions),), self.dirichlet_alpha)).sample()
        for a, n in zip(actions, noise):
            node.children[a].P = node.children[a].P * (1 - self.root_exploration_fraction) + n * self.root_exploration_fraction
            
    def search(self, obs, temperature, add_noise=True):
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
            # value normalization statistics are local to one search tree
            self.min_Q = float('inf')
            self.max_Q = -float('inf')

            # initialize tree root node
            # to_play is only meaningful relative to other nodes in this search tree,
            # so 0 is an arbitrary but consistent reference point for this call
            root_node = Node(0)
            root_node.to_play = 0

            # encode observation into a hidden state represnetation
            initial_state = self.state_function(obs.to(self.device).unsqueeze(0))

            # predict initial policy logits and value
            policy_logits, value = self.prediction_function(initial_state)

            # get list of current available legal actions
            root_actions = torch.where(obs == 0)[0].tolist()

            # expand root node
            self.expansion(root_node, initial_state, 0, policy_logits, root_actions)
            
            # add exploration noise to root node's children
            if add_noise:
                self.add_exploration_noise(root_node)

            for i in range(self.max_iters):
                # select leaf node
                last_node, search_path, action_history = self.selection(root_node)

                # get leaf node's parent
                parent_node = search_path[-2]

                # get latest candidate action as a tensor
                latest_action = F.one_hot(
                    torch.tensor([action_history[-1]], device=self.device),
                    num_classes=self.action_size,
                ).float()  # shape (1, action_size)

                # dynamics function predicts next state
                # reward output is unused and the reward head
                # is never trained
                state, _ = self.dynamics_function(parent_node.state, latest_action)

                # prediciton function estimates policy logits and value based on next state
                policy_logits, value = self.prediction_function(state)

                # expand leaf node with child nodes for each legal action
                # board games have no intermediate rewards
                self.expansion(last_node, state, 0.0, policy_logits, list(range(self.action_size)))
                
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
                self.backup(value.item(), search_path)
                
                # --- DEBUG START ---
                # print("Path AFTER Backup:")
                # for depth, node in enumerate(search_path):
                #     print(f"  Depth {depth} | N: {node.N} | V_sum: {float(node.value_sum):.4f} | Mean V: {float(node.mean_value()):.4f}")
                # --- DEBUG END ---

                # pause = input('end of one search simulation\n')
            
            # store the mean value of the root node
            self.root_value = root_node.mean_value()

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
        # an absorbing state has value 0 and no policy
        if action is None:
            value = 0.0                                # absorbing-state bootstrap value
            policy = torch.zeros(self.action_size)     # absorbing-state policy target
        else:
            value = self.root_value
            policy = self.action_probs.clone()
        self.replay_buffer.store_step(obs, player_turn, action, policy, reward, value)
        if terminal:
            # once final_outcome is nonzero, label the entire trajectory with the final outcome 
            # self.replay_buffer.final_outcomes = [final_outcome if i == player_turn else -final_outcome for i in self.replay_buffer.player_turns]
            self.replay_buffer.store_trajectory()

            # after game is over advance the temperature schedule
            self.episodes_played += 1
        pass

    def current_temperature(self):
        """
        get current softmax sampling temperature
        based on constant or variable temperature annealing schedule
        """
        if not self.temp_schedule:
            return self.temperature
        for threshold, temp in self.temp_schedule:
            if self.episodes_played < threshold:
                return temp
        return self.temp_schedule[-1][1]
    
    def step(self, observation):
        """
        select next action to play in the game
        with softmax temperature sampling
        """
        obs = self.preprocess_obs(observation)
        
        # simple temperature annealing for tic-tac-toe:
        # if both players have placed two pieces (5 or fewer blank spaces), 
        # there is likely a single move to exploit
        # that is best to block of connect 3 in a row
        blanks = torch.where(obs == 0)[0].tolist()
        if len(blanks) > 5:
            action = self.search(obs, self.current_temperature())
        else:
            action = self.search(obs, 0.0)
        
        # action = self.search(obs, self.current_temperature())

        return action
    
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
        action = self.search(obs, 0.0, add_noise=False)
        return action

    def scale_gradient(self, tensor, scale):
        """
        scale gradients for stability during recurrent network training
        """
        return tensor * scale + tensor.detach() * (1.0 - scale)

    def update(self):
        """
        update neural network parameters
        """
        # print('update()')

        # update weights only after replay buffer is full
        if len(self.replay_buffer.buffer) >= self.min_replay_size:
            # set neural nets to training mode
            self.state_function.train()
            self.dynamics_function.train()
            self.prediction_function.train()

            for epoch in range(self.train_iters):
                loss = 0
                
                # sample batch as stacked tensors
                obs_batch, actions_batch, target_rewards_batch, target_values_batch, target_policies_batch, legal_masks_batch = \
                    self.replay_buffer.sample_batch(self.k_unroll_steps, self.gamma, self.device)
                
                # process entire batch of observations at once
                state = self.state_function(obs_batch)
                policy_logits, value = self.prediction_function(state)
                
                # get batch targets for step 0
                t_value = target_values_batch[:, 0].unsqueeze(1)
                t_policy = target_policies_batch[:, 0]

                # mask illegal moves from predicted policy
                # mask = legal_masks_batch[:, 0]
                # policy_logits = policy_logits.masked_fill(~mask, -float('inf'))

                # compute loss for step 0
                # board games have no intermediate rewards, so the reward
                # prediction loss is omitted (Appendix G)
                loss = F.cross_entropy(policy_logits, t_policy) + \
                       F.mse_loss(value, t_value)
                
                # gradient scale for recurrent backpropagation through time stable training
                gradient_scale = 1.0 / self.k_unroll_steps
                
                # unroll steps
                for k in range(self.k_unroll_steps):
                    # get actions for step k for the whole batch
                    action = F.one_hot(actions_batch[:, k], num_classes=self.action_size).float()  # (batch, action_size)
                    
                    # dynamics and prediction function on batch
                    # reward output unused: board games have no intermediate
                    # rewards, so the reward prediction loss is omitted (Appendix G)
                    state, _ = self.dynamics_function(state, action)
                    policy_logits, value = self.prediction_function(state)
                    
                    # scale hidden state gradient for stability
                    state = self.scale_gradient(state, 0.5)
                    
                    # get get batch targets
                    t_value = target_values_batch[:, k+1].unsqueeze(1)
                    t_policy = target_policies_batch[:, k+1]
                    
                    # mask illegal moves from predicted policy 
                    # mask = legal_masks_batch[:, k+1]
                    # policy_logits = policy_logits.masked_fill(~mask, -float('inf'))
                    
                    # compute loss w.r.t each target and prediction
                    l = F.cross_entropy(policy_logits, t_policy) + \
                        F.mse_loss(value, t_value)
                    
                    # scale loss for stability
                    loss += self.scale_gradient(l, gradient_scale)

                self.optimizer.zero_grad()

                # backprop
                loss.backward()

                 # NOTE: AdamW also does L2 regularization via weight decay setting, see above
                self.optimizer.step()

            # revert neural nets back to evaluation mode
            self.state_function.eval()
            self.dynamics_function.eval()
            self.prediction_function.eval()

            # save neural net and optimizer parameters
            self.save_model()

            # pause=input('done update()\n')
            pass
