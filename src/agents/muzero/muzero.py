import math
import copy
import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return x

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

        # track the player whose turn it is at this node
        # ex: current_player = 0, player X's turn is at this node, player O has already made a move
        # ex: current_player = 0, player O's turn is at this node, player X has already made a move
        self.current_player = -1 

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

            td_steps = len(root_values) - ix

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
        
        self.dynamics_function = DynamicsFunction(config['state_size']+1,
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
    def get_legal_actions(self, temp_board, history):
        """
        given a game board and action history
        return indices of available legal actions
        that can be taken

        return empty list if game is over
        """
        temp_board = temp_board.clone()
        
        for depth, action in enumerate(history):
            temp_board[action] = 1 if depth % 2 == 0 else -1
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

    def whose_turn(self, action_history):
        """
        based on the environment and action history in simulation
        return the player whose turn it is in simulation
        (NOT the current player turn in the game being played)
        """
        if self.env.agent_selection == 'player_1':
            if len(action_history) % 2 == 0:
                return 0 # player 1's turn
            else:
                return 1 # player 2's turn
        if self.env.agent_selection == 'player_2':
            if len(action_history) % 2 == 0:
                return 1 # player 2's turn
            else:
                return 0 # player 1's turn

    def update_min_max_Q(self, node_mean_value):
        """
        keep track of min and max over entire search tree
        """
        if node_mean_value > self.max_Q:
            self.max_Q = node_mean_value
        elif node_mean_value < self.min_Q:
            self.min_Q = node_mean_value
        pass

    def expansion(self, last_node, state, reward, policy_logits, legal_actions, action_history):
        """
        expansion phase of MuZero search
        add child nodes to a given leaf node based on legal actions
        and initialize them with policy priors based on the prediction network
        """

        last_node.state = state
        last_node.R = reward
        last_node.current_player = self.whose_turn(action_history)

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
            # child values are stored from player's perspective
            # the parent is the opposing player, so negate value
            # before using to select parent action
            Q = -node.mean_value()
            if self.max_Q > self.min_Q:
                # revese normalization bounds as well
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
        # value starts from the leaf player's perspective
        # rewards from the perspective of the player who acted at the parent node
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
        noise = torch.distributions.Dirichlet(torch.full((len(actions),), self.dirichlet_alpha)).sample()
        for a, n in zip(actions, noise):
            node.children[a].P = node.children[a].P * (1 - root_exploration_fraction) + n * root_exploration_fraction
            
    def search(self, obs, temperature):
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
                latest_action = torch.tensor([[action_history[-1]]], dtype=torch.long, device=self.device)

                # dynamics function predicts next state and immediate reward
                state, reward = self.dynamics_function(parent_node.state, latest_action)

                # prediciton function estimates policy logits and value based on next state
                policy_logits, value = self.prediction_function(state)

                # get legal actions avalaible at the leaf node given previous actions
                legal_actions = self.get_legal_actions(obs, action_history)

                # expand leaf node with child nodes for each legal action
                self.expansion(last_node, state, reward.item(), policy_logits, legal_actions, action_history)
                
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
                leaf_value = value.item() if legal_actions else 0
                self.backup(leaf_value, search_path)
                
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
        # If terminal, the value of the state is 0. Using self.root_value would be stale/incorrect.
        # val = 0 if terminal else self.root_value
        self.replay_buffer.store_step(obs, player_turn, action, self.action_probs.clone(), reward, self.root_value)
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
        # blanks = torch.where(obs == 0)[0].tolist()
        # if len(blanks) > 5:
        #     action = self.search(obs, self.current_temperature())
        # else:
        #     action = self.search(obs, 0.0)
        
        action = self.search(obs, self.current_temperature())

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
        action = self.search(obs, 0.0)
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
                t_reward = target_rewards_batch[:, 0].unsqueeze(1)
                mask = legal_masks_batch[:, 0]
                
                # mask illegal moves from predicted policy 
                policy_logits = policy_logits.masked_fill(~mask, -float('inf'))
                
                # predicted_reward is 0 for initial step (no dynamics yet)
                pred_reward_0 = torch.zeros_like(t_reward)
                
                # compute loss for step 0
                loss = F.mse_loss(pred_reward_0, t_reward) + \
                       F.cross_entropy(policy_logits, t_policy) + \
                       F.mse_loss(value, t_value)
                
                # gradient scale for recurrent backpropagation through time stable training
                gradient_scale = 1.0 / self.k_unroll_steps
                
                # unroll steps
                for k in range(self.k_unroll_steps):
                    # get actions for step k for the whole batch: (Batch, 1)
                    action = actions_batch[:, k].unsqueeze(1)
                    
                    # dynamics and prediction function on batch
                    state, pred_reward = self.dynamics_function(state, action)
                    policy_logits, value = self.prediction_function(state)
                    
                    # scale hidden state gradient for stability
                    state = self.scale_gradient(state, 0.5)
                    
                    # get get batch targets
                    t_value = target_values_batch[:, k+1].unsqueeze(1)
                    t_policy = target_policies_batch[:, k+1]
                    t_reward = target_rewards_batch[:, k+1].unsqueeze(1)
                    mask = legal_masks_batch[:, k+1]
                    
                    # mask illegal moves from predicted policy 
                    policy_logits = policy_logits.masked_fill(~mask, -float('inf'))
                    
                    # compute loss w.r.t each target and prediction
                    l = F.mse_loss(pred_reward, t_reward) + \
                        F.cross_entropy(policy_logits, t_policy) + \
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
