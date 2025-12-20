import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

"""ENIVORNMENT HELPER FUNCTIONS"""



def flatten(observation_space):
    # TODO: return observation space as a 1-D vector
    return (9,)

def preprocess_obs(observation):
    # pre-process observation dictionary into tensor
    obs = torch.zeros((3,3), dtype=torch.float32)
    current_player_plane = torch.tensor(observation["observation"][:, :, 0])
    opponent_plane = torch.tensor(observation["observation"][:, :, 1]) * 2
    obs = obs + current_player_plane + opponent_plane
    obs = obs.reshape((9,1)).squeeze()
    return obs

def get_legal_actions(observation, history, env):
    if len(history) > 0:
        for a in history:
            observation = env.transition(observation, a)
    return env.available_actions(observation)

class StateFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)
        pass

    def forward(self, obs):
        x = self.lin1(obs)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        x = self.lin3(x)
        return x

class DynamicsFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.state_head = nn.Linear(hidden_size, output_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        pass

    def forward(self, s_prev, a):
        x = torch.cat((s_prev, a))
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        s = self.state_head(x)
        r = self.reward_head(x)
        return s, r
    
class PredictionFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)
        pass

    def forward(self, s):
        x = self.lin1(s)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

class Node(object):
    def __init__(self, prior):
        self.state = None
        self.N, self.value_sum = 0, 0
        self.P = prior
        self.R = 0
        self.current_player = -1

        self.children = {}
        pass

    def expanded(self):
        return len(self.children.items()) > 0
    
    def mean_value(self):
        if self.N > 0:
            return  self.value_sum / self.N
        else:
            return 0

"""REPLAY BUFFER"""

class ReplayBuffer(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer = [] # store each game's trajectory

        # TRAJECTORY = zip(observations, player_turns, actions, immediate_rewards, target_policies, final_outcomes)
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.immediate_rewards = []
        self.final_outcomes = []
        pass

    def step_count(self):
        count = 0
        for trajectory in self.buffer:
            count += len(trajectory)
        return count
    
    def reset_trajectory(self):
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.immediate_rewards = []
        self.final_outcomes = []
        pass

    def store_trajectory(self):
        trajectory = zip(self.observations, 
                         self.player_turns, self.actions, 
                         self.immediate_rewards, # dynamics function target
                         self.target_policies, self.final_outcomes) # prediction function targets
        trajectory = copy.deepcopy(trajectory)
        self.buffer.append(trajectory)

        self.reset_trajectory()
        pass

    def sample_batch(self, k_unroll_steps, td_steps):
        # TODO: return batches of trajectories
        # each of length k_unroll_steps
        batch = []

        for b in self.batch_size:
            # td_steps, n steps into the future for target_value
            # k_unroll_steps, capped k steps in trajectory for training
            
            # inputs
            # observations (but only first observation is used)
            # player_turns
            # actions

            # targets
            # immediate_reward
            # target_policy
            # target_value # same as final outcome for board games OR discounted from final_outcome based on td_steps
            sequence = zip(inputs, targets)
            batch.append(sequence)
        return batch


class MuZeroAgent(object):
    def __init__(self, env, config):
        self.env = env
        self.observation_space = flatten(env.observation_space)
        self.obs_size = self.observation_space.shape
        self.action_space = env.action_space

        self.replay_buffer = ReplayBuffer(config['batch_size'])
        self.buffer_size = config['buffer_size']

        self.state_function = StateFunction(self.obs_size[0],
                                            config['state_size'],
                                            config['hidden_size'])
        
        self.dynamics_function = DynamicsFunction(self.obs_size[0]+1,
                                                  config['state_size'],
                                                  config['hidden_size'])
        
        self.prediction_function = PredictionFunction(config['state_size'],
                                                      self.action_space.shape[0],
                                                      config['hidden_size'])
        
        # TODO: somehow "wrap" all three neural nets into one model for easier turning on/off model.train()
        self.model = todo()
        # TODO: three neural nets' weights need to be given to the optimizer
        self.optimizer = torch.optim.AdamW(all_function_params)

        self.min_Q = float('inf')
        self.max_Q = -float('inf')
        self.max_iters = config['max_iters']
        self.gamma = config['gamma']
        pass

    def update_min_max_Q(self, node_mean_value):
        # keep track of min Q and max Q over entire tree
        if node_mean_value > self.max_Q:
            self.max_Q = node_mean_value
        elif node_mean_value < self.min_Q:
            self.min_Q = node_mean_value
        pass

    def whose_turn(self, env, action_history):
        # based on the environment and action history in simulation
        # NOT the current player index in the game being played
        if env.agent_selection == 'player_1':
            if len(action_history) % 2 == 0:
                return 0 # player 1's turn
            else:
                return 1 # player 2's turn
        if env.agent_selection == 'player_2':
            if len(action_history) % 2 == 0:
                return 1 # player 2's turn
            else:
                return 0 # player 1's turn
        
    def expansion(self, last_node, state, reward, policy_logits, legal_actions, action_history):
        last_node.state = state
        last_node.reward = reward
        last_node.current_player = self.whose_turn(self.env, action_history)

        # mask illegal actions and normalize the policy over legal moves
        policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for a in policy.keys():
            last_node.children[a] = Node(policy[a]/policy_sum)
        pass

    def pUCT(self, node, sum_visits):
        # refer to Equation 2, Appendix B
        c1 = 1.25
        c2 = 19652
        Q = node.mean_value()
        if self.max_Q > self.min_Q:
            Q = (Q - self.min_Q) / (self.max_Q - self.min_Q)
        N = node.N
        P = node.P
        N_sum = sum_visits
        return (Q + (P*math.sqrt(N_sum)/(1+N))*(c1+math.log((N_sum+c2+1)/c2)))

    def select_child(self, node):
        # parent node visit count is the sum of its children's visit counts.
        sum_visits = node.N

        best_uct = -float('inf')
        best_child = None
        for a, child_node in node.children.items():
            uct = self.pUCT(child_node, sum_visits)
            if uct > best_uct:
                best_uct = uct
                best_action = a
                best_child = child_node
        return best_child, best_action

    def selection(self, node):
        search_path = [node]
        action_history = []

        while node.expanded():
            node, action = self.select_child(node)
            search_path.append(node)
            action_history.append(action)
        return node, search_path, action_history

    def backup(self, value, search_path):
        L = len(search_path)
        # initialize with value estimate
        G = value
        # iterate backwards from the leaf node to the root
        for k in range(L - 1, -1, -1):
            current_node = search_path[k]
            if k < L - 1: # when not the leaf node
                reward = search_path[k+1].R
                G = reward + self.gamma * G
            current_player = 0 if self.env.agent_selection == 'player_1' else 1
            current_node.value_sum += G if current_node.current_player == current_player else -G
            current_node.N += 1
            self.update_min_max_Q(current_node.mean_value())
        pass

    def select_action(self, node):
        # TODO: revise this to sample with softmax and temperature
        max_visits = -1
        best_action = None
        for a, child_node in node.children.items():
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_action = a
        return best_action

    def search(self, obs):
        root_node = Node(0)
        action_history = []
        search_path = [root_node]
        initial_state = self.state_function(obs)
        policy_logits, value = self.prediction_function(initial_state)
        legal_actions = get_legal_actions(obs, action_history, self.env)
        self.expansion(root_node, initial_state, 0, policy_logits, legal_actions, action_history)
        self.backup(value, search_path) # backup the value of the root

        for _ in range(self.max_iters):
            last_node, search_path, action_history = self.selection(root_node)

            parent_node = search_path[-2]
            state, reward = self.dynamics_function(parent_node.state, action_history[-1])
            policy_logits, value = self.prediction_function(state)
            
            legal_actions = get_legal_actions(obs, action_history, self.env)
            self.expansion(last_node, state, reward, policy_logits, legal_actions, action_history)
            
            self.backup(value, search_path)
        return self.select_action(root_node)

    def store_trajectory_step(self, obs, player_turn, action, immediate_reward, final_outcome):
        # TODO: call this within environment training loop after env.step(action)

        # observation vector
        self.replay_buffer.observations.append(obs)

        # player turn
        self.replay_buffer.player_turns.append(player_turn)

        # action direct from what was chosen by select_action (different from child visits if sampled)
        self.replay_buffer.actions.append(action)

        # somehow get the immediate_reward from environment
        self.replay_buffer.immediate_rewards.append(immediate_reward)
        
        if final_outcome != 0:
            # once final_outcome is nonzero, label the entire trajectory with the final outcome 
            # TODO: +/- based on player_turn
            # TODO: maybe with a discount factor???
            self.replay_buffer.final_outcomes = [final_outcome if i == 'player_1' else -final_outcome for i in self.replay_buffer.player_turns]
            self.replay_buffer.store_trajectory()
        pass

    def step(self, observation):
        obs = preprocess_obs(observation)
        action = self.search(obs)
        return action

    def update(self):
        if self.replay_buffer.step_count() > self.buffer_size:
            model.train()
            # load from batch
            batch = self.replay_buffer.sample_batch()
            for sequence in batch:
                inputs, targets = sequence

                obs, actions = inputs
                
                # neural nets predict
                # predicted_reward
                # policy_logits
                # predicted_value
                state = self.state_function(obs[0])
                predicted_reward = 0
                policy_logits, value = self.prediction_function(state)
                action_history = []
                predictions = [(policy_logits, predicted_reward, value)]
                for a in actions:
                    state, predicted_reward  = self.dynamics_function(state, a)
                    policy_logits, value = self.prediction_function(state)
                    legal_actions = get_legal_actions(obs, action_history, self.env)
                    policy_logits = [policy_logits[i] if i in legal_actions else 0 for i in self.action_space]
                    predictions.append((policy_logits, predicted_reward, value))
                    action_history.append(a)
                
                # loss
                # TODO: add gradient scale
                l = 0
                for i, pred in enumerate(predictions):
                    policy_logits, predicted_reward, value = pred
                    u, target_policy, target_value = targets[i]

                    
                    # compare with corresponding
                    # immediate_reward, MSE
                    # target_policy, cross_entropy
                    # target_value, MSE
                    l += F.mse_loss(predicted_reward, u) + F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)
            loss = l + regularization(weights)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            model.eval()