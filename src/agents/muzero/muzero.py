import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SPACE = [a for a in range(9)]

MIN_Q = float('inf')
MAX_Q = -float('inf')

def whose_turn(action_history):
    if len(action_history) % 2 == 0:
        return 0 # player 1's turn
    else:
        return 1 # player 2's turn

def preprocess_obs(self, observation):
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

MAX_ITERS = 800
GAMMA = 0.8

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
        
def update_min_max_Q(node_mean_value):
    # keep track of min Q and max Q over entire tree
    if node_mean_value > MAX_Q:
        MAX_Q = node_mean_value
    elif node_mean_value < MIN_Q:
        MIN_Q = node_mean_value
    pass

def expansion(last_node, state, reward, policy_logits, legal_actions, action_history):
    last_node.state = state
    last_node.reward = reward
    last_node.current_player = whose_turn(action_history)

    # mask illegal actions and normalize the policy over legal moves
    policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
    policy_sum = sum(policy.values())
    for a in policy.keys():
        last_node.children[a] = Node(policy[a]/policy_sum)
    pass

def pUCT(node, sum_visits):
    # refer to Equation 2, Appendix B
    c1 = 1.25
    c2 = 19652
    Q = node.mean_value()
    if MAX_Q > MIN_Q:
        Q = (Q - MIN_Q) / (MAX_Q - MIN_Q)
    N = node.N
    P = node.P
    N_sum = sum_visits
    return (Q + (P*math.sqrt(N_sum)/(1+N))*(c1+math.log((N_sum+c2+1)/c2)))


def select_child(node):
    # parent node visit count is the sum of its children's visit counts.
    sum_visits = node.N

    best_uct = -float('inf')
    best_child = None
    for a, child_node in node.children.items():
        uct = pUCT(child_node, sum_visits)
        if uct > best_uct:
            best_uct = uct
            best_action = a
            best_child = child_node
    return best_child, best_action

def selection(node):
    search_path = [node]
    action_history = []

    while node.expanded():
        node, action = select_child(node)
        search_path.append(node)
        action_history.append(action)
    return node, search_path, action_history

def backup(value, search_path):
    L = len(search_path)
    # initialize with value estimate
    G = value
    # iterate backwards from the leaf node to the root
    for k in range(L - 1, -1, -1):
        current_node = search_path[k]
        if k < L - 1: # Not the leaf node
            # reward is from the transition *to* the next state in the path.
            reward = search_path[k+1].R
            G = reward + GAMMA * G
        current_node.value_sum += G
        current_node.N += 1
        update_min_max_Q(current_node.mean_value())
    pass

def search(obs):
    root_node = Node(0)
    action_history = []
    search_path = [root_node]
    initial_state = StateFunction(obs)
    policy_logits, value = PredictionFunction(initial_state)
    legal_actions = get_legal_actions(obs, action_history, env)
    expansion(root_node, initial_state, 0, policy_logits, legal_actions, action_history)
    backup(value, search_path) # backup the value of the root

    for _ in range(MAX_ITERS):
        last_node, search_path, action_history = selection(root_node)

        parent_node = search_path[-2]
        state, reward = DynamicsFunction(parent_node.state, action_history[-1])
        policy_logits, value = PredictionFunction(state)
        
        legal_actions = get_legal_actions(obs, action_history, env)
        expansion(last_node, state, reward, policy_logits, legal_actions, action_history)
        
        backup(value, search_path)
    return select_action(root_node)


def select_action(node):
    # TODO: revise this to sample with softmax and temperature
    max_visits = -1
    best_action = None
    for a, child_node in node.children.items():
        if child_node.N > max_visits:
            max_visits = child_node.N
            best_action = a
    return best_action

def step(observation):
    obs = preprocess_obs(observation)
    return search(obs)