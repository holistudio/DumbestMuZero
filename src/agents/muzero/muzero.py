import torch
import torch.nn as nn
import torch.nn.functional as F

class Node(object):
    def __init__(self, state, u, policy, value, available_actions, parent=None, incoming_action=None):
        self.parent = parent
        self.children = {} # keys are actions, values are Nodes

        self.state = state
        self.reward = u
        self.policy = policy
        self.value = value

        self.N = 0

        self.legal_actions = available_actions
        self.untried_actions = available_actions
        self.incoming_action = incoming_action
        pass
    
    def sample_untried_actions(self):
        def tensor_delete(tensor, indices):
            mask = torch.ones(tensor.numel(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]
        # print(f"before: {self.untried_actions}")
        num_actions = len(self.untried_actions)
        idx = torch.randint(0, num_actions)
        a = self.untried_actions[idx].item()
        self.untried_actions = tensor_delete(self.untried_actions, idx)
        # print(f"action: {a}")
        # print(f"after: {self.untried_actions}")
        return a
    
    def is_full_expanded(self):
        if len(self.untried_actions) == 0:
            return True
        else:
            return False
        
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
    

class MCTS(nn.Module):
    def __init__(self, input_size, state_size, policy_size, hidden_size, action_size, K):
        super().__init__()
        self.rep = StateFunction(input_size, state_size, hidden_size)
        self.dynamics = DynamicsFunction(input_size, state_size, hidden_size)
        self.prediction = PredictionFunction(input_size, policy_size, hidden_size)

        self.K = K
        self.max_iters = 10
        self.action_size = action_size
        self.action_space = torch.arange(0,action_size)
        pass

    def expand(self, parent_node):
        action = parent_node.sample_untried_actions()
        return action
    
    def best_child(self, parent_node):
        # TODO: select next best child using values in each child node?
        max_node = None
        max_value = -float('inf')
        for a in sorted(parent_node.children):
            child_node = parent_node.children[a]
            value = child_node.value
            if value > max_value:
                max_value = value
                max_node = child_node
        return max_node
    
    def tree_policy(self, parent_node):
        terminal = False
        while not terminal:
            if not parent_node.is_full_expanded():
                return self.expand(parent_node)
            else:
                next_node = self.best_child(parent_node)
                if next_node is None:
                    terminal = True
        return parent_node
    
    def default_policy(self, current_node):
        for _ in range(self.K):
            action = current_node.sample_untried_actions()
            state, reward = self.dynamics(state, action)
            policy_params, value = self.prediction(state)
            new_node = Node(state, reward, policy_params, value, self.action_space, current_node, action)
            current_node = new_node
        pass
    
    def forward(self, obs):
        state = self.rep(obs)
        policy_params, value = self.prediction(state)
        current_node = Node(state, 0, policy_params, value, self.action_size)
        new_node = self.tree_policy(current_node)
        self.default_policy(new_node)
        return current_node # TODO: this will need to be modified to get loss.backward() to work

class MuZeroAgent(object):
    def __init__(self, input_size, state_size, policy_size, hidden_size, K=10, c=1.0):
        self.tree_search = MCTS(input_size, state_size, policy_size, hidden_size, K)
        self.K = K
        self.c = c # L2 regularization term
        pass

    def sample_action(self, action_probs):
        # TODO: given action probabilities, sample action
        return action
    
    def policy_function(self, root_node):
        # TODO: use policies from MCTS to produce a policy
        # and output action scores
        return action_scores
    
    def sample_policy(self, root_node, action_mask):
        action_scores = self.policy_function(root_node)
        action_scores = action_scores[action_mask]
        action_probs = torch.softmax(action_scores)
        action = self.sample_action(action_probs)
        return action
    
    def value_function(self, root_node):
        # TODO: use values predicted during MCTS to output a final value estimate
        return v
    
    def preprocess_obs(self, observation):
        # TODO: pre-process observation dictionary into tensor
        obs = observation
        return obs

    def step(self, obs, action_mask):
        obs = self.preprocess_obs(obs)
        root_node = MCTS(obs)
        action =  self.sample_policy(root_node, action_mask)
        value = self.value_function(root_node)
        # TODO: push action and value to ReplayBuffer
        return action, value
    
    
    def reward_loss(self, r, u):
        # TODO: reward loss term
        return loss
    
    def value_loss(self, v, z):
        # TODO: value loss term
        return loss
    
    def policy_loss(self, p, pi):
        # TODO: policy loss term
        return loss
    
    def loss_function(self, preds, targets, params):
        # TODO: get these to be tuples of K records from tree search + replay buffer?
        r, v, p = preds
        u, z, pi = targets

        loss = 0
        for k in range(self.K):
            l_r = self.reward_loss(r[k], u[k])
            l_v = self.value_loss(v[k], z[k])
            l_p = self.policy_loss(p[k], pi[k])
            theta = params[k] # TODO: what params go here? neural nets'?
            loss += l_r + l_v + l_p + self.c * theta # TODO: probably need L2 norm for regularization
        return  loss
    
    def store(self, u, z):
        # TODO: store recent returns from environment
        pass
    
    def update(self, u, z):
        self.store(u, z)

        # TODO: check if ReplayBuffer is full

        # TODO: loop through ReplayBuffer
        
        # TODO: loss function
        loss = loss_function(self, preds, targets, params)
        loss.backward()

        pass
        