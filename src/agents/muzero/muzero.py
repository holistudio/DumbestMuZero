import torch
import torch.nn as nn
import torch.nn.functional as F

def preprocess_obs(observation):
    # TODO: pre-process observation dictionary into tensor
    obs = observation
    return obs

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
    def __init__(self, input_size, state_size, policy_size, hidden_size, K):
        super().__init__()
        self.rep = StateFunction(input_size, state_size, hidden_size)
        self.dynamics = DynamicsFunction(input_size, state_size, hidden_size)
        self.prediction = PredictionFunction(input_size, policy_size, hidden_size)

        self.K = K
        pass

    def forward(self, obs):
        state = self.rep(obs)
        tree = {} # TODO: figure out tree structure for exploring actions and tracking function outputs
        for _ in range(self.K):
            # TODO: this loop needs to use something akin to tree policy and default policy, probably
            action = todo() # TODO: sample available and unxplored actions only 
            state, reward = self.dynamics(state, action)
            policy_params, value = self.prediction(state)
        return tree # TODO: this will need to be modified to get loss.backward() to work

class MuZeroAgent(object):
    def __init__(self, input_size, state_size, policy_size, hidden_size, K=10, c=1.0):
        self.tree_search = MCTS(input_size, state_size, policy_size, hidden_size, K)
        self.K = K
        self.c = c # L2 regularization term
        pass

    def policy_function(self, tree):
        # TODO: use policies from MCTS to produce a policy
        # and output action scores
        return action_scores
    
    def sample_policy(self, tree, action_mask):
        action_scores = policy_function(tree)
        action_scores = action_scores[action_mask]
        action_probs = torch.softmax(action_scores)
        action = sample(action_probs)
        return action
    
    def value_function(self, tree):
        # TODO: use values predicted during MCTS to output a final value estimate
        return v

    def step(self, obs, action_mask):
        tree = MCTS(obs)
        action =  self.sample_policy(tree, action_mask)
        value = self.value_function(tree)
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
            loss += l_r + l_v + l_p + self.c * theta # TODO: probably need L2 norm
        return  loss
    
    def store(self, u, z):
        pass
    
    def update(self, u, z):
        # TODO: store recent returns from environment
        self.store(u, z)

        # TODO: check if ReplayBuffer is full

        # TODO: loop through ReplayBuffer
        
        # TODO: loss function
        loss = loss_function(self, preds, targets, params)
        loss.backward()

        pass
        