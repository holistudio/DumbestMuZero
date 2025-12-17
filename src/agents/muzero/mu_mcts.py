import math
import random


ACTION_SPACE = [a for a in range(9)]

MAX_ITERS = 800 # number of simulations for each MCTS
L = 10 # search depth

MIN_Q = float('inf')
MAX_Q = -float('inf')

STATE_TRANSITION_LOOKUP = {}
# STATE_TRANSITION_LOOKUP = {
#     "state":
#     {
#         "action":
#         {
#             "next_node": TreeNode(),
#             "reward": r
#         }
#     }
# }



class TreeEdge(object):
    def __init__(self, parent_node, action, P):
        self.parent_node = parent_node
        self.action = action

        self.N, self.Q = 0, 0
        self.P = P
        self.S, self.R = None, 0
        pass

class TreeNode(object):
    def __init__(self, state, policy, action_space, parent=None, edge=None):
        self.state = state
        self.edges = {} # key: action, value: TreeEdge()
        for a in action_space:
            self.edges[a] = TreeEdge(self, a, policy)
            self.parent = parent
        self.edge = edge
        pass
    
    def expanded(self):
        sum_visits = 0
        for a in self.edges:
            sum_visits +=  self.edges[a].N
        if sum_visits > 0:
            return True
        return False

def state_transtion(state, action):
    # TODO: use STATE_TRANSITION_LOOKUP somehow
    return next_node, reward

def update_min_max_Q(root_node):
    # TODO: keep track of min Q and max Q over entire tree
    pass

def pUCT_rule(edge, sum_visits):
	c1 = 1.25
	c2 = 19652
	Q = edge.Q - MIN_Q / (MAX_Q - MIN_Q)
	P = edge.P
	N_sum = sum_visits
	N = edge.N
	return (Q + (P*math.sqrt(N_sum)/(1+N))*(c1+math.log((N_sum+c2+1)/c2)))

def selection(node):
    # for k in range(L-2):
    while node.expanded():
        sum_visits = 0
        for a in node.edges:
            edge = node.edges[a]
            sum_visits += edge.N
        
        if sum_visits != 0:
            next_action = None
            next_edge = None
            max_ucb = - float('inf')
            for a in node.edges:
                edge = node.edges[a]
                ucb = pUCT_rule(edge, sum_visits)
                if ucb > max_ucb:
                    next_edge = edge
                    next_action = a
            next_node, reward = state_transition(node.state, next_action)
            node = next_node
    return node

def expansion(last_node):
    state = last_node.state

    # record state and reward transtion in lookup table
            
        

def mcts_search(obs):
    state = representation_function(obs)
    policy, value = prediction_function(state)
    root_node = TreeNode(state, policy, ACTION_SPACE)
    for _ in range(MAX_ITERS):
        last_node = selection(root_node)
        expansion(last_node)
        backup()