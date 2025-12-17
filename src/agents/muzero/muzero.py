import math

ACTION_SPACE = [a for a in range(9)]

MIN_Q = float('inf')
MAX_Q = -float('inf')

SEARCH_PATH = [] # tracks nodes traversed during simulation
ACTION_HISTORY = [] # tracks actions taken during simulation

def whose_turn():
    if len(ACTION_HISTORY) % 2 == 0:
        return 0 # player 1's turn
    else:
        return 1 # player 2's turn
    
MAX_ITERS = 800

class Node(object):
    def __init__(self, prior):
        self.state = None
        self.N, self.value_sum = 0, 0
        self.P = prior
        self.R = 0
        self.current_player = -1

        self.parent = None
        self.children = {}
        pass

    def expanded(self):
        return len(self.children.items()) > 0
    
    def mean_value(self):
        if self.N > 0:
            return  self.value_sum / self.N
        else:
            return 0
        
def update_min_max_Q(root_node):
    # TODO: keep track of min Q and max Q over entire tree
    pass

def expansion(last_node, state, reward, policy_logits):
    last_node.state = state
    last_node.reward = reward
    policy = {a: math.exp(policy_logits[a]) for a in ACTION_SPACE}
    policy_sum = sum(policy.values())
    for a in policy.keys():
        last_node.children[a] = Node(policy[a]/policy_sum)
    pass







def mcts_search(obs):
    root_node = Node(0)
    state = representation_function(obs)
    reward = 0
    policy_logits, value = prediction_function(state)
    expansion(root_node, state, reward, policy_logits)
    for _ in range(MAX_ITERS):
        last_node = selection(state)
        expansion(last_node)
        backup()