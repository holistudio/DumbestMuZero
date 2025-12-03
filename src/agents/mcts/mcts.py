import math

"""functions associated with environment"""
def transition(state, action):
    # TODO: use environment somehow
    return next_state

def check_terminal(state):
    # TODO: use environment somehow
    return terminal

"""plain UCT search"""

C_p = 1/math.sqrt(2)

class Node(object):
    def __init__(self, available_actions, parent=None):
        self.parent = parent
        self.children = {} # keys are actions, values are Nodes

        self.Q = [] # sum() to get total rewards
        self.N = 0

        self.untried_actions = available_actions
        pass

    def sample_untried_actions(self):
        # TODO: randomly choose untried action
        # TODO: de-list tried action?
        return
    
def expand(parent_node, parent_state, env):
    action = parent_node.untried_actions
    next_state = transition(parent_state, action)
    next_actions = env.available_actions(next_state)
    next_node = Node(available_actions=next_actions, parent=parent_node)
    parent_node.children[action]
    return next_node, next_state

def best_child(parent_node):
    N = parent_node.N
    max_uct_node = None
    max_uct_value = - float('inf')

    for a in parent_node.children:
        child_node = parent_node.children[a]

        q = sum(child_node.Q)
        n_child = child_node.N

        if n_child == 0:
            uct_value = float('inf')
        else:
            exploitation_term = (q/n_child)
            explortation_term = 2*C_p*math.sqrt(2*math.log(N)/n_child)
            uct_value = exploitation_term + explortation_term
        
        if uct_value > max_uct_value:
            max_uct_value = uct_value
            max_uct_node = child_node
    return max_uct_node

