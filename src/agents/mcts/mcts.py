import math

"""functions associated with environment"""
def transition(state, action):
    # TODO: use environment somehow
    return next_state

def available_actions(state):
    # TODO: use environment somehow
    return actions

def check_terminal(state):
    # TODO: use environment somehow
    return terminal

"""plain UCT search"""

C_p = 1/math.sqrt(2)

class Node(object):
    def __init__(self, available_actions, parent=None, incoming_action=None):
        self.parent = parent
        self.children = {} # keys are actions, values are Nodes

        self.Q = [] # sum() to get total rewards
        self.N = 0

        self.untried_actions = available_actions
        self.incoming_action = incoming_action
        pass

    def sample_untried_actions(self):
        # TODO: randomly choose untried action
        # TODO: de-list tried action?
        return
    
    def is_full_expanded(self):
        if len(self.untried_actions) == 0:
            return True
        else:
            return False
    
def expand(parent_node, parent_state):
    action = parent_node.untried_actions
    next_state = transition(parent_state, action)
    next_actions = available_actions(next_state)
    next_node = Node(available_actions=next_actions, parent=parent_node, incoming_action=action)
    parent_node.children[action] = next_node
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

def tree_policy(parent_node, parent_state):
    while not check_terminal(parent_state):
        if not parent_node.is_full_expanded:
            return expand(parent_node, parent_state)
        else:
            parent_node = best_child(parent_node)
            a = parent_node.incoming_action
            parent_state = transition(parent_state, a)
    return parent_node, parent_state