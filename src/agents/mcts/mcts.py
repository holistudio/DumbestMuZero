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

def transition(state, action):
    # TODO: use environment somehow
    return next_state

def expand(parent_node, parent_state, env):
    action = parent_node.untried_actions
    next_state = transition(parent_state, action)
    next_actions = env.available_actions(next_state)
    next_node = Node(available_actions=next_actions, parent=parent_node)
    parent_node.children[action]
    return next_node, next_state