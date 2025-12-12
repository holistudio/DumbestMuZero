class TreeEdge(object):
    def __init__(self, parent_node, action, P):
        self.N, self.Q = 0, 0
        self.P = P
        self.S, self.R = None, 0
        pass

class TreeNode(object):
    def __init__(self, state, policy, action_space, parent=None, incoming_action=None):
        self.state = state
        self.edges = {} # key: action, value: TreeEdge()
        for a in action_space:
            self.edge[a] = TreeEdge(self, a, policy)
            self.parent = parent
        self.incoming_action = incoming_action
        pass