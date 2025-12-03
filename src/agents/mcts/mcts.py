class Node(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}

        self.Q = [] # sum() to get total rewards
        self.N = 0
        pass