import math
import random


"""plain UCT search"""

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
        # print(f"before: {self.untried_actions}")
        num_actions = len(self.untried_actions)
        idx = random.randint(0,num_actions-1)
        a = self.untried_actions.pop(idx)
        # print(f"action: {a}")
        # print(f"after: {self.untried_actions}")
        return a
    
    def is_full_expanded(self):
        if len(self.untried_actions) == 0:
            return True
        else:
            return False
    

class UCTAgent(object):
    def __init__(self, environment, C_p=0.7, max_iters=100000):
        self.C_p = C_p

        self.iter = 0
        self.max_iters = max_iters

        self.env = environment
        pass

    def expand(self, parent_node, parent_state):
        # print('### EXPANDING')
        action = parent_node.sample_untried_actions()
        next_state = self.env.transition(parent_state, action)
        next_actions = self.env.available_actions(next_state)
        next_node = Node(available_actions=next_actions, parent=parent_node, incoming_action=action)
        parent_node.children[action] = next_node
        return next_node, next_state
    
    def best_child(self, parent_node):
        # print('### BEST CHILD')
        N = parent_node.N
        max_uct_node = None
        max_uct_value = -float('inf')

        for a in parent_node.children:
            child_node = parent_node.children[a]

            q = sum(child_node.Q)
            n_child = child_node.N

            if n_child == 0:
                uct_value = float('inf')
            else:
                exploitation_term = (q/n_child)
                exploration_term = 2*self.C_p*math.sqrt(2*math.log(N)/n_child)
                uct_value = exploitation_term + exploration_term
            
            if uct_value > max_uct_value:
                max_uct_value = uct_value
                max_uct_node = child_node
        return max_uct_node
    
    def tree_policy(self, parent_node, parent_state):
        # print('# TREE POLICY')
        while not self.env.check_terminal(parent_state):
            if not parent_node.is_full_expanded():
                # print('## EXPAND')
                return self.expand(parent_node, parent_state)
            else:
                # print('## EXPANDED => FIND BEST CHILD')
                parent_node = self.best_child(parent_node)
                a = parent_node.incoming_action
                parent_state = self.env.transition(parent_state, a)
        return parent_node, parent_state
    
    def default_policy(self, state):
        # print('# DEFAULT POLICY')
        # TODO: restrict search depth somehow
        while not self.env.check_terminal(state):
            actions = self.env.available_actions(state)
            num_actions = len(actions)
            idx = random.randint(0,num_actions-1)
            a = actions[idx]
            state = self.env.transition(state, a)
        outcome = self.env.outcome(state)
        # if outcome > 0:
        #     print(f'## terminal, outcome: {self.env.check_terminal(state)}, {outcome}')
        return outcome
    
    def backup_negamax(self, node, outcome):
        # print('# BACKUP')
        while node is not None:
            node.N += 1
            node.Q.append(outcome)
            outcome = -outcome
            node = node.parent

    def uct_search(self, initial_state):
        root_node = Node(self.env.available_actions(initial_state))
        while self.iter < self.max_iters:
            # print(self.iter)
            new_node, new_state = self.tree_policy(root_node, initial_state)
            outcome = self.default_policy(new_state)
            self.backup_negamax(new_node, outcome)
            # TODO: re-consider where computations are accounted for
            self.iter += 1
        return self.best_child(root_node).incoming_action

    def step(self, obs):
        self.iter = 0
        return self.uct_search(obs)

    def act(self):
        pass