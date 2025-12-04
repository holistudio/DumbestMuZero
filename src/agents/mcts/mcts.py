import math
import random

from agents.utils import display_board

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
    def __init__(self, environment, C_p=0.7, max_iters=10000):
        self.C_p = C_p
        self.max_iters = max_iters

        self.env = environment
        pass

    def expand(self, parent_node, parent_state):
        # print('### EXPANDING')
        action = parent_node.sample_untried_actions()
        next_state = self.env.transition(parent_state, action)
        # print('### NEXT STATE')
        # display_board(next_state)
        next_actions = self.env.available_actions(next_state)
        next_node = Node(available_actions=next_actions, parent=parent_node, incoming_action=action)
        parent_node.children[action] = next_node
        # pause = input('###')
        return next_node, next_state
    
    def best_child(self, parent_node, parent_state):
        N = parent_node.N
        max_uct_node = None
        max_uct_value = -float('inf')

        for a in sorted(parent_node.children):
            
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
        # print(f'# Terminal?: {self.env.check_terminal(parent_state)}')
        while not self.env.check_terminal(parent_state):
            if not parent_node.is_full_expanded():
                # print('## EXPAND')
                return self.expand(parent_node, parent_state)
            else:
                # print('## EXPANDED => FIND BEST CHILD')
                parent_node = self.best_child(parent_node, parent_state)
                a = parent_node.incoming_action
                parent_state = self.env.transition(parent_state, a)
                # print('## BEST CHILD STATE')
                # display_board(parent_state)
        # pause = input('# END OF TREE POLICY ITER')
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
            # print('## NEXT SIM STATE')
            # print(state["observation"][:,:,0])
            # print(state["observation"][:,:,1])
            # display_board(state)
            # pause = input('##')
        outcome = self.env.outcome(state)
        # if outcome > 0:
        # print(f'# terminal, outcome: {self.env.check_terminal(state)}, {outcome}')
        # pause = input('# END OF SIM\n')
        return outcome
    
    def backup_negamax(self, node, outcome):
        # print('# BACKUP')
        while node is not None:
            node.N += 1
            node.Q.append(outcome)
            outcome = -outcome
            node = node.parent
    
    def final_action(self, root_node, initial_state):
        print('# FINAL ACTION')
        N = root_node.N
        best_action = None
        max_visits = -1
        # for action, child in sorted(root_node.children.items()):
        for a in sorted(root_node.children):
            child_node = root_node.children[a]
            avg_q = sum(child_node.Q) / child_node.N if child_node.N > 0 else 0
            exploration_term = 2*self.C_p*math.sqrt(2*math.log(N)/child_node.N) if child_node.N > 0 else float('inf')
            if avg_q >= 0:
                print(f"### Action={a}, Expected Value=  {avg_q:.2f}, Num Visits= {child_node.N}, UCT Value= {avg_q+exploration_term:.2f}")
            else:
                print(f"### Action={a}, Expected Value= {avg_q:.2f}, Num Visits= {child_node.N}, UCT Value= {avg_q+exploration_term:.2f}")
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_action = a
        print("# NEXT BOARD STATE")
        display_board(self.env.transition(initial_state, best_action))
        # pause = input('#')
        return best_action

    def uct_search(self, initial_state):
        root_node = Node(self.env.available_actions(initial_state))
        for i in range(self.max_iters):
            # print(self.iter)
            new_node, new_state = self.tree_policy(root_node, initial_state)
            outcome = self.default_policy(new_state)
            self.backup_negamax(new_node, outcome)
            # pause = input('END OF SEARCH ITER')
        # print('END OF UCT SEARCH')

        # return self.best_child(root_node, initial_state).incoming_action
        return self.final_action(root_node, initial_state)

    def step(self, obs):
        display_board(obs)
        return self.uct_search(obs)

    def act(self):
        pass