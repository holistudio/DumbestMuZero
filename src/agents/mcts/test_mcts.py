from unittest.mock import patch

from agents.mcts.mcts import Node, UCTAgent


class CountdownEnvironment:
    """A deterministic game where each action removes one or two counters."""

    def available_actions(self, state):
        return [action for action in (1, 2) if action <= state]

    def transition(self, state, action):
        return state - action

    def check_terminal(self, state):
        return state == 0

    def outcome(self, state):
        assert state == 0
        return 1


class OneMoveEnvironment:
    def available_actions(self, state):
        return [0, 1] if state == "root" else []

    def transition(self, state, action):
        assert state == "root"
        return "win" if action == 1 else "loss"

    def check_terminal(self, state):
        return state != "root"

    def outcome(self, state):
        # Terminal outcomes are from the next player-to-move perspective.
        return -1 if state == "win" else 1


def test_node_samples_and_removes_an_untried_action():
    node = Node(state=3, available_actions=[1, 2])

    with patch("agents.mcts.mcts.np.random.randint", return_value=1):
        action = node.sample_untried_actions()

    assert action == 2
    assert node.untried_actions == [1]
    assert not node.is_full_expanded()


def test_expand_links_parent_child_and_applies_transition():
    agent = UCTAgent(CountdownEnvironment())
    parent = Node(state=3, available_actions=[1, 2])

    with patch("agents.mcts.mcts.np.random.randint", return_value=0):
        child, state = agent.expand(parent, 3)

    assert state == 2
    assert parent.children[1] is child
    assert child.parent is parent
    assert child.incoming_action == 1
    assert child.untried_actions == [1, 2]


def test_best_child_prefers_unvisited_then_worse_value_for_child_player():
    agent = UCTAgent(CountdownEnvironment(), C_p=0)
    parent = Node(state=3, available_actions=[])
    parent.N = 3
    visited = Node(state=2, available_actions=[], parent=parent, incoming_action=1)
    visited.N = 3
    visited.Q = [-1, -1, -1]
    unvisited = Node(state=1, available_actions=[], parent=parent, incoming_action=2)
    parent.children = {1: visited, 2: unvisited}

    assert agent.best_child(parent, 3) is unvisited

    unvisited.N = 1
    unvisited.Q = [1]
    assert agent.best_child(parent, 3) is visited


def test_backup_negamax_alternates_perspective_and_visits():
    agent = UCTAgent(CountdownEnvironment())
    root = Node(state=2, available_actions=[])
    child = Node(state=1, available_actions=[], parent=root, incoming_action=1)
    leaf = Node(state=0, available_actions=[], parent=child, incoming_action=1)

    agent.backup_negamax(leaf, 1)

    assert leaf.N == child.N == root.N == 1
    assert leaf.Q == [1]
    assert child.Q == [-1]
    assert root.Q == [1]


def test_tree_policy_expands_an_untried_action():
    agent = UCTAgent(CountdownEnvironment())
    root = Node(state=3, available_actions=[1, 2])

    with patch("agents.mcts.mcts.np.random.randint", return_value=0):
        node, state = agent.tree_policy(root, 3)

    assert node.parent is root
    assert state == 2
    assert len(root.children) == 1


def test_default_policy_reaches_terminal_state():
    agent = UCTAgent(CountdownEnvironment())

    with patch("agents.mcts.mcts.np.random.randint", return_value=0):
        assert agent.default_policy(3) == 1


def test_uct_allocates_more_visits_to_known_winning_move():
    agent = UCTAgent(OneMoveEnvironment(), C_p=0.7, max_iters=40)
    root = Node(state="root", available_actions=[0, 1])

    for _ in range(agent.max_iters):
        node, state = agent.tree_policy(root, "root")
        agent.backup_negamax(node, agent.default_policy(state))

    assert set(root.children) == {0, 1}
    assert root.N == agent.max_iters
    assert root.children[1].N > root.children[0].N
