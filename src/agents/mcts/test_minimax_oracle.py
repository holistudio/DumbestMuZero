from functools import lru_cache

import numpy as np

from agents.mcts.mcts import Node, UCTAgent
from envs.tictactoe.board import (
    TTT_GAME_NOT_OVER,
    TTT_TIE,
    Board,
)


def status(state):
    board = Board()
    board.squares = list(state)
    return board.game_status()


def legal_actions(state):
    return [action for action, value in enumerate(state) if value == 0]


def apply_action(state, player, action):
    next_state = list(state)
    next_state[action] = player + 1
    return tuple(next_state)


@lru_cache(None)
def minimax_value(state, player):
    game_status = status(state)
    if game_status == TTT_TIE:
        return 0
    if game_status != TTT_GAME_NOT_OVER:
        return 1 if game_status == player else -1
    return max(
        -minimax_value(apply_action(state, player, action), player ^ 1)
        for action in legal_actions(state)
    )


def optimal_actions(state, player):
    best_value = minimax_value(state, player)
    return {
        action
        for action in legal_actions(state)
        if -minimax_value(apply_action(state, player, action), player ^ 1) == best_value
    }


def reachable_states():
    seen = set()

    def visit(state, player):
        key = (state, player)
        if key in seen:
            return
        seen.add(key)
        if status(state) != TTT_GAME_NOT_OVER:
            return
        for action in legal_actions(state):
            visit(apply_action(state, player, action), player ^ 1)

    visit((0,) * 9, 0)
    return seen


class TupleTicTacToe:
    def available_actions(self, position):
        state, _ = position
        return legal_actions(state)

    def transition(self, position, action):
        state, player = position
        return apply_action(state, player, action), player ^ 1

    def check_terminal(self, position):
        state, _ = position
        return status(state) != TTT_GAME_NOT_OVER

    def outcome(self, position):
        state, player = position
        game_status = status(state)
        if game_status == TTT_TIE:
            return 0
        return 1 if game_status == player else -1


def run_uct(position, simulations=100):
    environment = TupleTicTacToe()
    agent = UCTAgent(environment, C_p=0.7, max_iters=simulations)
    root = Node(position, environment.available_actions(position))
    for _ in range(simulations):
        node, state = agent.tree_policy(root, position)
        agent.backup_negamax(node, agent.default_policy(state))
    return max(root.children, key=lambda action: root.children[action].N)


def run_uct_with_oracle_leaf_values(position):
    environment = TupleTicTacToe()
    agent = UCTAgent(environment, C_p=0)
    agent.default_policy = lambda leaf: minimax_value(*leaf)
    root = Node(position, environment.available_actions(position))
    simulations = len(root.untried_actions) + 10
    for _ in range(simulations):
        node, state = agent.tree_policy(root, position)
        agent.backup_negamax(node, agent.default_policy(state))
    return max(root.children, key=lambda action: root.children[action].N)


def test_minimax_oracle_satisfies_bellman_equation_for_every_reachable_state():
    states = reachable_states()
    assert len(states) == 5478

    for state, player in states:
        if status(state) == TTT_GAME_NOT_OVER:
            child_values = [
                -minimax_value(apply_action(state, player, action), player ^ 1)
                for action in legal_actions(state)
            ]
            assert minimax_value(state, player) == max(child_values)


def test_uct_matches_minimax_on_every_reachable_two_move_tactical_position():
    positions = [
        position
        for position in reachable_states()
        if status(position[0]) == TTT_GAME_NOT_OVER
        and len(legal_actions(position[0])) <= 2
        and len(legal_actions(position[0])) > 1
    ]

    for index, position in enumerate(positions):
        np.random.seed(index)
        assert run_uct(position) in optimal_actions(*position)


def test_uct_tree_mechanics_choose_minimax_action_on_every_reachable_position():
    positions = [
        position
        for position in reachable_states()
        if status(position[0]) == TTT_GAME_NOT_OVER
    ]

    for index, position in enumerate(positions):
        np.random.seed(index)
        assert run_uct_with_oracle_leaf_values(position) in optimal_actions(*position)
