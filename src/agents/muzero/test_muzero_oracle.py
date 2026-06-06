import torch

from agents.mcts.test_minimax_oracle import (
    apply_action,
    minimax_value,
    optimal_actions,
    reachable_states,
    status,
)
from agents.muzero.muzero import MuZeroAgent
from envs.tictactoe.board import TTT_GAME_NOT_OVER


class ExactRepresentation:
    def __call__(self, observation):
        return observation.clone()


class ExactDynamics:
    def __call__(self, states, actions):
        next_states = []
        rewards = []
        for state, action in zip(states, actions.argmax(dim=1)):
            board = tuple(int(value) for value in state.tolist())
            player = 0 if board.count(1) == board.count(2) else 1
            next_board = apply_action(board, player, int(action))
            next_states.append(torch.tensor(next_board, dtype=torch.float32))
            rewards.append([float(status(next_board) == player)])
        return torch.stack(next_states), torch.tensor(rewards)


class ExactPrediction:
    def __call__(self, states):
        policies = []
        values = []
        for state in states:
            board = tuple(int(value) for value in state.tolist())
            player = 0 if board.count(1) == board.count(2) else 1
            policy = torch.full((9,), -10.0)
            if status(board) == TTT_GAME_NOT_OVER:
                for action in optimal_actions(board, player):
                    policy[action] = 10
                value = minimax_value(board, player)
            else:
                value = 0
            policies.append(policy)
            values.append([float(value)])
        return torch.stack(policies), torch.tensor(values)


def exact_muzero_agent():
    agent = MuZeroAgent.__new__(MuZeroAgent)
    agent.device = torch.device("cpu")
    agent.action_size = 9
    agent.max_iters = 5
    agent.gamma = 1
    agent.dirichlet_alpha = 1
    agent.state_function = ExactRepresentation()
    agent.dynamics_function = ExactDynamics()
    agent.prediction_function = ExactPrediction()
    agent.action_probs = torch.zeros(9)
    return agent


def test_exact_model_muzero_selects_optimal_action_on_every_reachable_position():
    agent = exact_muzero_agent()

    for board, player in reachable_states():
        if status(board) == TTT_GAME_NOT_OVER:
            action = agent.search(
                torch.tensor(board, dtype=torch.float32),
                temperature=0,
                add_exploration_noise=False,
            )
            assert action in optimal_actions(board, player)
