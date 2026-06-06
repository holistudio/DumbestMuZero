import numpy as np

from agents.mcts.test_minimax_oracle import (
    apply_action,
    legal_actions,
    reachable_states,
    status,
)
from envs.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE
from envs.tictactoe.tictactoe import raw_env


def test_observation_planes_and_masks_follow_player_perspective():
    env = raw_env()
    env.reset()

    player_one = env.observe("player_1")
    player_two = env.observe("player_2")
    assert player_one["action_mask"].tolist() == [1] * 9
    assert player_two["action_mask"].tolist() == [0] * 9

    env.step(4)
    player_one = env.observe("player_1")
    player_two = env.observe("player_2")

    assert player_one["observation"][:, :, 0].flat[4] == 1
    assert player_two["observation"][:, :, 1].flat[4] == 1
    assert player_one["action_mask"].tolist() == [0] * 9
    assert player_two["action_mask"][4] == 0
    assert player_two["action_mask"].sum() == 8


def test_transition_is_pure_and_swaps_to_next_player_perspective():
    env = raw_env()
    env.reset()
    observation = env.observe("player_1")
    original_board = observation["observation"].copy()
    original_mask = observation["action_mask"].copy()

    next_observation = env.transition(observation, 4)

    assert np.array_equal(observation["observation"], original_board)
    assert np.array_equal(observation["action_mask"], original_mask)
    assert next_observation["observation"][:, :, 0].sum() == 0
    assert next_observation["observation"][:, :, 1].flat[4] == 1
    assert next_observation["action_mask"][4] == 0
    assert next_observation["action_mask"].sum() == 8


def test_transition_sequence_matches_real_environment_observations():
    env = raw_env()
    env.reset()
    simulated = env.observe("player_1")

    for action in [0, 4, 1, 3]:
        simulated = env.transition(simulated, action)
        env.step(action)
        actual = env.observe(env.agent_selection)
        assert np.array_equal(simulated["observation"], actual["observation"])
        assert np.array_equal(simulated["action_mask"], actual["action_mask"])


def test_real_environment_assigns_terminal_rewards_and_status():
    env = raw_env()
    env.reset()

    for action in [0, 3, 1, 4, 2]:
        env.step(action)

    assert env.terminations == {"player_1": True, "player_2": True}
    assert env.rewards == {"player_1": 1, "player_2": -1}
    terminal_observation = env.observe("player_1")
    assert env.check_terminal(terminal_observation)


def test_full_board_draw_is_terminal_with_zero_rewards():
    env = raw_env()
    env.reset()

    for action in [0, 3, 1, 4, 5, 2, 6, 7, 8]:
        env.step(action)

    assert env.terminations == {"player_1": True, "player_2": True}
    assert env.rewards == {"player_1": 0, "player_2": 0}
    assert env.check_terminal(env.observe("player_1"))


def test_outcome_matches_current_player_perspective_for_every_reachable_state():
    env = raw_env()

    for state, player in reachable_states():
        board = np.array(state).reshape(3, 3)
        current = np.equal(board, player + 1)
        opponent = np.equal(board, (player ^ 1) + 1)
        observation = {
            "observation": np.stack([current, opponent], axis=-1).astype(np.int8),
            "action_mask": np.array([value == 0 for value in state], dtype=np.int8),
        }
        game_status = status(state)
        if game_status in (TTT_GAME_NOT_OVER, TTT_TIE):
            expected = 0
        else:
            expected = 1 if game_status == player else -1

        env.agent_selection = "player_1" if player == 1 else "player_2"
        assert env.outcome(observation) == expected
        assert env.check_terminal(observation) == (game_status != TTT_GAME_NOT_OVER)


def test_every_reachable_transition_matches_independent_board_oracle():
    env = raw_env()

    for state, player in reachable_states():
        if status(state) != TTT_GAME_NOT_OVER:
            continue
        board = np.array(state).reshape(3, 3)
        observation = {
            "observation": np.stack(
                [
                    np.equal(board, player + 1),
                    np.equal(board, (player ^ 1) + 1),
                ],
                axis=-1,
            ).astype(np.int8),
            "action_mask": np.array([value == 0 for value in state], dtype=np.int8),
        }

        for action in legal_actions(state):
            next_state = apply_action(state, player, action)
            next_board = np.array(next_state).reshape(3, 3)
            actual = env.transition(observation, action)
            expected_board = np.stack(
                [
                    np.equal(next_board, (player ^ 1) + 1),
                    np.equal(next_board, player + 1),
                ],
                axis=-1,
            ).astype(np.int8)
            expected_mask = np.array(
                [value == 0 for value in next_state], dtype=np.int8
            )
            if status(next_state) != TTT_GAME_NOT_OVER:
                expected_mask[:] = 0

            assert np.array_equal(actual["observation"], expected_board)
            assert np.array_equal(actual["action_mask"], expected_mask)
