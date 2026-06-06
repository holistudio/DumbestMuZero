"""Property-based tests for the MuZero implementation.

These use Hypothesis to generate many random-but-legal tic-tac-toe situations
and check invariants that the implementation must satisfy, cross-checking
against the real environment (``raw_env`` / ``Board``) as an independent oracle
wherever possible.

Run just this file with:

    python -m pytest src/agents/muzero/test_muzero_properties.py -q
"""
import copy
import math
import random
import tempfile

import numpy as np
import torch
import pytest
from unittest.mock import patch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from agents.muzero.muzero import (
    DynamicsFunction,
    MuZeroAgent,
    Node,
    PredictionFunction,
    ReplayBuffer,
    StateFunction,
)
from envs.tictactoe.board import TTT_GAME_NOT_OVER, Board
from envs.tictactoe.tictactoe import raw_env

WINNING_COMBINATIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def bare_agent():
    """A MuZeroAgent with just the attributes the pure-search helpers need."""
    agent = MuZeroAgent.__new__(MuZeroAgent)
    agent.env = type("Env", (), {"agent_selection": "player_1"})()
    agent.gamma = 1.0
    agent.min_Q = float("inf")
    agent.max_Q = -float("inf")
    return agent


def search_agent(seed=0, state_size=6, hidden=8, action_size=9, max_iters=24):
    """A fully wired agent with small, randomly-initialised real networks."""
    torch.manual_seed(seed)
    agent = bare_agent()
    agent.device = torch.device("cpu")
    agent.action_size = action_size
    agent.max_iters = max_iters
    agent.temperature = 1.0
    agent.dirichlet_alpha = 1.0
    agent.action_probs = torch.zeros(action_size)
    agent.root_value = 0
    agent.state_function = StateFunction(9, state_size, hidden).eval()
    agent.dynamics_function = DynamicsFunction(state_size + action_size, state_size, hidden).eval()
    agent.prediction_function = PredictionFunction(state_size, action_size, hidden).eval()
    return agent


def play_game(selectors):
    """Play a legal tic-tac-toe game driven by integer move-selectors.

    Returns the env plus a per-step record list mirroring what train.py stores:
    (pre-move observation dict, player label, action, reward, squares-before).
    """
    env = raw_env()
    env.reset()
    records = []
    for sel in selectors:
        if any(env.terminations.values()):
            break
        player = env.agent_selection
        legal = env.board.legal_moves()
        if not legal:
            break
        obs = env.observe(player)
        squares_before = list(env.board.squares)
        action = legal[sel % len(legal)]
        env.step(action)
        records.append((obs, player, action, env.rewards[player], squares_before))
    return env, records


def play_completed_game(selectors):
    """Play until terminal, using selectors when available and defaulting to 0."""
    env = raw_env()
    env.reset()
    records = []
    selector_iter = iter(selectors)
    while not any(env.terminations.values()):
        player = env.agent_selection
        legal = env.board.legal_moves()
        if not legal:
            break
        obs = env.observe(player)
        squares_before = list(env.board.squares)
        sel = next(selector_iter, 0)
        action = legal[sel % len(legal)]
        env.step(action)
        records.append((obs, player, action, env.rewards[player], squares_before))
    return env, records


def trajectory_from_records(records):
    """Build the replay-buffer trajectory dict that experience() would store."""
    obs = []
    turns = []
    actions = []
    rewards = []
    target_policies = []
    root_values = []
    for observation, player, action, reward, _squares in records:
        agent = bare_agent()
        obs.append(agent.preprocess_obs(observation))
        turns.append(0 if player == "player_1" else 1)
        actions.append(action)
        rewards.append(reward)
        policy = torch.zeros(9)
        policy[action] = 1.0
        target_policies.append(policy)
        root_values.append(0.0)
    return dict(
        obs=obs, turns=turns, actions=actions, rewards=rewards,
        target_policies=target_policies, root_values=root_values,
    )


def policy_from_legal_actions(legal_actions, seed, step):
    rng = random.Random(seed * 1009 + step * 9176 + len(legal_actions))
    weights = [rng.random() + 0.01 for _ in legal_actions]
    total = sum(weights)
    policy = torch.zeros(9)
    for action, weight in zip(legal_actions, weights):
        policy[action] = weight / total
    return policy


def make_training_agent(seed, train_iters=1, k_unroll_steps=2):
    class ObservationSpace:
        shape = (3, 3, 2)

    class ActionSpace:
        n = 9

    class Environment:
        agent_selection = "player_1"

        def observation_space(self, _):
            return {"observation": ObservationSpace()}

        def action_space(self, _):
            return ActionSpace()

    torch.manual_seed(seed)
    config = {
        "batch_size": 1,
        "buffer_size": 10,
        "min_replay_size": 1,
        "state_size": 8,
        "hidden_size": 8,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_iters": 2,
        "train_iters": train_iters,
        "gamma": 1.0,
        "k_unroll_steps": k_unroll_steps,
        "temperature": 1.0,
        "dirichlet_alpha": 1.0,
        "checkpoint_interval": 10**9,
    }
    return MuZeroAgent(Environment(), config)


def assert_trajectory_dict_equal(actual, expected):
    assert actual["turns"] == expected["turns"]
    assert actual["actions"] == expected["actions"]
    assert actual["rewards"] == expected["rewards"]
    assert len(actual["obs"]) == len(expected["obs"])
    assert len(actual["target_policies"]) == len(expected["target_policies"])
    assert len(actual["root_values"]) == len(expected["root_values"])
    for left, right in zip(actual["obs"], expected["obs"]):
        assert torch.equal(left, right)
    for left, right in zip(actual["target_policies"], expected["target_policies"]):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)
    for left, right in zip(actual["root_values"], expected["root_values"]):
        assert math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def assert_replay_buffers_equal(actual_buffer, expected_buffer):
    assert len(actual_buffer) == len(expected_buffer)
    for actual, expected in zip(actual_buffer, expected_buffer):
        assert_trajectory_dict_equal(actual, expected)


def assert_optimizer_state_equal(left_state, right_state):
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id, left_values in left_state["state"].items():
        right_values = right_state["state"][parameter_id]
        assert left_values.keys() == right_values.keys()
        for key, left_value in left_values.items():
            right_value = right_values[key]
            if torch.is_tensor(left_value):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def board_to_grid(state):
    return np.array(state, dtype=np.int8).reshape((3, 3))


def grid_to_board(grid):
    return tuple(np.asarray(grid, dtype=np.int8).reshape(9).tolist())


def transform_grid(grid, sym):
    if sym == 0:
        return grid
    if sym == 1:
        return np.rot90(grid, 1)
    if sym == 2:
        return np.rot90(grid, 2)
    if sym == 3:
        return np.rot90(grid, 3)
    if sym == 4:
        return np.flipud(grid)
    if sym == 5:
        return np.fliplr(grid)
    if sym == 6:
        return np.flipud(np.rot90(grid, 1))
    if sym == 7:
        return np.fliplr(np.rot90(grid, 1))
    raise ValueError(sym)


def transform_state(state, sym):
    return grid_to_board(transform_grid(board_to_grid(state), sym))


def transform_action(action, sym):
    marker = np.zeros(9, dtype=np.int8)
    marker[action] = 1
    transformed = grid_to_board(transform_grid(board_to_grid(marker), sym))
    return transformed.index(1)


def observation_from_state(state, player):
    board = board_to_grid(state)
    if player == "player_1":
        current = (board == 1).astype(np.int8)
        opponent = (board == 2).astype(np.int8)
    else:
        current = (board == 2).astype(np.int8)
        opponent = (board == 1).astype(np.int8)
    return {
        "observation": np.stack([current, opponent], axis=-1).astype(np.int8),
        "action_mask": np.array(board.reshape(9) == 0, dtype=np.int8),
    }


def reference_sample_batch(traj, start, k_unroll_steps, gamma):
    observations = traj["obs"]
    turns = traj["turns"]
    actions = traj["actions"]
    rewards = traj["rewards"]
    target_policies = traj["target_policies"]
    root_values = traj["root_values"]

    obs = observations[start]
    sampled_actions = []
    sample_rewards = []
    sample_values = []
    sample_policies = []
    sample_masks = []

    for k in range(k_unroll_steps):
        if start + k < len(actions):
            sampled_actions.append(actions[start + k])
        else:
            sampled_actions.append(0)

    for i in range(start, start + k_unroll_steps + 1):
        if i < len(observations):
            mask = observations[i] == 0
        else:
            mask = torch.ones_like(observations[0], dtype=torch.bool)
        sample_masks.append(mask)

        current_player = turns[i] if i < len(turns) else None
        bootstrap_ix = i + k_unroll_steps
        if bootstrap_ix < len(root_values):
            value = root_values[bootstrap_ix] * gamma**k_unroll_steps
            bootstrap_player = turns[bootstrap_ix]
            if current_player is not None and current_player != bootstrap_player:
                value = -value
        else:
            value = 0.0
        for j, reward in enumerate(rewards[i:bootstrap_ix]):
            if i + j < len(turns) and turns[i + j] == current_player:
                value += reward * gamma**j
            else:
                value -= reward * gamma**j
        sample_values.append(value)

        if i < len(root_values):
            sample_policies.append(target_policies[i])
        else:
            sample_policies.append(torch.zeros_like(target_policies[0]))

        if i > 0 and i <= len(rewards):
            sample_rewards.append(rewards[i - 1])
        else:
            sample_rewards.append(0.0)

    return (
        obs,
        torch.tensor(sampled_actions, dtype=torch.long),
        torch.tensor(sample_rewards, dtype=torch.float32),
        torch.tensor(sample_values, dtype=torch.float32),
        torch.stack(sample_policies),
        torch.stack(sample_masks),
    )


# ---------------------------------------------------------------------------
# P1: preprocess_obs reconstructs the canonical (mover-relative) board for any
# reachable state: +1 = current player's pieces, -1 = opponent's, 0 = empty.
# ---------------------------------------------------------------------------
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=0, max_value=8), max_size=9))
def test_preprocess_obs_reconstructs_canonical_board(selectors):
    _env, records = play_game(selectors)
    assume(records)
    agent = bare_agent()
    for observation, player, _action, _reward, squares_before in records:
        current_mark = 1 if player == "player_1" else 2
        expected = [
            0.0 if v == 0 else (1.0 if v == current_mark else -1.0)
            for v in squares_before
        ]
        board = agent.preprocess_obs(observation)
        assert board.tolist() == expected


# ---------------------------------------------------------------------------
# P2: at the root (empty history), get_legal_actions matches the env exactly.
# ---------------------------------------------------------------------------
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=0, max_value=8), max_size=9))
def test_root_legal_actions_match_environment(selectors):
    env, records = play_game(selectors)
    assume(records)
    agent = bare_agent()
    # Use the last pre-move observation that is still a live position.
    observation, _player, _action, _reward, squares_before = records[-1]
    if any(env.terminations.values()):
        return  # last recorded position led to terminal; nothing to compare
    board = agent.preprocess_obs(observation)
    got = sorted(agent.get_legal_actions(board, []))
    expected = sorted(i for i, v in enumerate(squares_before) if v == 0)
    assert got == expected


# ---------------------------------------------------------------------------
# P3: get_legal_actions over a hypothetical continuation agrees with a fully
# independent oracle (the real Board's win detection + legal_moves).
# ---------------------------------------------------------------------------
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(st.integers(min_value=0, max_value=8), max_size=8),
    st.lists(st.integers(min_value=0, max_value=8), max_size=9),
)
def test_hypothetical_legal_actions_match_independent_oracle(root_sel, cont_sel):
    # Build a reachable (possibly terminal) root via a real game.
    _env, records = play_game(root_sel)
    assume(records)
    root_squares = list(records[-1][4])  # squares before the last recorded move
    counts = (root_squares.count(1), root_squares.count(2))
    root_player = 0 if counts[0] == counts[1] else 1

    # Independent oracle board, and the history we will feed get_legal_actions.
    oracle = Board()
    oracle.squares = list(root_squares)
    history = []
    for sel in cont_sel:
        if oracle.game_status() != TTT_GAME_NOT_OVER:
            break
        empties = [i for i, v in enumerate(oracle.squares) if v == 0]
        if not empties:
            break
        action = empties[sel % len(empties)]
        player = (root_player + len(history)) % 2
        oracle.squares[action] = player + 1
        history.append(action)

    if oracle.game_status() != TTT_GAME_NOT_OVER:
        expected = []
    else:
        expected = sorted(oracle.legal_moves())

    agent = bare_agent()
    # get_legal_actions now consumes the canonical (mover-relative) board.
    board = agent.preprocess_obs(records[-1][0])
    got = sorted(agent.get_legal_actions(board, history))
    assert got == expected


# ---------------------------------------------------------------------------
# P4: board symmetries must preserve terminal status, legal move structure,
# and preprocess_obs should remain consistent under transformed observations.
# ---------------------------------------------------------------------------
@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), max_size=9),
    st.integers(min_value=0, max_value=7),
)
def test_board_symmetry_preserves_status_legal_moves_and_observations(selectors, sym):
    _env, records = play_game(selectors)
    assume(records)
    observation, player, _action, _reward, squares_before = records[-1]

    board = Board()
    board.squares = list(squares_before)
    transformed_state = transform_state(squares_before, sym)
    transformed_board = Board()
    transformed_board.squares = list(transformed_state)

    agent = bare_agent()
    assert board.game_status() == transformed_board.game_status()
    assert sorted(transform_action(action, sym) for action in board.legal_moves()) == sorted(transformed_board.legal_moves())
    current_mark = 1 if player == "player_1" else 2
    expected_canonical = [
        0.0 if v == 0 else (1.0 if v == current_mark else -1.0)
        for v in transformed_state
    ]
    assert agent.preprocess_obs(observation_from_state(transformed_state, player)).tolist() == expected_canonical


# ---------------------------------------------------------------------------
# P5: search without exploration noise should be deterministic for a fixed
# board, fixed weights, and repeated calls on the same agent.
# ---------------------------------------------------------------------------
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), max_size=7),
    st.integers(min_value=0, max_value=10_000),
)
def test_search_is_deterministic_without_exploration_noise(selectors, seed):
    _env, records = play_game(selectors)
    assume(records)
    squares = list(records[-1][4])
    board_check = Board()
    board_check.squares = squares
    assume(board_check.game_status() == TTT_GAME_NOT_OVER)

    agent = search_agent(seed=seed, max_iters=24)
    board = bare_agent().preprocess_obs(records[-1][0])

    first_action = agent.search(board, temperature=0, add_exploration_noise=False)
    first_probs = agent.action_probs.clone()
    first_root = agent.root_value

    second_action = agent.search(board, temperature=0, add_exploration_noise=False)
    second_probs = agent.action_probs.clone()
    second_root = agent.root_value

    assert first_action == second_action
    assert torch.allclose(first_probs, second_probs, atol=1e-6, rtol=1e-6)
    assert first_root == pytest.approx(second_root)


# ---------------------------------------------------------------------------
# P6: experience() must preserve the exact trajectory that was played, clone
# the live policy tensor, and isolate stored replay data from later mutation.
# ---------------------------------------------------------------------------
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), min_size=9, max_size=40),
    st.integers(min_value=0, max_value=10_000),
)
def test_experience_stores_exact_trajectory_and_isolates_mutation(selectors, seed):
    env = raw_env()
    env.reset()
    agent = bare_agent()
    agent.replay_buffer = ReplayBuffer(buffer_size=20, batch_size=1)
    agent.action_probs = torch.zeros(9)
    expected = {
        "obs": [],
        "turns": [],
        "actions": [],
        "rewards": [],
        "target_policies": [],
        "root_values": [],
    }

    step = 0
    for sel in selectors:
        if any(env.terminations.values()):
            break
        player = env.agent_selection
        legal = env.board.legal_moves()
        assume(legal)
        observation = env.observe(player)
        action = legal[sel % len(legal)]
        policy = policy_from_legal_actions(legal, seed, step)
        root_value = ((seed % 11) - 5 + step) / 7.0

        agent.action_probs = policy.clone()
        agent.root_value = root_value
        env.step(action)
        terminal = any(env.terminations.values())
        reward = env.rewards[player]
        agent.experience(observation, player, action, reward, terminal)

        expected["obs"].append(agent.preprocess_obs(observation))
        expected["turns"].append(0 if player == "player_1" else 1)
        expected["actions"].append(action)
        expected["rewards"].append(reward)
        expected["target_policies"].append(policy)
        expected["root_values"].append(root_value)
        step += 1

    assume(any(env.terminations.values()))
    assert len(agent.replay_buffer.buffer) == 1
    assert_trajectory_dict_equal(agent.replay_buffer.buffer[0], expected)

    # Later mutations to the live tensors must not leak into stored replay.
    agent.action_probs[0] = 1.0
    agent.root_value = 999.0
    assert_trajectory_dict_equal(agent.replay_buffer.buffer[0], expected)


# ---------------------------------------------------------------------------
# P7: sample_batch() must match an independent reference implementation for
# value bootstrapping, reward alignment, action padding, masks, and absorbing
# state handling.
# ---------------------------------------------------------------------------
@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), min_size=9, max_size=40),
    st.integers(min_value=0, max_value=8),
    st.integers(min_value=0, max_value=8),
    st.integers(min_value=0, max_value=10_000),
    st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_sample_batch_matches_reference_for_random_trajectories(selectors, start, k_unroll_steps, seed, gamma):
    env, records = play_game(selectors)
    assume(records)
    assume(any(env.terminations.values()))

    traj = trajectory_from_records(records)
    traj["root_values"] = [((seed % 7) - 3 + i) / 4.0 for i in range(len(records))]
    replay = ReplayBuffer(buffer_size=1, batch_size=1)
    replay.buffer = [traj]

    start = min(start, len(records) - 1)
    k_unroll_steps = min(k_unroll_steps, 8)
    expected = reference_sample_batch(traj, start, k_unroll_steps, gamma)
    script = [0, start]
    with patched_randint(script):
        actual = replay.sample_batch(k_unroll_steps, gamma, "cpu")

    for left, right in zip(actual, expected):
        assert torch.equal(left, right) or torch.allclose(left, right, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# P8: sample_batch() should agree with the reference recurrence across many
# batch rows and unroll lengths.
# ---------------------------------------------------------------------------
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), min_size=9, max_size=40),
    st.lists(st.integers(min_value=0, max_value=8), min_size=1, max_size=4),
    st.integers(min_value=0, max_value=8),
    st.integers(min_value=0, max_value=10_000),
    st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_sample_batch_matches_reference_across_many_rows(selectors, starts, k_unroll_steps, seed, gamma):
    env, records = play_completed_game(selectors)

    traj = trajectory_from_records(records)
    traj["root_values"] = [((seed % 7) - 3 + i) / 4.0 for i in range(len(records))]
    batch_size = min(len(starts), 4)
    starts = [min(start, len(records) - 1) for start in starts[:batch_size]]
    k_unroll_steps = min(k_unroll_steps, 8)
    replay = ReplayBuffer(buffer_size=1, batch_size=batch_size)
    replay.buffer = [traj]

    expected_rows = [reference_sample_batch(traj, start, k_unroll_steps, gamma) for start in starts]
    expected = tuple(torch.stack([row[i] for row in expected_rows]) for i in range(6))

    script = []
    for start in starts:
        padding = max(0, k_unroll_steps - (len(records) - start))
        script.extend([0, start] + [0] * padding)
    with patched_randint(script):
        actual = replay.sample_batch(k_unroll_steps, gamma, "cpu")

    for left, right in zip(actual, expected):
        assert torch.equal(left, right) or torch.allclose(left, right, atol=1e-6, rtol=1e-6)


class patched_randint:
    """Context manager: force random.randint inside sample_batch to a script."""

    def __init__(self, script):
        self.script = list(script)

    def __enter__(self):
        import agents.muzero.muzero as mz

        self.orig = mz.random.randint
        script = iter(self.script)

        def fake_randint(low, high):
            # Follow the script, then fall back to `low` for any extra calls
            # (e.g. random action padding for absorbing states).
            return next(script, low)

        mz.random.randint = fake_randint
        return self

    def __exit__(self, *exc):
        import agents.muzero.muzero as mz

        mz.random.randint = self.orig
        return False


# ---------------------------------------------------------------------------
# P5: a full search yields a valid policy supported only on legal actions, and
# never returns an illegal move.
# ---------------------------------------------------------------------------
@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), max_size=7),
    st.integers(min_value=0, max_value=10),
)
def test_search_policy_is_legal_distribution(root_sel, seed):
    _env, records = play_game(root_sel)
    assume(records)
    squares = list(records[-1][4])
    legal = [i for i, v in enumerate(squares) if v == 0]
    assume(legal)
    # Skip already-won roots (no decision to make there).
    board_check = Board()
    board_check.squares = squares
    assume(board_check.game_status() == TTT_GAME_NOT_OVER)

    agent = search_agent(seed=seed, max_iters=24)
    board = bare_agent().preprocess_obs(records[-1][0])

    action = agent.search(board, temperature=0, add_exploration_noise=False)

    probs = agent.action_probs
    assert action in legal
    assert math.isclose(probs.sum().item(), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    illegal = [i for i in range(9) if i not in legal]
    for i in illegal:
        assert probs[i].item() == 0.0
    assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# P6: policy_loss must renormalize legal targets, ignore illegal mass, and
# produce zero gradients for illegal actions and absorbing states.
# ---------------------------------------------------------------------------
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=0, max_value=10_000),
    st.lists(st.lists(st.booleans(), min_size=9, max_size=9), min_size=1, max_size=4),
    st.lists(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=9, max_size=9), min_size=1, max_size=4),
)
def test_policy_loss_property_masks_and_renormalizes(seed, masks, targets):
    agent = bare_agent()
    torch.manual_seed(seed)
    batch = min(len(masks), len(targets))
    masks = masks[:batch]
    targets = targets[:batch]
    logits = torch.randn(batch, 9, requires_grad=True)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    mask_tensor = torch.tensor(masks, dtype=torch.bool)

    loss = agent.policy_loss(logits, target_tensor, mask_tensor)
    loss.backward()

    legal_targets = target_tensor.masked_fill(~mask_tensor, 0)
    target_sums = legal_targets.sum(dim=1, keepdim=True)
    valid_targets = target_sums.squeeze(1) > 0
    normalized = torch.where(
        target_sums > 0,
        legal_targets / target_sums.clamp_min(torch.finfo(legal_targets.dtype).eps),
        legal_targets,
    )
    masked_logits = logits.detach().masked_fill(~mask_tensor, -1e9)
    per_sample = -(normalized * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)
    expected = per_sample[valid_targets].mean() if valid_targets.any() else logits.sum() * 0

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6, abs_tol=1e-6)
    assert logits.grad.masked_select(~mask_tensor).abs().sum().item() == 0
    if not valid_targets.any():
        assert logits.grad.abs().sum().item() == 0


# ---------------------------------------------------------------------------
# P7: negamax backup invariant. After a single backup, the leaf carries the
# leaf value and the root carries the alternating-sign return.
# ---------------------------------------------------------------------------
@settings(max_examples=300)
@given(
    st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=1, max_size=6),
    st.floats(min_value=-1, max_value=1, allow_nan=False),
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
)
def test_backup_follows_negamax_recurrence(rewards, leaf_value, gamma):
    agent = bare_agent()
    agent.gamma = gamma
    path = [Node(0) for _ in rewards]
    for node, r in zip(path, rewards):
        node.R = r

    agent.backup(leaf_value, path)

    # Recompute the expected per-node accumulated return.
    g = leaf_value
    expected = []
    for node in reversed(path):
        expected.append(g)
        g = node.R - gamma * g
    expected.reverse()

    for node, exp in zip(path, expected):
        assert node.N == 1
        assert math.isclose(node.mean_value(), exp, rel_tol=1e-6, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# P8: update() must move all three networks, increment the training counter,
# keep replay data unchanged, and produce finite losses.
# ---------------------------------------------------------------------------
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), min_size=9, max_size=40),
    st.integers(min_value=0, max_value=10_000),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=4),
)
def test_update_changes_parameters_and_preserves_replay(selectors, seed, train_iters, k_unroll_steps):
    env, records = play_game(selectors)
    assume(records)
    assume(any(env.terminations.values()))

    traj = trajectory_from_records(records)
    traj["root_values"] = [((seed % 7) - 3 + i) / 5.0 for i in range(len(records))]
    agent = make_training_agent(seed, train_iters=train_iters, k_unroll_steps=k_unroll_steps)
    agent.replay_buffer = ReplayBuffer(buffer_size=10, batch_size=2)
    agent.replay_buffer.buffer = [copy.deepcopy(traj)]

    before_buffer = copy.deepcopy(agent.replay_buffer.buffer)
    before_networks = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in (agent.state_function, agent.dynamics_function, agent.prediction_function)
    ]

    with patch.object(agent, "save_model"):
        agent.update()

    assert agent.training_steps == train_iters
    assert len(agent.last_update_losses) == train_iters
    assert agent.last_loss == pytest.approx(sum(agent.last_update_losses) / len(agent.last_update_losses))
    assert math.isfinite(agent.last_loss)
    assert_replay_buffers_equal(agent.replay_buffer.buffer, before_buffer)

    for old_parameters, network in zip(
        before_networks,
        (agent.state_function, agent.dynamics_function, agent.prediction_function),
    ):
        new_parameters = list(network.parameters())
        assert any(
            not torch.equal(old, new)
            for old, new in zip(old_parameters, new_parameters)
        )
        assert all(torch.isfinite(parameter).all() for parameter in new_parameters)


# ---------------------------------------------------------------------------
# P9: checkpoint save/load must round-trip network parameters and optimizer
# state for random initialized agents after an update step.
# ---------------------------------------------------------------------------
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    st.lists(st.integers(min_value=0, max_value=8), min_size=9, max_size=40),
    st.integers(min_value=0, max_value=10_000),
)
def test_checkpoint_round_trip_preserves_parameters_and_optimizer(selectors, seed):
    env, records = play_game(selectors)
    assume(records)
    assume(any(env.terminations.values()))

    traj = trajectory_from_records(records)
    traj["root_values"] = [((seed % 5) - 2 + i) / 3.0 for i in range(len(records))]

    source = make_training_agent(seed, train_iters=1, k_unroll_steps=1)
    source.replay_buffer = ReplayBuffer(buffer_size=10, batch_size=1)
    source.replay_buffer.buffer = [copy.deepcopy(traj)]
    with patch.object(source, "save_model"):
        source.update()

    with tempfile.TemporaryDirectory() as tmpdir:
        source.save_model(tmpdir)

        restored = make_training_agent(seed + 1, train_iters=1, k_unroll_steps=1)
        restored.load_model(tmpdir)

        for source_network, restored_network in zip(
            (source.state_function, source.dynamics_function, source.prediction_function),
            (restored.state_function, restored.dynamics_function, restored.prediction_function),
        ):
            for source_parameter, restored_parameter in zip(
                source_network.parameters(), restored_network.parameters()
            ):
                assert torch.equal(source_parameter, restored_parameter)

        assert_optimizer_state_equal(source.optimizer.state_dict(), restored.optimizer.state_dict())


# ---------------------------------------------------------------------------
# P7: the n-step bootstrap must use stored MCTS root values when they exist.
# Here every reward is zero but the future root values are large; a working
# bootstrap must surface them in the interior targets.
# ---------------------------------------------------------------------------
def test_bootstrap_branch_uses_future_root_values():
    traj = dict(
        obs=[torch.zeros(9) for _ in range(5)],
        turns=[0, 1, 0, 1, 0],
        actions=[0, 1, 2, 3, 4],
        rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        target_policies=[torch.ones(9) / 9 for _ in range(5)],
        root_values=[5.0, 5.0, 5.0, 5.0, 5.0],
    )
    replay = ReplayBuffer(buffer_size=1, batch_size=1)
    replay.buffer = [copy.deepcopy(traj)]

    # Sample from the very first position and unroll across the whole game.
    # Every interior target has a valid bootstrap index, so the large future
    # root_values must appear in the sampled value targets.
    with patched_randint([0, 0]):
        _, _, _, values, _, _ = replay.sample_batch(4, 1.0, "cpu")

    assert values.shape == (1, 5)
    assert values[0, 0].item() == 5.0
    assert values[0, 1].item() == 0.0
    assert values[0, 2].item() == 0.0
    assert values[0, 3].item() == 0.0
    assert values[0, 4].item() == 0.0
