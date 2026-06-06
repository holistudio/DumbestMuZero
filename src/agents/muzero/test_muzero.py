from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from agents.muzero.muzero import (
    DynamicsFunction,
    MuZeroAgent,
    Node,
    PredictionFunction,
    ReplayBuffer,
    StateFunction,
)
from envs.tictactoe.tictactoe import raw_env


def make_search_agent():
    agent = MuZeroAgent.__new__(MuZeroAgent)
    agent.env = type("Env", (), {"agent_selection": "player_1"})()
    agent.gamma = 1.0
    agent.min_Q = float("inf")
    agent.max_Q = -float("inf")
    return agent


def make_observation(current=(), opponent=()):
    board = np.zeros((3, 3, 2), dtype=np.int8)
    for action in current:
        board[:, :, 0].flat[action] = 1
    for action in opponent:
        board[:, :, 1].flat[action] = 1
    return {
        "observation": board,
        "action_mask": (board.sum(axis=2).flatten() == 0).astype(np.int8),
    }


def make_trajectory():
    observations = []
    board = torch.zeros(9)
    actions = [0, 3, 1, 4, 2]
    turns = [0, 1, 0, 1, 0]
    for action, turn in zip(actions, turns):
        observations.append(board.clone())
        board[action] = turn + 1

    return {
        "obs": observations,
        "turns": turns,
        "actions": actions,
        "rewards": [0, 0, 0, 0, 1],
        "target_policies": [torch.ones(9) / 9 for _ in actions],
        # Only the final position carries a nonzero search value. This lets the
        # replay-target tests verify that bootstrapping actually pulls values
        # forward from later positions instead of collapsing everything to zero.
        "root_values": [0, 0, 0, 0, 1],
    }


def make_terminal_trajectory(actions):
    observations = []
    board = torch.zeros(9)
    turns = []
    for index, action in enumerate(actions):
        player = index % 2
        observations.append(board.clone())
        turns.append(player)
        board[action] = player + 1
    rewards = [0] * len(actions)
    rewards[-1] = 0 if len(actions) == 9 else 1
    if rewards[-1] == 1:
        root_values = [(-1) ** (index + 1) for index in range(len(actions))]
    else:
        root_values = [0] * len(actions)
    return {
        "obs": observations,
        "turns": turns,
        "actions": actions,
        "rewards": rewards,
        "target_policies": [torch.ones(9) / 9 for _ in actions],
        "root_values": root_values,
    }


def test_preprocess_obs_encodes_canonical_board_on_player_one_turn():
    agent = make_search_agent()
    observation = make_observation(current=[0], opponent=[4])

    board = agent.preprocess_obs(observation)

    # Canonical encoding: +1 = current player's pieces, -1 = opponent's.
    assert board.tolist() == [1, 0, 0, 0, -1, 0, 0, 0, 0]


def test_preprocess_obs_encodes_canonical_board_on_player_two_turn():
    agent = make_search_agent()
    # PettingZoo plane 0 is always the observing player's pieces. After X
    # opens, player two sees X in the opponent plane, so it becomes -1.
    observation = make_observation(current=[], opponent=[0])

    board = agent.preprocess_obs(observation)

    assert board.tolist() == [-1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_expansion_stores_reward_used_by_search():
    agent = make_search_agent()
    node = Node(1.0)

    agent.expansion(
        node,
        torch.zeros(1, 2),
        0.75,
        torch.zeros(1, 2),
        [0, 1],
        [0],
    )

    assert node.R == 0.75


def test_backup_converts_terminal_reward_to_parent_perspective():
    agent = make_search_agent()
    root = Node(0)
    child = Node(1)
    child.R = 1

    agent.backup(0, [root, child])

    assert child.mean_value() == 0
    assert root.mean_value() == 1


def test_puct_negates_child_player_value():
    agent = make_search_agent()
    agent.min_Q = -1
    agent.max_Q = 1

    opponent_loses = Node(1)
    opponent_loses.N = 1
    opponent_loses.value_sum = -1

    opponent_wins = Node(1)
    opponent_wins.N = 1
    opponent_wins.value_sum = 1

    assert agent.pUCT(opponent_loses, 2) > agent.pUCT(opponent_wins, 2)


def test_puct_normalizes_negated_values_with_parent_perspective_bounds():
    agent = make_search_agent()
    agent.min_Q = 0
    agent.max_Q = 2
    child = Node(1)
    child.N = 1
    child.value_sum = 0

    # Parent-perspective value is the maximum of the negated [0, 2] range.
    expected_value_score = 1
    expected_prior_score = 1.25 + np.log((1 + 19652 + 1) / 19652)
    assert agent.pUCT(child, 1) == pytest.approx(
        expected_value_score + expected_prior_score / 2
    )


def test_simulated_win_has_no_legal_followup_actions():
    agent = make_search_agent()
    # Canonical board: current player (+1) at 0,1; opponent (-1) at 3,4.
    board = torch.tensor([1, 1, 0, -1, -1, 0, 0, 0, 0])

    # The root mover plays +1 at action 2, completing the top row.
    assert agent.get_legal_actions(board, [2]) == []


def test_simulated_actions_are_removed_from_legal_actions():
    agent = make_search_agent()
    # Canonical board: current player (+1) at 0; opponent (-1) at 4.
    board = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, 0])

    assert agent.get_legal_actions(board, [1, 2]) == [3, 5, 6, 7, 8]


def test_simulated_players_alternate_from_canonical_root_without_environment():
    # The root mover is always +1 regardless of which env agent is selected;
    # get_legal_actions never reads agent.env.
    board = torch.tensor([1, 1, 0, -1, -1, 0, 0, 0, 0])
    player_one_env = make_search_agent()
    player_two_env = make_search_agent()
    player_two_env.env.agent_selection = "player_2"

    assert player_one_env.get_legal_actions(board, [2]) == []
    assert player_two_env.get_legal_actions(board, [2]) == []


def test_experience_stores_one_complete_trajectory_and_preserves_terminal_root_value():
    agent = make_search_agent()
    agent.replay_buffer = ReplayBuffer(buffer_size=10, batch_size=1)

    first_policy = torch.zeros(9)
    first_policy[0] = 1
    agent.action_probs = first_policy
    agent.root_value = 0.25
    agent.experience(make_observation(), "player_1", 0, 0, False)

    # Mutating the live search statistics must not mutate stored replay data.
    agent.action_probs[0] = 0
    agent.action_probs[1] = 1
    agent.root_value = 0.75
    agent.experience(make_observation(current=[], opponent=[0]), "player_2", 1, 1, True)

    assert len(agent.replay_buffer.buffer) == 1
    trajectory = agent.replay_buffer.buffer[0]
    assert trajectory["actions"] == [0, 1]
    assert trajectory["turns"] == [0, 1]
    assert trajectory["rewards"] == [0, 1]
    assert trajectory["root_values"] == [0.25, 0.75]
    assert trajectory["target_policies"][0][0].item() == 1
    assert agent.replay_buffer.actions == []


def test_replay_buffer_evicts_oldest_trajectory_at_capacity():
    replay = ReplayBuffer(buffer_size=2, batch_size=1)

    for action in [0, 1, 2]:
        replay.store_step(torch.zeros(9), 0, action, torch.ones(9) / 9, 0, 0)
        replay.store_trajectory()

    assert [trajectory["actions"][0] for trajectory in replay.buffer] == [1, 2]


def test_store_trajectory_deep_copies_pending_data():
    replay = ReplayBuffer(buffer_size=2, batch_size=1)
    observation = torch.zeros(9)
    policy = torch.ones(9) / 9

    replay.store_step(observation, 0, 4, policy, 0, 0.5)
    replay.store_trajectory()
    observation[0] = 1
    policy[4] = 0

    assert replay.buffer[0]["obs"][0][0].item() == 0
    assert replay.buffer[0]["target_policies"][0][4].item() == pytest.approx(1 / 9)


def test_scripted_trajectory_actions_are_legal_and_observations_are_consistent():
    trajectory = make_trajectory()

    for index, action in enumerate(trajectory["actions"]):
        observation = trajectory["obs"][index]
        assert observation[action].item() == 0
        if index + 1 < len(trajectory["obs"]):
            expected_next = observation.clone()
            expected_next[action] = trajectory["turns"][index] + 1
            assert torch.equal(expected_next, trajectory["obs"][index + 1])


def test_real_scripted_game_stores_exactly_one_consistent_trajectory():
    env = raw_env()
    env.reset()
    agent = make_search_agent()
    agent.env = env
    agent.replay_buffer = ReplayBuffer(buffer_size=10, batch_size=1)

    actions = [0, 3, 1, 4, 2]
    for action in actions:
        player = env.agent_selection
        observation = env.observe(player)
        agent.action_probs = torch.zeros(9)
        agent.action_probs[action] = 1
        agent.root_value = 0
        env.step(action)
        agent.experience(
            observation,
            player,
            action,
            env.rewards[player],
            any(env.terminations.values()),
        )

    assert len(agent.replay_buffer.buffer) == 1
    trajectory = agent.replay_buffer.buffer[0]
    assert trajectory["actions"] == actions
    assert trajectory["turns"] == [0, 1, 0, 1, 0]
    assert trajectory["rewards"] == [0, 0, 0, 0, 1]
    assert all(policy[action].item() == 1 for policy, action in zip(
        trajectory["target_policies"], actions
    ))
    for observation, action in zip(trajectory["obs"], actions):
        assert observation[action].item() == 0


def test_replay_value_targets_flip_for_the_opposing_player():
    replay = ReplayBuffer(buffer_size=10, batch_size=1)
    replay.buffer = [make_trajectory()]

    with patch("agents.muzero.muzero.random.randint", side_effect=lambda low, high: low):
        _, _, _, values, _, _ = replay.sample_batch(4, 1.0, "cpu")
    assert values[0, 0].item() == 1
    assert values[0, 1].item() == -1
    assert values[0, 2].item() == 1


def test_replay_value_targets_apply_discount_and_reward_perspective():
    replay = ReplayBuffer(buffer_size=10, batch_size=1)
    replay.buffer = [make_trajectory()]

    with patch("agents.muzero.muzero.random.randint", side_effect=lambda low, high: low):
        _, _, _, values, _, _ = replay.sample_batch(4, 0.5, "cpu")

    assert values[0, 0].item() == 0.5**4
    assert values[0, 1].item() == -(0.5**3)


@pytest.mark.parametrize(
    ("actions", "expected_values"),
    [
        ([0, 3, 1, 4, 8, 5], [-1, 1, -1, 1, -1, 1]),
        ([0, 3, 1, 4, 5, 2, 6, 7, 8], [0] * 9),
    ],
)
def test_replay_terminal_returns_match_each_players_perspective(actions, expected_values):
    replay = ReplayBuffer(buffer_size=10, batch_size=1)
    replay.buffer = [make_terminal_trajectory(actions)]

    for position, expected in enumerate(expected_values):
        with patch(
            "agents.muzero.muzero.random.randint",
            side_effect=[0, position],
        ):
            _, _, _, values, _, _ = replay.sample_batch(0, 1.0, "cpu")
        assert values[0, 0].item() == expected


def test_sample_batch_shapes_masks_and_reward_alignment():
    replay = ReplayBuffer(buffer_size=10, batch_size=2)
    replay.buffer = [make_trajectory()]

    with patch("agents.muzero.muzero.random.randint", side_effect=lambda low, high: low):
        obs, actions, rewards, values, policies, masks = replay.sample_batch(2, 1.0, "cpu")

    assert obs.shape == (2, 9)
    assert actions.shape == (2, 2)
    assert rewards.shape == (2, 3)
    assert values.shape == (2, 3)
    assert policies.shape == (2, 3, 9)
    assert masks.shape == (2, 3, 9)
    assert rewards[0].tolist() == [0, 0, 0]
    assert masks[0, 0].all()
    assert not masks[0, 1, 0]
    assert masks[0, 1].sum().item() == 8


def test_replay_can_sample_last_real_position_and_pads_absorbing_targets():
    replay = ReplayBuffer(buffer_size=10, batch_size=1)
    replay.buffer = [make_trajectory()]

    with patch("agents.muzero.muzero.random.randint", side_effect=lambda low, high: high):
        obs, _, rewards, values, policies, _ = replay.sample_batch(2, 1.0, "cpu")

    assert torch.equal(obs[0], make_trajectory()["obs"][-1])
    assert values[0, 0].item() == 1
    assert rewards[0, 1].item() == 1
    assert policies[0, 1].sum().item() == 0


def test_expansion_normalizes_priors_over_only_legal_actions():
    agent = make_search_agent()
    node = Node(0)

    agent.expansion(
        node,
        torch.zeros(1, 2),
        0,
        torch.tensor([[10.0, 1.0, 0.0, -1.0]]),
        [1, 3],
        [],
    )

    assert set(node.children) == {1, 3}
    assert sum(child.P for child in node.children.values()) == pytest.approx(1)
    assert node.children[1].P > node.children[3].P


def test_selection_follows_highest_puct_until_unexpanded_leaf():
    agent = make_search_agent()
    root = Node(0)
    root.N = 4
    first = Node(1)
    first.N = 3
    first.value_sum = -3
    second = Node(0)
    second.N = 1
    second.value_sum = 1
    root.children = {2: first, 3: second}

    leaf = Node(1)
    first.children = {5: leaf}

    selected, path, actions = agent.selection(root)

    assert selected is leaf
    assert path == [root, first, leaf]
    assert actions == [2, 5]


def test_select_action_records_normalized_visit_policy():
    agent = make_search_agent()
    agent.action_size = 4
    root = Node(0)
    root.N = 10
    root.children = {1: Node(0.5), 3: Node(0.5)}
    root.children[1].N = 7
    root.children[3].N = 3

    assert agent.select_action(root, temperature=0) == 1
    assert agent.action_probs.tolist() == pytest.approx([0, 0.7, 0, 0.3])


def test_search_selects_immediate_winning_move_and_counts_simulations():
    agent = make_search_agent()
    agent.device = torch.device("cpu")
    agent.action_size = 9
    agent.max_iters = 20
    agent.temperature = 0
    agent.dirichlet_alpha = 1
    agent.state_function = lambda obs: torch.zeros((obs.shape[0], 2))

    def dynamics(state, action):
        reward = (action.argmax(dim=1, keepdim=True) == 2).float()
        return state, reward

    def prediction(state):
        logits = torch.zeros((state.shape[0], 9))
        logits[:, 2] = 10
        return logits, torch.zeros((state.shape[0], 1))

    agent.dynamics_function = dynamics
    agent.prediction_function = prediction
    board = torch.tensor([1, 1, 0, 2, 2, 0, 0, 0, 0], dtype=torch.float32)

    action = agent.search(board, temperature=0, add_exploration_noise=False)

    assert action == 2
    assert agent.action_probs.sum().item() == pytest.approx(1)
    assert agent.action_probs[2].item() > 0.5
    assert agent.root_value > 0


def test_act_disables_root_exploration_noise():
    agent = make_search_agent()
    agent.state_function = MagicMock()
    agent.dynamics_function = MagicMock()
    agent.prediction_function = MagicMock()
    observation = make_observation()

    with patch.object(agent, "search", return_value=4) as search:
        assert agent.act(observation) == 4

    search.assert_called_once()
    assert search.call_args.kwargs["add_exploration_noise"] is False


def test_network_interfaces_have_expected_batch_shapes():
    batch_size = 4
    state_function = StateFunction(input_size=9, output_size=8, hidden_size=16)
    dynamics_function = DynamicsFunction(input_size=17, output_size=8, hidden_size=16)
    prediction_function = PredictionFunction(input_size=8, output_size=9, hidden_size=16)

    state = state_function(torch.zeros(batch_size, 9))
    actions = F.one_hot(torch.arange(batch_size) % 9, num_classes=9).float()
    next_state, reward = dynamics_function(state, actions)
    policy, value = prediction_function(next_state)

    assert state.shape == (batch_size, 8)
    assert next_state.shape == (batch_size, 8)
    assert reward.shape == (batch_size, 1)
    assert policy.shape == (batch_size, 9)
    assert value.shape == (batch_size, 1)


def test_encode_actions_produces_orthogonal_one_hot_dynamics_inputs():
    agent = make_search_agent()
    agent.device = torch.device("cpu")
    agent.action_size = 9

    encoded = agent.encode_actions(torch.arange(9))

    assert encoded.shape == (9, 9)
    assert torch.equal(encoded, torch.eye(9))
    assert encoded.sum(dim=1).tolist() == [1.0] * 9


def test_policy_loss_ignores_illegal_target_mass_and_absorbing_states():
    agent = make_search_agent()
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.0, 0.0]])
    masks = torch.tensor([[True, False, True], [True, True, True]])

    loss = agent.policy_loss(logits, targets, masks)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() == 0
    assert logits.grad[1].abs().sum().item() == 0


def test_update_changes_all_network_parameters_with_replay_data():
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

    config = {
        "batch_size": 1,
        "buffer_size": 1,
        "state_size": 8,
        "hidden_size": 8,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_iters": 2,
        "train_iters": 1,
        "gamma": 1.0,
        "k_unroll_steps": 1,
        "temperature": 1.0,
        "dirichlet_alpha": 1.0,
    }
    agent = MuZeroAgent(Environment(), config)
    agent.replay_buffer.buffer = [make_trajectory()]
    networks = [
        agent.state_function,
        agent.dynamics_function,
        agent.prediction_function,
    ]
    before = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]

    with patch.object(agent, "save_model"):
        agent.update()

    for old_parameters, network in zip(before, networks):
        new_parameters = list(network.parameters())
        assert any(
            not torch.equal(old, new)
            for old, new in zip(old_parameters, new_parameters)
        )
        assert all(torch.isfinite(parameter).all() for parameter in new_parameters)
    assert agent.last_loss == pytest.approx(
        sum(agent.last_update_losses) / len(agent.last_update_losses)
    )
    assert len(agent.last_update_losses) == config["train_iters"]


def test_update_waits_for_configured_replay_warmup():
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

    config = {
        "batch_size": 1,
        "buffer_size": 10,
        "min_replay_size": 2,
        "state_size": 8,
        "hidden_size": 8,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_iters": 2,
        "train_iters": 1,
        "gamma": 1.0,
        "k_unroll_steps": 1,
        "temperature": 1.0,
        "dirichlet_alpha": 1.0,
    }
    agent = MuZeroAgent(Environment(), config)
    agent.replay_buffer.buffer = [make_trajectory()]
    agent.update()
    assert agent.training_steps == 0

    agent.replay_buffer.buffer.append(make_trajectory())
    with patch.object(agent, "save_model"):
        agent.update()
    assert agent.training_steps == 1


def test_checkpoint_round_trip_restores_networks_and_optimizer(tmp_path):
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

    config = {
        "batch_size": 1,
        "buffer_size": 1,
        "state_size": 8,
        "hidden_size": 8,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_iters": 2,
        "train_iters": 1,
        "gamma": 1.0,
        "k_unroll_steps": 1,
        "temperature": 1.0,
        "dirichlet_alpha": 1.0,
    }
    source = MuZeroAgent(Environment(), config)
    source.replay_buffer.buffer = [make_trajectory()]
    with patch.object(source, "save_model"):
        source.update()
    source.save_model(tmp_path)

    restored = MuZeroAgent(Environment(), config)
    restored.load_model(tmp_path)

    for source_network, restored_network in zip(
        [source.state_function, source.dynamics_function, source.prediction_function],
        [restored.state_function, restored.dynamics_function, restored.prediction_function],
    ):
        for source_parameter, restored_parameter in zip(
            source_network.parameters(), restored_network.parameters()
        ):
            assert torch.equal(source_parameter, restored_parameter)

    source_optimizer = source.optimizer.state_dict()
    restored_optimizer = restored.optimizer.state_dict()
    assert source_optimizer["param_groups"] == restored_optimizer["param_groups"]
    assert source_optimizer["state"].keys() == restored_optimizer["state"].keys()
    for parameter_id, source_state in source_optimizer["state"].items():
        restored_state = restored_optimizer["state"][parameter_id]
        assert source_state.keys() == restored_state.keys()
        for key, source_value in source_state.items():
            restored_value = restored_state[key]
            if torch.is_tensor(source_value):
                assert torch.equal(source_value, restored_value)
            else:
                assert source_value == restored_value


def test_load_model_with_missing_checkpoint_files_keeps_initial_weights(tmp_path):
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

    config = {
        "batch_size": 1,
        "buffer_size": 1,
        "state_size": 8,
        "hidden_size": 8,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_iters": 2,
        "train_iters": 1,
        "gamma": 1.0,
        "k_unroll_steps": 1,
        "temperature": 1.0,
        "dirichlet_alpha": 1.0,
    }
    agent = MuZeroAgent(Environment(), config)
    before = [parameter.detach().clone() for parameter in agent.state_function.parameters()]

    agent.load_model(tmp_path)

    assert all(
        torch.equal(old, new)
        for old, new in zip(before, agent.state_function.parameters())
    )
