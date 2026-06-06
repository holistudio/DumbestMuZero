import argparse
import math
import random

import torch
import torch.nn.functional as F

from agents.muzero.muzero import ReplayBuffer
from envs.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board


def random_trajectory():
    board = Board()
    observations = []
    turns = []
    actions = []
    rewards = []
    target_policies = []
    root_values = []
    player = 0

    while board.game_status() == TTT_GAME_NOT_OVER:
        legal_actions = board.legal_moves()
        action = random.choice(legal_actions)
        observation = torch.tensor(board.squares, dtype=torch.float32)
        policy = torch.zeros(9)
        policy[legal_actions] = 1 / len(legal_actions)

        board.play_turn(player, action)
        game_status = board.game_status()

        observations.append(observation)
        turns.append(player)
        actions.append(action)
        rewards.append(0 if game_status in (TTT_GAME_NOT_OVER, TTT_TIE) else 1)
        target_policies.append(policy)
        root_values.append(0)
        player ^= 1

    return {
        "obs": observations,
        "turns": turns,
        "actions": actions,
        "rewards": rewards,
        "target_policies": target_policies,
        "root_values": root_values,
    }


def uniform_legal_policy_loss(target_policy, legal_mask):
    legal_counts = legal_mask.sum(dim=1)
    valid = target_policy.sum(dim=1) > 0
    if valid.any():
        return torch.log(legal_counts[valid].float()).mean()
    return torch.tensor(0.0)


def calculate_baseline(games, batches, batch_size, unroll_steps, gamma, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    replay = ReplayBuffer(buffer_size=games, batch_size=batch_size)
    replay.buffer = [random_trajectory() for _ in range(games)]

    totals = []
    policy_totals = []
    value_totals = []
    reward_totals = []

    for _ in range(batches):
        _, _, target_rewards, target_values, target_policies, legal_masks = (
            replay.sample_batch(unroll_steps, gamma, "cpu")
        )

        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        reward_loss = torch.tensor(0.0)

        for step in range(unroll_steps + 1):
            policy_loss += uniform_legal_policy_loss(
                target_policies[:, step], legal_masks[:, step]
            )
            value_loss += F.mse_loss(
                torch.zeros_like(target_values[:, step]), target_values[:, step]
            )
            if step > 0:
                reward_loss += F.mse_loss(
                    torch.zeros_like(target_rewards[:, step]), target_rewards[:, step]
                )

        total = policy_loss + value_loss + reward_loss
        totals.append(total.item())
        policy_totals.append(policy_loss.item())
        value_totals.append(value_loss.item())
        reward_totals.append(reward_loss.item())

    def mean(values):
        return sum(values) / len(values)

    def standard_error(values):
        average = mean(values)
        variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance / len(values))

    return {
        "total": mean(totals),
        "total_standard_error": standard_error(totals),
        "policy": mean(policy_totals),
        "value": mean(value_totals),
        "reward": mean(reward_totals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = calculate_baseline(
        games=args.games,
        batches=args.batches,
        batch_size=args.batch_size,
        unroll_steps=args.unroll_steps,
        gamma=args.gamma,
        seed=args.seed,
    )
    print(
        f"Random-agent baseline train_loss={result['total']:.6f} "
        f"+/- {result['total_standard_error']:.6f}"
    )
    print(
        f"components policy={result['policy']:.6f} "
        f"value={result['value']:.6f} reward={result['reward']:.6f}"
    )


if __name__ == "__main__":
    main()
