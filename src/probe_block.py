"""
Probe a single tactical position with the real MCTS:

  . O .       O = player_2 (to move), X = player_1
  . X X       X has cells 4 and 5 -> threatens to win by playing 3.
  . . .       Correct move for O is to BLOCK at cell 3.

We run the actual agent.search() and dump per-action root statistics
(prior P, visits N, value Q from O's perspective, predicted reward R),
sweeping the simulation budget to see whether deeper search finds the block.
"""
import os, shutil
import numpy as np
import torch
from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

CONFIG = {
    'batch_size': 128, 'buffer_size': 1000, 'min_replay_size': 100,
    'state_size': 16, 'hidden_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4,
    'max_iters': 50, 'train_iters': 2, 'checkpoint_interval': 10_000,
    'gamma': 1.0, 'k_unroll_steps': 5, 'temperature': 1.0, 'dirichlet_alpha': 1.0,
}
CKPT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', 'muzero')


def snapshot():
    dst = '/tmp/mu_ckpt_probe'; os.makedirs(dst, exist_ok=True)
    for f in ('mu_state_rep_params.pth.tar', 'mu_dyn_func_params.pth.tar',
              'mu_pred_func_params.pth.tar', 'mu_optimizer_params.pth.tar'):
        s = os.path.join(CKPT_SRC, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
    return dst


def build_obs():
    """O (current player) at cell 1; X (opponent) at cells 4,5."""
    plane0 = np.zeros(9, dtype=np.int8); plane0[1] = 1          # O = current player
    plane1 = np.zeros(9, dtype=np.int8); plane1[4] = 1; plane1[5] = 1  # X = opponent
    board = np.stack([plane0.reshape(3, 3), plane1.reshape(3, 3)], axis=-1)
    mask = 1 - (plane0 + plane1)
    return {"observation": board, "action_mask": mask.astype(np.int8)}


def dump_root(agent, label):
    root = agent.search_root
    sum_visits = root.N
    print(f"\n{label}  | root_value (O persp) = {root.mean_value():+.3f} | total sims = {sum_visits}")
    print(f"  {'act':>3} {'prior P':>8} {'visits N':>9} {'visit%':>7} "
          f"{'Q(O persp)':>11} {'child.R':>8}  note")
    rows = []
    for a, ch in root.children.items():
        q_o = -ch.mean_value()            # child value is opponent perspective; negate for O
        rows.append((a, ch.P, ch.N, ch.N / sum_visits, q_o, ch.R))
    for a, P, N, frac, q, R in sorted(rows, key=lambda r: -r[2]):
        note = 'BLOCK (correct)' if a == 3 else ''
        Pv = P.item() if torch.is_tensor(P) else float(P)
        print(f"  {a:>3} {Pv:>8.3f} {N:>9} {frac*100:>6.1f}% {q:>11.3f} {R:>8.3f}  {note}")


if __name__ == "__main__":
    ckpt = snapshot()
    env = tictactoe.env()
    agent = MuZeroAgent(environment=env, config=dict(CONFIG))
    agent.load_model(directory=ckpt)
    env.close()

    print("POSITION:  . O . / . X X / . . .   (O to move; X threatens 3)")
    print("Correct play: O must block at cell 3.\n")

    obs = agent.preprocess_obs(build_obs())
    print("canonical board (flat):", obs.numpy().astype(int).tolist(),
          " (+1=O, -1=X, 0=empty)")

    for sims in (25, 50, 100, 200, 400):
        agent.max_iters = sims
        agent.search(obs, 0.0, add_exploration_noise=False)
        dump_root(agent, f"=== MCTS with {sims} sims ===")
