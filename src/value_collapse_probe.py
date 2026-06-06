"""
Confirm/refute hypothesis (a): the net's value head has collapsed to pessimism
for the SECOND player (O), i.e. it predicts ~-1 for O-to-move positions that are
actually drawable, while staying calibrated/optimistic for X-to-move positions.

Method (ground-truth, not eyeballed):
  1. Enumerate every reachable tic-tac-toe position (X moves first).
  2. Compute its exact minimax value from the MOVER's perspective (+1/0/-1).
  3. Build the canonical (mover-relative) board the net trains on and read the
     net's predicted value (categorical head -> support_to_scalar).
  4. Bucket by side-to-move x true value; report mean/std predicted value.

If O-to-move / true=0 (drawable) positions have net mean ~-1 with small spread,
the value head has collapsed for the defender. The X-to-move buckets are the
control.
"""
import os, shutil, sys
from functools import lru_cache
import numpy as np
import torch
from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

CONFIG = {
    'batch_size': 128, 'buffer_size': 1000, 'min_replay_size': 100,
    'state_size': 16, 'hidden_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4,
    'max_iters': 50, 'train_iters': 2, 'checkpoint_interval': 10_000,
    'gamma': 1.0, 'k_unroll_steps': 5, 'temperature': 1.0, 'dirichlet_alpha': 1.0,
    'num_bins': 51, 'support_limit': 1.0, 'value_transform': True,
}
CKPT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', 'muzero')
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]


def snapshot():
    dst = '/tmp/mu_ckpt_collapse'; os.makedirs(dst, exist_ok=True)
    for f in ('mu_state_rep_params.pth.tar', 'mu_dyn_func_params.pth.tar',
              'mu_pred_func_params.pth.tar', 'mu_optimizer_params.pth.tar'):
        s = os.path.join(CKPT_SRC, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
    return dst


def winner(board):
    for a, b, c in LINES:
        if board[a] != '.' and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=None)
def minimax(board, mover):
    """Exact value from `mover`'s perspective: +1 win, 0 draw, -1 loss."""
    w = winner(board)
    if w is not None:
        # someone already won; the player who just moved was the opponent
        return -1  # mover faces a completed line made by opponent -> mover lost
    if '.' not in board:
        return 0
    opp = 'O' if mover == 'X' else 'X'
    best = -2
    for i in range(9):
        if board[i] == '.':
            nb = board[:i] + mover + board[i+1:]
            if winner(nb) == mover:
                val = 1
            else:
                val = -minimax(nb, opp)
            best = max(best, val)
    return best


def enumerate_states():
    """All reachable non-terminal positions: (board, mover)."""
    out = []
    seen = set()
    def rec(board, mover):
        if (board, mover) in seen:
            return
        seen.add((board, mover))
        if winner(board) is not None or '.' not in board:
            return
        out.append((board, mover))
        opp = 'O' if mover == 'X' else 'X'
        for i in range(9):
            if board[i] == '.':
                rec(board[:i] + mover + board[i+1:], opp)
    rec('.' * 9, 'X')
    return out


def canonical_obs(board, mover):
    """+1 = mover's pieces, -1 = opponent's, 0 = empty (flattened 9)."""
    opp = 'O' if mover == 'X' else 'X'
    v = np.zeros(9, dtype=np.float32)
    for i, ch in enumerate(board):
        if ch == mover: v[i] = 1.0
        elif ch == opp: v[i] = -1.0
    return torch.tensor(v)


def main():
    ckpt = snapshot()
    env = tictactoe.env()
    agent = MuZeroAgent(environment=env, config=dict(CONFIG))
    agent.load_model(directory=ckpt)
    env.close()
    agent.state_function.eval(); agent.dynamics_function.eval(); agent.prediction_function.eval()

    states = enumerate_states()
    obs = torch.stack([canonical_obs(b, m) for b, m in states]).to(agent.device)
    with torch.no_grad():
        s = agent.state_function(obs)
        _, vlogits = agent.prediction_function(s)
        pred = agent.support_to_scalar(vlogits).reshape(-1).cpu().numpy()

    true = np.array([minimax(b, m) for b, m in states])
    side = np.array(['X' if m == 'X' else 'O' for b, m in states])

    print(f"reachable non-terminal positions: {len(states)}\n")
    print(f"{'side':>5} {'true':>5} {'count':>6} {'net_mean':>9} {'net_std':>8} "
          f"{'net_min':>8} {'net_max':>8}")
    for sd in ('X', 'O'):
        for tv in (1, 0, -1):
            m = (side == sd) & (true == tv)
            if m.sum() == 0:
                continue
            p = pred[m]
            print(f"{sd:>5} {tv:>5} {m.sum():>6} {p.mean():>9.3f} {p.std():>8.3f} "
                  f"{p.min():>8.3f} {p.max():>8.3f}")

    # headline: drawable defender positions
    print("\n--- headline ---")
    for sd in ('O', 'X'):
        m = (side == sd) & (true == 0)
        if m.sum():
            print(f"{sd}-to-move, TRUE=0 (drawable): net mean={pred[m].mean():+.3f} "
                  f"std={pred[m].std():.3f}  -> {'COLLAPSED to pessimism' if pred[m].mean() < -0.6 else 'not collapsed'}")
    # correlation of net value with truth, per side
    for sd in ('X', 'O'):
        m = side == sd
        c = np.corrcoef(pred[m], true[m])[0, 1]
        print(f"{sd}-to-move: corr(net, minimax) = {c:+.3f}  (mean net={pred[m].mean():+.3f}, mean true={true[m].mean():+.3f})")


if __name__ == "__main__":
    main()
