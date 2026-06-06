"""
Punishment-rate diagnostic.

For self-play games under both (a) the training-time policy (temp schedule +
Dirichlet noise) and (b) the greedy policy, we track every "must-block" ply:
the mover faces exactly one opponent open 2-in-a-row and cannot win this turn.

We measure:
  block_rate      = P(mover plays the blocking cell | must-block)
  punish_rate     = P(attacker completes the line next ply | mover did NOT block)

Logic of the experiment: if punish_rate is HIGH, the defensive value targets are
correct (failing to block really does lead to an immediate loss in the data), so
the net's threat-blindness is a target-staleness/learning problem, not a data
problem -- and "more exploration" cannot fix it. If punish_rate is LOW, the
attacker fails to punish, the targets are corrupted at the source, and the fix is
a sharper attacker, not more exploration.
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
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]


def snapshot():
    dst = '/tmp/mu_ckpt_punish'; os.makedirs(dst, exist_ok=True)
    for f in ('mu_state_rep_params.pth.tar', 'mu_dyn_func_params.pth.tar',
              'mu_pred_func_params.pth.tar', 'mu_optimizer_params.pth.tar'):
        s = os.path.join(CKPT_SRC, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
    return dst


def open_lines(board, mark):
    """Return list of (blocking_cell) for each line with 2*mark + 1 empty."""
    out = []
    for a, b, c in LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(mark) == 2 and cells.count('.') == 1:
            empty = [x for x in (a, b, c) if board[x] == '.'][0]
            out.append(empty)
    return out


def wins(board, mark):
    return any(all(board[i] == mark for i in line) for line in LINES)


def run(mode, n_games, base_seed, ckpt_dir=None):
    ckpt = ckpt_dir if ckpt_dir is not None else snapshot()
    env = tictactoe.env()
    agent = MuZeroAgent(environment=env, config=dict(CONFIG))
    agent.load_model(directory=ckpt)
    env.close()
    agent.state_function.eval(); agent.dynamics_function.eval(); agent.prediction_function.eval()

    n_mustblock = 0       # exactly-one-threat, mover can't win
    n_blocked = 0         # mover played a blocking cell
    n_notblocked = 0      # mover failed to block
    n_punished = 0        # attacker completed the line on the very next ply
    outcomes = {'p1_win': 0, 'p2_win': 0, 'draw': 0}
    plies_per_game = []

    for g in range(n_games):
        env = tictactoe.env()
        seed = base_seed + g
        env.reset(seed=seed)
        env.action_space('player_1').seed(seed)
        env.action_space('player_2').seed(seed)
        board = ['.'] * 9
        pending = None     # (defender_mark, attacker_mark, blocking_cells) awaiting attacker reply
        nplies = 0

        for a in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None); continue

            mark = 'X' if a == 'player_1' else 'O'
            opp = 'O' if mark == 'X' else 'X'

            # Resolve a pending "defender failed to block" -> did THIS mover (attacker) punish?
            if pending is not None:
                d_mark, atk_mark, _ = pending
                if mark == atk_mark:
                    # attacker is to move; check after the move whether they win
                    pass  # resolved after we compute the action below
            # Detect must-block for the CURRENT mover (before they move)
            opp_threats = open_lines(board, opp)
            my_threats = open_lines(board, mark)
            is_mustblock = (len(opp_threats) == 1 and len(my_threats) == 0)

            # choose action
            obs = agent.preprocess_obs(observation)
            if mode == 'greedy':
                action = agent.search(obs, 0.0, add_exploration_noise=False)
            else:
                blanks = int((obs == 0).sum().item())
                temp = agent.temperature if blanks > 5 else 0.0
                action = agent.search(obs, temp, add_exploration_noise=True)

            # apply
            board[action] = mark
            nplies += 1

            # if we had a pending punishment opportunity and this mover is the attacker:
            if pending is not None and mark == pending[1]:
                if wins(board, mark):
                    n_punished += 1
                pending = None  # opportunity consumed either way

            # record must-block bookkeeping for the mover that just moved
            if is_mustblock:
                n_mustblock += 1
                if action in opp_threats:
                    n_blocked += 1
                    pending = None
                else:
                    n_notblocked += 1
                    # set up punishment check: attacker is `opp`, still has an open line
                    pending = (mark, opp, opp_threats)

            env.step(action)
            if any(env.terminations.values()) or any(env.truncations.values()):
                break
        r = env.rewards
        if r['player_1'] == 1:
            outcomes['p1_win'] += 1
        elif r['player_2'] == 1:
            outcomes['p2_win'] += 1
        else:
            outcomes['draw'] += 1
        plies_per_game.append(nplies)
        env.close()

    print(f"\n=== mode={mode}  games={n_games} ===")
    print(f"must-block plies:        {n_mustblock}")
    print(f"  blocked:               {n_blocked}  "
          f"(block_rate = {n_blocked/max(n_mustblock,1):.3f})")
    print(f"  NOT blocked:           {n_notblocked}")
    print(f"  of those, punished:    {n_punished}  "
          f"(punish_rate = {n_punished/max(n_notblocked,1):.3f})")
    g = max(n_games, 1)
    print(f"outcomes: p1_win={outcomes['p1_win']} ({outcomes['p1_win']/g:.1%})  "
          f"p2_win={outcomes['p2_win']} ({outcomes['p2_win']/g:.1%})  "
          f"draw={outcomes['draw']} ({outcomes['draw']/g:.1%})  "
          f"decisive={ (outcomes['p1_win']+outcomes['p2_win'])/g:.1%}")
    print(f"avg plies/game: {np.mean(plies_per_game):.2f}")
    return n_mustblock, n_blocked, n_notblocked, n_punished


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mode = sys.argv[2] if len(sys.argv) > 2 else 'both'
    ckpt_dir = sys.argv[3] if len(sys.argv) > 3 else None  # load a specific checkpoint dir
    if mode in ('both', 'selfplay'):
        run('selfplay', n, base_seed=900_000, ckpt_dir=ckpt_dir)
    if mode in ('both', 'greedy'):
        run('greedy', n, base_seed=950_000, ckpt_dir=ckpt_dir)
