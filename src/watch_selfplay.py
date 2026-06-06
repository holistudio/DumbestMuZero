"""
Watch what the MuZero networks are doing during self-play.

Loads the current checkpoint and plays annotated self-play games, printing for
each ply: the absolute board (X/O), the MCTS policy (visit %), the root value
estimate (from the mover's perspective), and the chosen move.
"""
import os
import shutil
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

CKPT_SRC = os.path.dirname(os.path.abspath(tictactoe.__file__)).replace(
    'envs/tictactoe', 'agents/muzero')


def snapshot_checkpoint():
    """Copy live checkpoint files to /tmp to avoid reading a half-written file."""
    dst = '/tmp/mu_ckpt_snapshot'
    os.makedirs(dst, exist_ok=True)
    for f in ('mu_state_rep_params.pth.tar', 'mu_dyn_func_params.pth.tar',
              'mu_pred_func_params.pth.tar', 'mu_optimizer_params.pth.tar'):
        src = os.path.join(CKPT_SRC, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, f))
    return dst


def absolute_board(observation, mover):
    """Return a 3x3 array of strings X/O/. from the canonical observation.
    plane0 = mover's pieces, plane1 = opponent's pieces."""
    p0 = np.array(observation["observation"][:, :, 0]).reshape(9)
    p1 = np.array(observation["observation"][:, :, 1]).reshape(9)
    x = p0 if mover == 'player_1' else p1   # player_1 is X
    o = p1 if mover == 'player_1' else p0
    cells = []
    for i in range(9):
        cells.append('X' if x[i] == 1 else 'O' if o[i] == 1 else '.')
    return cells


def fmt_side_by_side(board_cells, policy):
    """Render board and policy(%) grids side by side."""
    lines = []
    lines.append("   board        policy (visit %)")
    for r in range(3):
        brow = " ".join(board_cells[r*3 + c] for c in range(3))
        prow = " ".join(f"{policy[r*3+c]*100:3.0f}" for c in range(3))
        lines.append(f"  {brow}      {prow}")
    return "\n".join(lines)


def play_annotated_game(agent, seed, mode='greedy'):
    """mode='greedy': honest deterministic best line (no noise).
       mode='selfplay': training-time policy (Dirichlet noise + temp schedule)."""
    env = tictactoe.env()
    env.reset(seed=seed)
    env.action_space('player_1').seed(seed)
    env.action_space('player_2').seed(seed)

    agent.state_function.eval()
    agent.dynamics_function.eval()
    agent.prediction_function.eval()

    ply = 0
    for a in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            continue

        obs = agent.preprocess_obs(observation)
        if mode == 'greedy':
            action = agent.search(obs, 0.0, add_exploration_noise=False)
        else:
            # mirror step(): explore early (T=1 while >5 blanks) then greedy,
            # with root Dirichlet noise on as in real self-play.
            blanks = int((obs == 0).sum().item())
            temp = agent.temperature if blanks > 5 else 0.0
            action = agent.search(obs, temp, add_exploration_noise=True)
        policy = agent.action_probs.numpy()           # visit distribution over 9 cells
        value = agent.root_value                       # mover-perspective value

        ply += 1
        mark = 'X' if a == 'player_1' else 'O'
        print(f"\n--- ply {ply}: {a} ({mark}) to move | root_value = {value:+.3f} ---")
        print(fmt_side_by_side(absolute_board(observation, a), policy))
        print(f"  chosen action = {action}  (visit% = {policy[action]*100:.0f})")

        env.step(action)
        if any(env.terminations.values()) or any(env.truncations.values()):
            break

    result = env.rewards
    print(f"\n=== RESULT: player_1={result['player_1']:+d}  "
          f"player_2={result['player_2']:+d}  "
          f"({'P1 win' if result['player_1']==1 else 'P2 win' if result['player_2']==1 else 'draw'}) ===")
    env.close()


def play_vs_random(agent, agent_side, seed):
    """Agent plays `agent_side` greedily (eval mode); the other side is random.
    Annotates the agent's moves with policy + value; shows random's moves plainly."""
    env = tictactoe.env()
    env.reset(seed=seed)
    other = 'player_2' if agent_side == 'player_1' else 'player_1'
    env.action_space(other).seed(seed)

    agent.state_function.eval(); agent.dynamics_function.eval(); agent.prediction_function.eval()
    ply = 0
    for a in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None); continue
        ply += 1
        mark = 'X' if a == 'player_1' else 'O'
        if a == agent_side:
            obs = agent.preprocess_obs(observation)
            action = agent.search(obs, 0.0, add_exploration_noise=False)
            policy = agent.action_probs.numpy()
            value = agent.root_value
            print(f"\n--- ply {ply}: AGENT {a} ({mark}) | root_value={value:+.3f} ---")
            print(fmt_side_by_side(absolute_board(observation, a), policy))
            print(f"  chosen action = {action}  (visit% = {policy[action]*100:.0f})")
        else:
            mask = observation["action_mask"]
            action = int(env.action_space(a).sample(mask))
            print(f"\n--- ply {ply}: random {a} ({mark}) plays {action} ---")
            print("   " + "\n   ".join(
                " ".join(absolute_board(observation, a)[r*3+c] for c in range(3)) for r in range(3)))
        env.step(action)
        if any(env.terminations.values()) or any(env.truncations.values()):
            break
    r = env.rewards
    tag = ('P1 win' if r['player_1'] == 1 else 'P2 win' if r['player_2'] == 1 else 'draw')
    agent_res = r[agent_side]
    print(f"\n=== RESULT: {tag} | agent({agent_side}) {'WON' if agent_res==1 else 'LOST' if agent_res==-1 else 'DREW'} ===")
    env.close()


if __name__ == "__main__":
    import sys
    ckpt_dir = snapshot_checkpoint()
    env = tictactoe.env()
    agent = MuZeroAgent(environment=env, config=dict(CONFIG))
    agent.load_model(directory=ckpt_dir)
    print(f"Loaded checkpoint from {CKPT_SRC} (snapshot @ {ckpt_dir})")
    env.close()

    mode = sys.argv[1] if len(sys.argv) > 1 else 'selfplay'
    if mode == 'vsrandom':
        # Agent as P2 (the weak side) vs random P1. Eval seeds: SEED+200_000+ep.
        for ep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
            seed = 42 + 200_000 + ep
            print("\n" + "=" * 52)
            print(f"AGENT=P2 vs RANDOM  (eval seed ep={ep}, seed={seed})")
            print("=" * 52)
            play_vs_random(agent, agent_side='player_2', seed=seed)
    else:
        print("\n" + "=" * 52)
        print("GAME 1 — honest greedy self-play (net's best line)")
        print("=" * 52)
        play_annotated_game(agent, seed=0, mode='greedy')
        for g, seed in enumerate([1, 7], start=2):
            print("\n" + "=" * 52)
            print(f"GAME {g} — training-time self-play (noise + temp), seed={seed}")
            print("=" * 52)
            play_annotated_game(agent, seed=seed, mode='selfplay')
