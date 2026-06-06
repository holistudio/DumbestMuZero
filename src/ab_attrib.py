"""
A/B attribution harness for the P2-vs-random regression.

Isolated from the live train.py run: builds its OWN fresh MuZeroAgent per
condition, never calls save_model, writes only to /tmp. Each condition varies
ONE factor from the NEW baseline so we can attribute the regression:

  C0_new      : value_loss_weight=1.0 , buffer=20000   (reproduce regression)
  C1_valuefix : value_loss_weight=0.25, buffer=20000   (only the value coeff)
  C2_bufferfix: value_loss_weight=1.0 , buffer=1000    (only the buffer)
  C3_both     : value_loss_weight=0.25, buffer=1000    (old-like control)

For each, we self-play TRAIN_EPS games (same loop as train.py: update every 5
eps, 2 train_iters), and every EVAL_EVERY eps evaluate the agent as PLAYER 2
against a random opponent over EVAL_GAMES games. We log P2 win/loss over
training and report the slope. The factor whose "fix" flattens/reverses the P2
decline is the cause.

Usage:  python ab_attrib.py <label> <value_loss_weight> <buffer_size> <train_eps> <seed>
Writes: /tmp/ab_<label>.json
"""
import json, os, sys, random
import numpy as np
import torch
from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

EVAL_EVERY = int(os.getenv("EVAL_EVERY", "500"))
EVAL_GAMES = int(os.getenv("EVAL_GAMES", "150"))
UPDATE_INTERVAL = 5


def base_config(value_loss_weight, buffer_size, noise_on_greedy=True):
    # td_steps=None ties the value-target bootstrap horizon to k_unroll_steps
    # (=5, current behavior). Set TD_STEPS>=9 to use the full Monte-Carlo game
    # return (the canonical MuZero board-game value target).
    td_env = os.getenv("TD_STEPS", "")
    td_steps = int(td_env) if td_env else None
    # Optional temperature schedule, e.g. "200:1.0,400:0.5,1000000000:0.25"
    sched_env = os.getenv("TEMP_SCHEDULE", "")
    temp_schedule = None
    if sched_env:
        temp_schedule = [tuple(float(x) for x in pair.split(":"))
                         for pair in sched_env.split(",")]
    return {
        'batch_size': 128, 'buffer_size': buffer_size, 'min_replay_size': 100,
        'state_size': 16, 'hidden_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4,
        'max_iters': int(os.getenv("MAX_ITERS", "50")), 'train_iters': 2, 'checkpoint_interval': 10**9,
        'gamma': 1.0, 'k_unroll_steps': 5, 'td_steps': td_steps,
        'temperature': 1.0, 'dirichlet_alpha': 0.1, 'temp_schedule': temp_schedule,
        'num_bins': 51, 'support_limit': 1.0, 'value_transform': True,
        'lr_decay_rate': 0.1, 'lr_decay_steps': 200000,
        'value_loss_weight': value_loss_weight,
        'noise_on_greedy': noise_on_greedy,
        'PER': os.getenv("PER", "0") == "1",
        'PER_alpha': float(os.getenv("PER_ALPHA", "0.5")),
        'PER_beta': float(os.getenv("PER_BETA", "1.0")),
    }


def eval_p2(agent, n_games, base_seed):
    """Agent plays player_2 vs a random player_1. Returns (win,loss,draw)."""
    env = tictactoe.env()
    wld = [0, 0, 0]
    for ep in range(n_games):
        s = base_seed + ep
        env.reset(seed=s)
        env.action_space('player_1').seed(s)
        for a in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            elif a == 'player_1':
                action = int(env.action_space(a).sample(obs["action_mask"]))
            else:
                action = agent.act(obs)
            env.step(action)
            if len(env.terminations.keys()) == 2 and env.terminations[a]:
                r = env.rewards['player_2']
                wld[0 if r == 1 else 1 if r == -1 else 2] += 1
        # agent_iter ends each game; loop resets on next reset
    env.close()
    return wld


def run(label, vlw, buf, train_eps, seed, noise_on_greedy=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = tictactoe.env()
    for i, name in enumerate(env.possible_agents):
        env.action_space(name).seed(seed + i)
    agent = MuZeroAgent(environment=env,
                        config=base_config(vlw, buf, noise_on_greedy))

    records = []
    for ep in range(train_eps):
        env.reset(seed=seed + ep)
        for a in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                action = agent.step(obs)
            env.step(action)
            done = any(env.terminations.values()) or any(env.truncations.values())
            agent.experience(obs, a, action, env.rewards[a], done)
            if done:
                break
        if (ep + 1) % UPDATE_INTERVAL == 0:
            agent.update()
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == train_eps:
            w, l, d = eval_p2(agent, EVAL_GAMES, seed + 500_000)
            tot = w + l + d
            rec = {"ep": ep, "p2_win": w, "p2_loss": l, "p2_draw": d,
                   "p2_win_pct": w / tot * 100, "p2_loss_pct": l / tot * 100}
            records.append(rec)
            print(f"[{label}] ep={ep+1:>5} P2 win%={rec['p2_win_pct']:5.1f} "
                  f"loss%={rec['p2_loss_pct']:5.1f} (updates={agent.training_steps})",
                  flush=True)
    env.close()

    eps = np.array([r["ep"] for r in records], dtype=float)
    win = np.array([r["p2_win_pct"] for r in records])
    loss = np.array([r["p2_loss_pct"] for r in records])
    slope_win = float(np.polyfit(eps, win, 1)[0] * 1000) if len(eps) > 1 else 0.0
    slope_loss = float(np.polyfit(eps, loss, 1)[0] * 1000) if len(eps) > 1 else 0.0
    out = {"label": label, "value_loss_weight": vlw, "buffer_size": buf,
           "td_steps": os.getenv("TD_STEPS", "") or "k_unroll(5)",
           "seed": seed, "train_eps": train_eps, "noise_on_greedy": noise_on_greedy,
           "p2_win_slope_per1k": slope_win, "p2_loss_slope_per1k": slope_loss,
           "records": records}
    with open(f"/tmp/ab_{label}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{label}] DONE  P2 win slope={slope_win:+.2f}/1k  "
          f"loss slope={slope_loss:+.2f}/1k", flush=True)
    return out


if __name__ == "__main__":
    label = sys.argv[1]
    vlw = float(sys.argv[2])
    buf = int(sys.argv[3])
    train_eps = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    # 7th arg: noise_on_greedy ("1"/"0"); default 1 = current behavior
    nog = (sys.argv[6] != "0") if len(sys.argv) > 6 else True
    run(label, vlw, buf, train_eps, seed, nog)
