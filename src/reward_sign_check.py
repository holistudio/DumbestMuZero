"""
Diagnostic: prove rewards + value targets are assigned with correct signs
for BOTH players, by driving the REAL env + agent.experience() code path
(mirroring train.py exactly) through scripted P1-win and P2-win games.
"""
import torch
from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

CONFIG = {
    'batch_size': 4, 'buffer_size': 50, 'min_replay_size': 1,
    'state_size': 16, 'hidden_size': 64, 'lr': 1e-3, 'weight_decay': 1e-4,
    'max_iters': 1, 'train_iters': 1, 'checkpoint_interval': 10_000,
    'gamma': 1.0, 'k_unroll_steps': 5, 'temperature': 1.0, 'dirichlet_alpha': 1.0,
}


def play_scripted(scripted_actions, label):
    """Replay a fixed action sequence through the real env, mirroring train.py's
    experience() call. Returns the stored trajectory dict."""
    env = tictactoe.env()
    agent = MuZeroAgent(environment=env, config=dict(CONFIG))
    env.reset(seed=0)
    move_iter = iter(scripted_actions)

    for a in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = next(move_iter)
            # experience() clones these; set markers so it runs (values irrelevant
            # to reward-sign correctness, root_value=0 -> pure Monte-Carlo return).
            agent.action_probs = torch.zeros(agent.action_size)
            agent.root_value = 0.0
        env.step(action)
        terminal = any(env.terminations.values()) or any(env.truncations.values())
        agent.experience(observation, a, action, env.rewards[a], terminal)
        if terminal:
            break

    traj = agent.replay_buffer.buffer[-1]
    print(f"\n===== {label} =====")
    print(f"final env.rewards: {env.rewards}")
    print(f"{'pos':>3} {'player':>7} {'action':>6} {'reward':>7} {'value_target':>12}  ok?")

    turns, actions, rewards = traj['turns'], traj['actions'], traj['root_values']
    rewards = traj['rewards']
    root_values = traj['root_values']

    # Ground-truth final outcome per absolute player from env.rewards.
    # player_turn encoding: 0 == player_1, 1 == player_2 (see experience()).
    final = {0: env.rewards['player_1'], 1: env.rewards['player_2']}

    all_ok = True
    n = len(turns)
    for ix in range(n):
        cur = turns[ix]
        # Pure Monte-Carlo return from this position's mover perspective, using
        # the EXACT sign logic from ReplayBuffer.sample_batch (gamma=1, no bootstrap).
        value = 0.0
        for j, r in enumerate(rewards[ix:]):
            if turns[ix + j] == cur:
                value += r
            else:
                value -= r
        expected = final[cur]            # +1 if this mover ultimately won, -1 if lost, 0 draw
        ok = (value == expected)
        all_ok &= ok
        pname = 'player_1' if cur == 0 else 'player_2'
        print(f"{ix:>3} {pname:>7} {actions[ix]!s:>6} {rewards[ix]:>7} "
              f"{value:>12.1f}  {'OK' if ok else 'WRONG (exp %s)' % expected}")
    print(f"ALL POSITIONS CORRECT: {all_ok}")
    return all_ok


# P2 wins: P1=0, P2=3, P1=1, P2=4, P1=8, P2=5 -> P2 completes middle row 3,4,5
p2_ok = play_scripted([0, 3, 1, 4, 8, 5], "P2 WINS")
# P1 wins: P1=0, P2=3, P1=1, P2=4, P1=2 -> P1 completes top row 0,1,2
p1_ok = play_scripted([0, 3, 1, 4, 2], "P1 WINS")

print("\n================ SUMMARY ================")
print(f"P2-win game targets correct: {p2_ok}")
print(f"P1-win game targets correct: {p1_ok}")
