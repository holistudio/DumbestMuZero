"""
Rigorous check of ReplayBuffer.sample_batch's VALUE TARGET signs, including the
bootstrap branch (which reward_sign_check.py never exercises because it sets
root_value=0 and sums all future rewards instead of capping at td_steps).

We feed ONE hand-built trajectory with:
  - strictly alternating turns [0,1,0,1,...]  (real shared-agent self-play)
  - distinct, recognizable root_values so the bootstrap term is identifiable
  - a single terminal reward, so the reward-sign path is also covered
and then call the REAL sample_batch. Each sampled obs uniquely identifies its
position ix (we encode obs[i] = i+1), so we can map every returned step-0 value
target back to its source position and compare to a hand-computed ground truth.

Ground-truth value target at position i (gamma=1, td=k_unroll_steps=5), matching
the intended negamax semantics:
    bootstrap = root_values[i+5] * (+1 if turns[i+5]==turns[i] else -1)   if i+5 < L else 0
    plus, for each reward r at index t in [i, i+5):
        + r if turns[t]==turns[i] else  - r
"""
import torch
from agents.muzero.muzero import ReplayBuffer

K = 5          # k_unroll_steps (real config)
GAMMA = 1.0
L = 8          # trajectory length: positions 0,1,2 use bootstrap; 3..7 don't


def ground_truth(turns, rewards, root_values, i):
    boot_ix = i + K
    if boot_ix < L:
        val = root_values[boot_ix]
        if turns[boot_ix] != turns[i]:
            val = -val
    else:
        val = 0.0
    for j, r in enumerate(rewards[i:boot_ix]):
        if turns[i + j] == turns[i]:
            val += r
        else:
            val -= r
    return float(val)


def main():
    turns = [0, 1, 0, 1, 0, 1, 0, 1]
    # one terminal reward (+1) received by the mover at the last ply (O):
    rewards = [0, 0, 0, 0, 0, 0, 0, 1]
    root_values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    # obs[i] encodes position i uniquely (no zeros -> trivial legal mask)
    obs = [torch.full((9,), float(i + 1)) for i in range(L)]
    target_policies = [torch.zeros(9) for _ in range(L)]
    actions = list(range(L))

    traj = dict(obs=obs, turns=turns, actions=actions, rewards=rewards,
                target_policies=target_policies, root_values=root_values)

    buf = ReplayBuffer(buffer_size=10, batch_size=4000)
    buf.buffer = [traj]

    obs_b, act_b, trew_b, tval_b, tpol_b, mask_b = buf.sample_batch(K, GAMMA, torch.device("cpu"))

    # map each sampled element back to its position via obs[0] = ix+1
    seen = {}
    for b in range(obs_b.shape[0]):
        ix = int(round(obs_b[b, 0].item())) - 1
        seen.setdefault(ix, set()).add(round(tval_b[b, 0].item(), 4))

    print(f"{'ix':>3} {'turn':>4} {'expected':>9} {'sample_batch':>13}  ok?")
    all_ok = True
    for ix in range(L):
        exp = ground_truth(turns, rewards, root_values, ix)
        got = seen.get(ix)
        if got is None:
            print(f"{ix:>3} {turns[ix]:>4} {exp:>9.1f}   <not sampled>")
            continue
        ok = (len(got) == 1 and abs(next(iter(got)) - exp) < 1e-3)
        all_ok &= ok
        gotv = next(iter(got)) if len(got) == 1 else sorted(got)
        mover = 'X(p1)' if turns[ix] == 0 else 'O(p2)'
        print(f"{ix:>3} {mover:>4} {exp:>9.1f} {str(gotv):>13}  {'OK' if ok else 'WRONG'}")

    print(f"\nbootstrap-branch positions (ix 0,1,2) test the root_value sign-flip;")
    print(f"reward-branch positions (ix 3..7) test the terminal-reward sign.")
    print(f"ALL VALUE TARGETS CORRECT (incl. bootstrap, both movers): {all_ok}")


if __name__ == "__main__":
    main()
