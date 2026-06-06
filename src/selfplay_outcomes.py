"""
Reconstruct self-play game outcomes (P1 win / P2 win / draw) over training from
the cumulative board-state frequency logs, and plot the trend.

How the reconstruction works
----------------------------
train.py logs, for every ply of every self-play game, the board state the mover
sees BEFORE moving (it then breaks on termination, so the post-deciding-move
board is never logged). The logs are CUMULATIVE: file board_states_eps{a}-{b}
holds counts over episodes 0..b. Differencing consecutive cumulative files gives
the per-window (500-episode) counts.

Let S_k = number of logged board states in a window with exactly k pieces.
S_k is the number of games that made more than k moves (reached pre-move ply k):

    S_0 = total games G   (every game logs the empty board once)
    S_0 = S_1 = ... = S_4  (no game can end before move 5 -> conservation check)

A game of length n (n moves made) contributes to S_0..S_{n-1}. So games of
length exactly n = S_{n-1} - S_n (with S_9 = 0, since the last logged pre-move
board has 8 pieces). The mover of move n is player_1 if n is odd:

    P1 decisive wins = (S_4 - S_5) + (S_6 - S_7)     # games of length 5, 7
    P2 decisive wins = (S_5 - S_6) + (S_7 - S_8)     # games of length 6, 8
    full-board games = S_8                            # length-9: X-win-on-9 OR draw

Length-9 games are split with the "forced move 9" rule: at an 8-piece board there
is exactly one blank and it is player_1's (X) move; fill it with X and check for a
completed line -> X win, else draw.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]


def start_ep(path):
    m = re.search(r"eps(\d+)-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def end_ep(path):
    m = re.search(r"eps(\d+)-(\d+)", os.path.basename(path))
    return int(m.group(2)) if m else -1


def diff_counts(cur, prev):
    """Per-window counts = cumulative cur - cumulative prev."""
    out = {}
    for k, v in cur.items():
        out[k] = v - prev.get(k, 0)
    return out


def x_wins_on_move9(board):
    """board: 9-char string ('0' empty, '1'=X/p1, '2'=O/p2), exactly one blank.
    Fill the blank with X and report whether that completes a line."""
    blank = board.index("0")
    filled = board[:blank] + "1" + board[blank + 1:]
    for a, b, c in LINES:
        if filled[a] == filled[b] == filled[c] == "1":
            return True
    return False


def window_outcomes(counts):
    S = [0] * 10  # S[k] = #states with exactly k pieces
    eight_piece = {}
    for board, n in counts.items():
        k = sum(ch != "0" for ch in board)
        S[k] += n
        if k == 8:
            eight_piece[board] = n

    G = S[0]
    # conservation: no game ends before move 5
    assert S[0] == S[1] == S[2] == S[3] == S[4], f"early-termination conservation failed: {S[:5]}"

    p1_decisive = (S[4] - S[5]) + (S[6] - S[7])
    p2_win = (S[5] - S[6]) + (S[7] - S[8])
    full9 = S[8]

    xwin9 = sum(n for b, n in eight_piece.items() if x_wins_on_move9(b))
    draw = full9 - xwin9
    p1_win = p1_decisive + xwin9

    assert p1_win + p2_win + draw == G, f"outcome conservation failed: {p1_win}+{p2_win}+{draw} != {G}"
    return {"games": G, "p1_win": p1_win, "p2_win": p2_win, "draw": draw}


def main(files, out_png):
    files = sorted(files, key=start_ep)
    cumulatives = []
    for f in files:
        with open(f) as fh:
            cumulatives.append(json.load(fh))

    rows = []
    prev = {}
    for f, cum in zip(files, cumulatives):
        win = diff_counts(cum, prev)
        o = window_outcomes(win)
        o["ep"] = end_ep(f)
        rows.append(o)
        prev = cum

    print(f"{'ep':>7} {'games':>6} {'P1win%':>7} {'P2win%':>7} {'draw%':>7} {'decisive%':>9}")
    for o in rows:
        g = max(o["games"], 1)
        print(f"{o['ep']:>7} {o['games']:>6} "
              f"{o['p1_win']/g*100:>7.1f} {o['p2_win']/g*100:>7.1f} "
              f"{o['draw']/g*100:>7.1f} {(o['p1_win']+o['p2_win'])/g*100:>9.1f}")

    eps = [o["ep"] for o in rows]
    p1 = [o["p1_win"] / max(o["games"], 1) * 100 for o in rows]
    p2 = [o["p2_win"] / max(o["games"], 1) * 100 for o in rows]
    dr = [o["draw"] / max(o["games"], 1) * 100 for o in rows]
    dec = [(o["p1_win"] + o["p2_win"]) / max(o["games"], 1) * 100 for o in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(eps, p1, "-o", label="P1 (X) win %", color="tab:blue")
    plt.plot(eps, p2, "-o", label="P2 (O) win %", color="tab:red")
    plt.plot(eps, dr, "-o", label="draw %", color="tab:green")
    plt.plot(eps, dec, "--", label="decisive %", color="gray", alpha=0.7)
    plt.xlabel("training episode")
    plt.ylabel("% of self-play games (per 500-ep window)")
    plt.title("Self-play outcomes over training (categorical + LR decay + norm + 20k buffer)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    print(f"\nsaved plot -> {out_png}")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "selfplay_trend_current.png"
    files = [a for a in args if a.endswith(".json")]
    for a in args:
        if a.endswith(".png"):
            out = a
    main(files, out)
