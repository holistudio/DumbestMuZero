from envs.tictactoe import tictactoe
import sys

EVAL_EPS = 500
SEED = 42


"""TERMINAL DISPLAY HELPERS"""
USE_COLOR = sys.stdout.isatty()

class C:
    RESET = '\033[0m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    GRAY = '\033[90m'
    BOLD_CYAN = '\033[1;36m'

def _c(text, code):
    return f'{code}{text}{C.RESET}' if USE_COLOR else text

def _fmt_wld(w_perc, l_perc, w_l_d):
    """format a win/loss/draw percentage triple with a raw record, colorized"""
    d_perc = 100 - w_perc - l_perc
    w = _c(f'{w_perc:5.1f}%W', C.GREEN)
    l = _c(f'{l_perc:5.1f}%L', C.RED)
    d = _c(f'{d_perc:5.1f}%D', C.GRAY)
    record = _c(f'({w_l_d[0]}-{w_l_d[1]}-{w_l_d[2]})', C.DIM)
    return f'{w} {l} {d} {record}'

def play_random_games(num_eps, report_for):
    """
    play two random agents against each other for num_eps games,
    reporting win/loss/draw counts for the given player ('player_1' or 'player_2')
    """

    w_l_d = [0, 0, 0]

    env = tictactoe.env()
    agents = {
        'player_1': 'random',
        'player_2': 'random'
    }

    for ep in range(num_eps):
        env.reset(seed=SEED + ep)
        for a in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                action = int(env.action_space(a).sample(mask))

            env.step(action)

            # when the game is terminated
            # and two players are still present in environment's
            # logging dictionary
            if len(env.terminations.keys()) == 2:
                if env.terminations[a] == True:
                    if env.rewards[report_for] == 1:
                        w_l_d[0] += 1
                    elif env.rewards[report_for] == -1:
                        w_l_d[1] += 1
                    else:
                        w_l_d[2] += 1
    env.close()

    return w_l_d


if __name__ == '__main__':
    p1_w_l_d = play_random_games(EVAL_EPS, 'player_1')
    p2_w_l_d = play_random_games(EVAL_EPS, 'player_2')

    # calculate win percentages
    p1_w_perc = p1_w_l_d[0] * 100 / sum(p1_w_l_d)
    p2_w_perc = p2_w_l_d[0] * 100 / sum(p2_w_l_d)
    p1_l_perc = p1_w_l_d[1] * 100 / sum(p1_w_l_d)
    p2_l_perc = p2_w_l_d[1] * 100 / sum(p2_w_l_d)

    # display performance in terminal
    print(f'P1 {_fmt_wld(p1_w_perc, p1_l_perc, p1_w_l_d)}  │ P2 {_fmt_wld(p2_w_perc, p2_l_perc, p2_w_l_d)}')
