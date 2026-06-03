# DumbestMuZero

A minimal (and therefore the dumbest) implementation of [MuZero](https://arxiv.org/abs/1911.08265)

## Goal

**What I wanted to do:** I wanted to implement MuZero from scratch and understand how it works, especially as a newcomer to model-based RL and Monte Carlo Tree Search methods.

**Why Tic Tac Toe?:** I applied to a programming retreat the Recurse Center (RC) and got admitted. The application involved coding a game of tic-tac-toe from scratch. After finding a great study group at RC focused on RL and games, naturally my mind kept going back to the game of tic-tac-toe as a way to ignore complexities in game environments (rules/rewards) and focus on thinking through how any RL algorithm works.

**Why MuZero?:** I've always remembered hearing about it when I first learned about RL as "the one that learns without being told what the rules are" (relative to AlphaGo). And as I learned and experimented more with other RL methods during my RC retreat, I converged on this idea that I should get a working implementation of MuZero to at least work on the simple game of tic-tac-toe...

...cause how hard could it be?

## What Works Now

Monte Carlo Tree Search (MCTS) agent self-play appears to work well.

```bash
cd src
python mcts_test.py
```

Watch the game that's being played in the PyGame window, and note the terminal displaying number of visits for each action during tree search and its estimated UCT value. The agents will always pick the action with the highest numbest of visits during search.

A MuZero implementation and training loop runs with no errors *at least...*

```bash
cd src
python train.py
```

## Current Work in Progress

...but the issue is the performance of this MuZero, particularly against a random agent.

After 5000 training episodes, this MuZero plays as Player 1 against a random agent, it loses or draws around 30-40% of the time.

And when this MuZero plays as Player 2 against a random agent, it loses or draws around 60-70% of the time.

## What I've Learned So Far

- Debugging RL can be a major pain, so thank goodness I am just looking at the game of tic-tac-toe where a single game has very short traces/trajectories and I can see value-estimates at every turn. This is what ultimately helped me getting MCTS for tic-tac-toe to work.

- At present though, the MCTS/UCT agent seems to require at least 10,000 simulations of tic-tac-toe games at each turn to reliably identify the best next move (`max_iters=10_000`).

- For MuZero, a key component is [adding Dirichlet noise](./DEVLOG.md#2025-12-26) to the root node to effectively encourage exploration.

## Key Things to Do

- A more systematic testing of MCTS with different hyperparamters (`C_p`, `max_iters`) could be worth doing.

- Training MuZero with the "always optimal MCTS" could be worth trying out, but ONLY to assess if there are errors in other parts of the implementation. That said, the "true test" is to only allow MuZero self-play during training.

- The current implementation of MuZero has a simple ReplayBuffer to update the different functions/neural nets. The setup needs to be re-evaluated.

- More generally, somehow someway get this MuZero to get near 0% lose rate against a random agent, regardless of whether MuZero plays as player 1 or player 2. 

## Setup

After activating a virtual environment (`conda` or `uv`):

Install PyTorch, with CUDA if available: https://pytorch.org/get-started/locally/

Then install other requirements

```bash
(uv) pip install -r requirements.txt
```