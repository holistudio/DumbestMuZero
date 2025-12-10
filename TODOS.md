# TODOS

## TicTacToe

 - [X] Clone the TicTacToe game from PettingZoo

## Precursors to MuZero

### Minimax

 - [ ] Watch video on minimax: https://www.youtube.com/watch?v=SLgZhpDsrfc 
 - [ ] Implement minimax
 - [ ] Implement alpha-beta pruning

### MCTS / UCT

 - [X] Learn about MCTS and UCT algorithms: http://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf
 - [X] Convert UCT pseudocode into Python script `mcts.py`
 - [X] Add functions to the TicTacToe game for `outcome(state)`, `check_terminal(state)`, `available_actions(state)`, `transition(state, action)`
 - [X] Test UCT Agent against Random Agent
 - [X] Debug why UCT Agent's playing suboptimally after connecting two pieces.
 - [X] (turns out, the UCT value / `best_child()` should not be used to determine the actual next move)
 - [X] Debug why UCT Agent's playing suboptimally as player 2
 - [X] (turns out, +1 and -1 outcomes were assigned to wrong player node in specific cases)
 - [ ] Test different values for hyperparameters `C_p` and `max_iters` and evaluate win-loss-draw ratios against an optimal UCT or random agent.


## MuZero

 - [X] Read the MuZero paper: https://arxiv.org/abs/1911.08265
 - [ ] Watch talk: https://www.youtube.com/watch?v=L0A86LmH7Yw
 - [ ] Write from scratch without looking at pseudocode.
   - [ ] figure out tree structure for exploring actions and tracking function outputs
   - [ ] loop needs to use something akin to tree policy and default policy
   - [ ] use policies from MCTS to produce a policy
   - [ ] use values predicted during MCTS to output a final value estimate
   - [ ] sample available and unxplored actions only 
   - [ ] pre-process observation dictionary into tensor

   - [ ] Store => ReplayBuffer => Loss => Backprop
      - [ ] push action and value to ReplayBuffer
      - [ ] get these to be tuples of K records from tree search + replay buffer?
      - [ ] what params go here? neural nets'?
      - [ ] probably need L2 norm for regularization
      - [ ] store recent returns from environment
      - [ ] check if ReplayBuffer is full
      - [ ] loop through ReplayBuffer
      - [ ] loss function
      - [ ] reward loss term
      - [ ] value loss term
      - [ ] policy loss term
      - [ ] get loss.backward() to work