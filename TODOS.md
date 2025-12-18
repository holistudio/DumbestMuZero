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
 - [x] Write MuZero's MCTS / "forward pass" from scratch ~~without looking at pseudocode~~ peeking at pseudocode.
   - [x] figure out tree structure for exploring actions and tracking function outputs
   - [x] loop needs to use something akin to tree policy and default policy
   - [x] use policy logits from MCTS to produce a policy
   - [x] ~~use values predicted during MCTS to output a final value estimate~~ (value used during training, not search)
   - [x] ~~sample available and unexplored actions only~~ all possible actions in action space are added during the expansion phase of MuZero search
   - [x] but what happens when there are no more unexplored actions
   - [x] pre-process observation dictionary into tensor
- [ ] Review forward pass functions with Gemini step by step

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
   - [ ] weight decay/regularization term
   - [ ] get loss.backward() to work

- [ ] Fit into `gymnasium` agent-game loop pattern
   - [ ] `agent.step(obs)`
   - [ ] `agent.update()`
- [ ] for tic tac toe the search depth should be tracked to within 9 total steps

- [ ] Get a full code review

- [ ] `select_action()` just chooses next action based on node with highest number of visits. this isn't "sampling from probability distribution". consider adding softmax/temperature computations

