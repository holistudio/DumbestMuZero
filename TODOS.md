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
 - [ ] Watch paper review: 
 - [x] Write MuZero's MCTS / "forward pass" from scratch ~~without looking at pseudocode~~ peeking at pseudocode.
   - [x] figure out tree structure for exploring actions and tracking function outputs
   - [x] loop needs to use something akin to tree policy and default policy
   - [x] use policy logits from MCTS to produce a policy
   - [x] ~~use values predicted during MCTS to output a final value estimate~~ (value used during training, not search)
   - [x] ~~sample available and unexplored actions only~~ all possible actions in action space are added during the expansion phase of MuZero search
   - [x] but what happens when there are no more unexplored actions
   - [x] pre-process observation dictionary into tensor
- [x] Review forward pass functions with Gemini step by step

- [ ] Store => ReplayBuffer => Loss => Backprop
   - [ ] ~~get these to be tuples of K records from tree search + replay buffer?~~
   - [x] push steps to ReplayBuffer via `agent.experience()`
   - [x] check if ReplayBuffer is full
   - [x] loop through ReplayBuffer
   - [x] ReplayBuffer sample_batch()
   - [ ] Final outcome z maybe with a discount factor???
   - [ ] Final outcome +/- based on current player and node player_turn
   - [ ] ~~Three neural nets as one model~~
   - [x] loss function
   - [x] reward loss term
   - [x] value loss term
   - [x] policy loss term
   - [x] weight decay/regularization term
   - [x] add gradient scale 
   - [x] what params go here? neural nets'?
   - [ ] ~~probably need L2 norm for regularization~~ handled by AdamW
   - [x] get loss.backward() to work
   - [x] re-evaluate where to put `torch.no_grad()` and `model.train` vs `eval()`
   - [ ] evaluate where to specify cuda
   - [ ] ReplayBuffer needs to discard the old episodes with newer experiences

- [ ] Fit into `gymnasium` agent-game loop pattern
   - [x] `flatten()` return observation space as a 1-D vector
   - [x] `agent.step(obs)`
   - [x] `agent.update()`
   - [x] `agent.experience()` store recent returns from environment

- [ ] ~~for tic tac toe the search depth should be tracked to within 9 total steps~~ not needed, when legal actions are still possible to deduce with initial observation and action history

- [ ] Get a full code review

- [ ] `select_action()` just chooses next action based on node with highest number of visits. this isn't "sampling from probability distribution". consider adding softmax/temperature computations

