# DEV LOG

## 2025-12-17

One of the struggles with implementing this based on the Appendix B text is that the writing does not describe auxiliary data structures that come in handy to help keep track of things during the tree search. A `TreeNode` and `TreeEdge` class storing states and other statistics alone isn't going to cut it.

A few data structures visible in the pseudocode:
- `search_path` a list that just tracks the nodes traversed during Simulation/Selection phase, which comes in handy during expansion phase
- `action_history` a list of actions explored during the Simulation/Selection phase, which comes in handy when determining whose turn it during the tree search and looking up the last action during the selection phase.
- Though Appendix B states that tree nodes store state and tree edges store statistics, a `Node` class is really all you need, provided that the `children` attribute is a dictionary with actions as keys and child `Nodes` as values.

In general, I'm referring to the pseudocode provided by the MuZero paper authors now and feel more confident in my latest draft of the "forward pass." I'm planning to review each function and see if anything needs corrections first before moving onto adding `ReplayBuffer` and figuring out how training/loss function/backprop actually works.

The somewhat unintuitive thing I got from the pseudocode is that `Nodes` are intiated only with a policy prior score at first and then later get hidden state and reward defined during expansion after representation/prediction/dynamics functions do their thing. I *guess* these things can happen in either order...I'll revisit this tomorrow during the review.

## 2025-12-10

Took a stab at programming the forward pass (i.e., the "planning" part A in Figure 1 of the MuZero paper)

Then I hit a wall in coding when I had to figure out how to actually select an action ("acting" part B in Figure 1). 

Couple points of confusion:
- Even during planning it was unclear where "candidate actions" come from. I reasonably assumed that they could at least be chosen based on "unexplored actions" for unexpanded tree nodes, but after a node has all possible actions explored, how are candidate actions selected?
- I thought the answer to my above question was to select children with the highest value estimated by the prediction function, but this turns out to be completely wrong.
- I got a sense that the above was wrong when I read Appendix B of the MuZero paper. Appendix B covers the actual Search algorithm of MuZero (*Note to self: Appendices of ML research papers contain at least 50% of their actual useful information*)
- Reading the above Appendix still seems daunting, but I'm seeing now-familiar terms like upper-confidence bound and backup from my MCTS reading.

Basically I'm starting to realize how my original diagram doesn't really capture the MCTS nature of the MuZero algorithm.
- During planning, MCTS follows Appendix B Search to select candidate actions and create a tree. Visit counts to each node/edge are stored, since this is the key part for...
- Acting, or more accurately, action sampling. A greedy method would just be to pick the node with the highest visit count, but there's a fancy formula involving temperature and probabilities based on the visit count to ensure some amount of exploration when the agent actually decides the next move.

I'm pushing the changes I have for now, but I think I need to re-visit the diagram I made and in particular see if I can somehow incorporate the Appendix B search steps/formulas. This should give me a better mental model before doing more coding. Appendix B also seems to have clearer language on the data structures and storing of the statistics to enable MCTS.

## 2025-12-08

This is the "official" hello world post for this project! I did start working this before today, but that was to familiarize myself first with Monte Carlo Tree Search (MCTS) and specifically Upper Confidence Tree (UCT) Search. After I read the MuZero paper, this seemed like a critical concept that I didn't know much about relative to other RL and deep RL concepts I have done in the past (Q-learning, DQN, PPO, MLP Actor-Critic). So last week I read about [MCTS](http://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf) and then got [it to work](https://github.com/holistudio/DumbestMuZero/blob/main/src/agents/mcts/mcts.py) for the game of tic-tac-toe!

For implementing MuZero, I made the following diagram as I read the [MuZero paper](https://arxiv.org/abs/1911.08265):

<img src="img/251208_forward_pass.png">

The goal for now is to implement the above diagram, before I (inevitably) struggle and refer to the paper's Python pseudo-code `D:`

Here goes nothing!
