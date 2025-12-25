# DEV LOG

## 2025-12-24

Calling it a night now that I'm confident the code is right and I can leave it to train over night.

Some notes on interesting things that came up as I reviewed and revised the code (a lot):

Ah now I understand `td_steps` being set to `max_moves` for a board game!

In the pseudocode's `make_target()` function:

```python
      bootstrap_index = current_index + td_steps
      ...
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0
```

Since `td_steps` is set to `9` for tic-tac-toe, `bootstrap_index` is guaranteed to go beyond the end of the game, and value is intialized to 0. THEN the final reward is accounted for at the end of the game:

```python
      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        if self.player_history[current_index + i] == current_player:
          value += reward * self.discount**i  # pytype: disable=unsupported-operands
        else:
          value -= reward * self.discount**i
```

And the signs are flipped appropriately so a player 1 win of +1 gets interpreted as a player 2 loss automatically. This means that the final outcome of the game is effectively accounted for regardless of where the sample trajectory starts (i.e., regardless of the random value of `current_index`)

(DeepMind did not provide this exact code accounting for whether the player's turn at index `i` is the same as `current_player` at `current_index`, I had Gemini look into this and add more if statements and player history tracking.)

Also absorbing states past the end of the game means a uniform policy is added as a target because it doesn't matter (may want to revisit this further but makes sense). DeepMind did it this way:

```python
# States past the end of games are treated as absorbing states.
targets.append((0, last_reward, []))
```

I ended up doing it this way:

```python
uniform_policy = torch.ones_like(target_policies[0]) / target_policies[0].numel()
targets.append((uniform_policy, last_reward, 0))
```

One of my earlier versions of `sample_batch()` makes targets for the neural nets predicting value and reward assuming the "current player" is always the player whose turn it is at the beginning of the trajectory slice (player turn at `sample_trajectory[0]`). Subsequent determinations of the sign of the value/reward (+1/-1) would compare whose turn it is in a future step relative to that "current player." So I set `current_player` outside of the `for i in range(ix, ix+k_unroll_steps+1):` loop

But reviewing this with Gemini, the correct thing is to set "current player" to whose turn it in each future unroll step, inside the `for i in range(ix, ix+k_unroll_steps+1):` loop. Because you actually need to compare the player's turn at future bootstrap index and at timesteps previous to the bootstrap to determine the correct sign.

Gemini explains:

The reason the perspective is based on current_index (the specific step in the unrolled future) rather than state_index (the root of the search) comes down to what the neural network is being trained to predict at each specific step.

Here is the reasoning in terms of the Network and MCTS:

1. The Network's Job is Local to the State
In MuZero, the training process "unrolls" the network forward in time.

At state_index (step 0), the network sees the root state $S_0$ and predicts Value $V(S_0)$.
At current_index (step $k$), the network sees a hidden state representing $S_k$ and predicts Value $V(S_k)$.
In a zero-sum game (like Chess or Go), the standard definition of a state's "Value" is relative to the player whose turn it is in that state.

If it is Player A's turn at $S_k$, a Value of $+1$ means Player A is winning.
If it is Player B's turn at $S_k$, a Value of $+1$ means Player B is winning (which implies Player A is losing).
The neural network is designed to be a function $f(s) \rightarrow v$. It doesn't know about the history of who started the search $k$ steps ago; it only knows "I am in this state, it is my turn, how good is this position for me?"

2. Consistency with MCTS (Negamax)
The MCTS algorithm typically uses a "Negamax" backup strategy for zero-sum games.

When the search tree reaches a leaf node (corresponding to current_index), the network returns a value $v$.
The MCTS algorithm assumes this $v$ is good for the player at that leaf node.
When backing up to the parent node (where it was the opponent's turn), MCTS flips the sign ($-v$).
If you trained the network using the perspective of state_index (the root player), then when the unroll reaches a state where it is the opponent's turn, the network would be trained to predict the root player's value (e.g., "I am Player B, but the value is -1 because Player A is winning"). This would break the Negamax assumption in MCTS, which expects the value to be positive if the current player (Player B) is winning.


## 2025-12-23

Slowly making my way through the TODOs I wrote yesterday. Going to call it a night but one other major thought bubble is I will probably resort to some vibe-coding JUST to help debug:

- Show in terminal output what the replay buffer looks like after a game is complete, just to double check the values
- Show what a single simulation tree looks like to make sure the values are the right sign.

I'll add more terminal output debugging ideas later but that's it for now...

## 2025-12-22

OK now going to just fully read the DeepMind's pseudo-code and take a look at each part side-by-side:

- DM's `Game` class tracks `child_visits` and `root_values`, which are:
  - `child_visits` the *normalized* child visits serving as target policy during training
  - `root_values` which are then used to derive a target value from discounted rewards/values
  - A `td_steps` parameter is used to look ahead a limited number of steps when computing the discounted value, necessary for very long games.
  - Another small-but-big detail is that `Game.make_target()` makes sure to include tuple for the targets for the `Network.initial_inference()`, which doesn't predict a reward value, but the target reward is provided as `last_reward = 0`.
- DM's `ReplayBuffer.window_size` specifies the number of game trajectories in the buffer.
  - Whem a game trajectory is stored with `ReplayBuffer.save_game()` the older games are popped off once the length of the buffer exceeds `window_size`.
  - `window_size` doesn't specify the number of total steps (state action transitions) an agent has made/experienced. It specifies the total number of game episodes/entire trajectories the agent has experienced.
- DM uses a `Network` class to do both `initial_inference()` and `recurrent_inference` to elegantly combine the outputs of the three neural nets, etc.
  - Then `SharedStorage` keeps track of the latest training network weights for running parallel self-play jobs
- `select_action()` makes use of softmax temperature when sampling the next action to play in a game.
  - similarly `add_exploration_noise` "adds dirichlet noise to the prior of the root to encourage the search to explore new actions."
- In `ucb_score()`:
  - This line looks different from what I did: `value_score = child.reward + config.discount * min_max_stats.normalize(child.value())`
  - In equivalent variable names, my `value_score = min_max_stats.normalize(child.value())`...
  - the `value_score = 0` if the child node in question doesn't have any visit counts. I've done the same thing basically by reporting a node's `mean_value` is 0 if it has `N=0` visit counts.
- `backpropagate()` looks way nicer than my `backup()` function, especially with fewer if statements.
  - it seems like the backpropagate starts with the leaf node.
- my `search()` function has to be compared to DM's `play_game()` and `run_mcts()` functions:
  - `play_game()` creates the root node and sets its state with the `intial_inference()` using representation and dynamics function
  - `run_mcts()` runs the tree search simulations
- `train_network()` initializes the networks and Momentum optimizer and includes calls to function that save the network weights over multiple training steps.

Things I want to review and revise carefully:
- [x] My `ReplayBuffer` stores the `final_outcome` and `reward` two separate lists.
  - [x] For a board game a single list is all you need and should look something like `[0, 0, 0,...,0,+1,-1]`
  - ~~This does leave me feeling a little confused about how self-play happens - how do you record the player_2's loss after the player_1 wins?~~ I'm guessing that so long as I track which player won and then alternate the reward/value signs via NegaMax when sampling targets and tree search backups, this should be OK.
  - [x] Looks like DM's pseudocode doesn't show how the target value needs to flip in sign for the opposing player (i.e., `Game.make_target()` is a little incomplete, it does have `to_play` as an input parameter, but doesn't make use of it. this function should check if the player at `state_index` is different from the player at `bootstrap_index` and flip the sign).
- [x] My current code doesn't yet discount the rewards and values when computing target values. Even though I am only interested in getting things working for tic-tac-toe with a max of 9 steps per game, this seems like a useful thing to compute should I ever extend this to other games.
  - [x]Also look into the value of `td_steps` for board games being set to `max_moves`...So for tic-tac-toe this value is 9...and that means the agent also looks 9 steps ahead, regardless of the state of the board?
- ~~I *think* that the reason my current `predictions` and `targets` lists are different lengths is because I do NOT include the target for the initial inference...~~ after revising `sample_batch()` the predictions and targets are the same length
- [x] Ensure the buffer length remains under a specified `buffer_size`
- Softmax temperature sampling and Dirichlet noise seem like overkill when I just want to get MuZero to play tic-tac-toe BUT could still be necessary ways to encourage exploration during training, generating more diverse data that we now really need since we are using three friggin' neural nets...
- [x] Double check my `pUCT()` function and see how to implement `value_score = child.reward + config.discount * min_max_stats.normalize(child.value())` using my own class definitions. 
- [x] More importantly ask why this is done, since it doesn't appear to match the Equation 2...
  - Welp, looks like I forgot the basics of the Bellman Equation and only used the value of the child as the "Q-value." I should remember that a Q-value is almost always an immediate reward and discounted value sum ($Q = R + \gamma V$)
- [x] I'll re-write my `backup()` using `reversed(search_path)` and see if it makes sense to not care if the current node is a leaf node or not.
- [x] In conjunction with the above, the `search()` function may need to be revised in a couple ways:
  - [x] After the root node is created with a hidden state, there shouldn't be a `backup()`...
  - [x] the `backup()` should account for the latest leaf node's `current_player` (found via the `whose_turn(action_history)`) and flip the signs of the value accordingly
- [x] add training loop to `update()` function so that the networks train over multiple epochs / multiple batches.
- consider switching to the MomentumOptimizer...
- [x] I sorta think I have to now set up a self-play muzero training loop, and write a separate evaluation function that pits MuZeroAgent against random agent as player 1 or player 2 and see the win-loss-draw proportions.



## 2025-12-21

Tried to make a push to finish this thing and...
- well it can run through games and update weights after computing a loss function but...
- I think there's an issue with the way I'm computing target value (without a discount factor)
- There may be some other issues with how I'm pairing the targets with the neural network predictions...
- And I still haven't gotten around to the issue of self play
- and storing neural net weights...
- and separate training and test loops...

So all in all, I still have a ways to go before I can start properly tuning the various hyperparameters.

Small update: twiddled a bit with hyperparameters (`max_iters`) and saw some encouraging training results at the end of 100 episodes:

```
0:07:51.005167 EP=99, [64, 29, 7] # W-L-D record as player 1 against random agent
```

Of course, I also continued running the trained model in a visible game:

<img src="./img/251221_initial_muzero_test.png" width="400 px">

The agent is Player 1 placing the X pieces. After Player O's random move in board location index=5, the agent predicted the following action probabilities:

```
Action Probability 0 = 0.00%
Action Probability 1 = 0.00%
Action Probability 2 = 31.83%
Action Probability 3 = 0.00%
Action Probability 4 = 23.64%
Action Probability 5 = 0.00%
Action Probability 6 = 15.59%
Action Probability 7 = 15.24%
Action Probability 8 = 13.64%
```

...and went to board index=2! Why isn't the probability for board location 6 wayyy higher?!?!?

Lots more debugging ahead.

#@ 2025-12-20

OK, time to make this more real with an actual `MuZeroAgent` class. I've been shying away from doing this until now because I really wanted to focus on the individual functions rather than worry about which variables have to be attributes of which class. But now I do want to start thinking about this more.

I'm still working out how the training / weight updates actually happen since I've come to realize I have not really don't three neural networks being trained as a single "model" with weight updates coming from multiple loss functionals summed into one. Will clarify this tomorrow and hopefull get something finally running!

## 2025-12-19

I did a quick draft of commented pseudo-code describing the ReplayBuffer and training/update neural net weights

The main thing that took me a while to understand is the `target_policy` (i.e., number of action node visits during MCTS simulation, normalized so it's like action probabilities), which is compared with the prediction function's `policy` output (i.e., logit scores for each action, ~~which can be converted into probabilities~~), via cross entropy loss.

Now I'll try to turn pseudo-code into some actual code.

## 2025-12-18

Going to review the code written so far with Gemini. My prompt template is something like this:

```
{
  What:
    - "I am writing my own implementation of MuZero from scratch."
    - "The implementation is NOT complete. I've done a basic draft of the MuZero Search algorithm"
    - "Please review the specific function I have highlighted and check for any major inconsistencies with how MuZero works for that specific function"
    - "Suggest any code changes I should make for the function and explain."
    
  Boundaries:
    - "Only consider the function that I have highlighted and point to."
    - "Only modify the lines within the function. DO NOT modify lines outside function"
    - "You may explain verbally (NOT CODE) how a modification within the function relates to other functions outside the highlighted scope."
    - "Your response should focus on one specific part/phase of MuZero search without explaining the rest of the algorithm in great detail."
    - "Do not add ReplayBuffer or other data structures that are required for loss function computation and neural network training."
    - "Only focus on the 'forward pass' or MuZero MCTS for deciding the next move given the latest/current game state."
    - "Do not modify the PyTorch neural network Functions defined above. I do not care about improving the neural nets."
    - "Do not worry about references to the environment `env`. I will take care of how the environment performs specific functions in a different class definition."
    
  Success:
    - "I am less interested in implementing a full MuZero algorithm for complicated games, and more interested in learning how to write a very basic form of MuZero for a simple game like tic-tac-toe."
    - "I am learning how to implement MuZero, so I want a detailed understanding at a function/specific phase level of how the algorithm works and how the code should reflect that."
}
```

Use the above prompt for reviewing my `expansion()` function I'm already learning that the expansion should only expand the legal moves.
- Initially I thought it's not possible to identify legal moves during MuZero search because it's using a hidden state representation...
- But that alone doesn't prevent one from determining legal moves.
- At the very start of MCTS we are given the explicit game board and the game tells us the legal moves (`gymnasium` does so with `observation[action_mask]`)
- But that's not all! MuZero search **still** predicts "explicit" actions to take at each node of the tree.
- So during simulation / `selection()` we track these action in `ACTION_HISTORY`
- Using both `ACTION_HISTORY` and combined with the original game board, we can also infer subsequent legal moves.
- To be clear, this does mean we do still track explicit representations of the future board states BUT those explicit representations are never given to the model.


Also, it appears I mis-read Equation 3 in Appendix B describing the bootstrap backup update of node values:

As written in the paper:

$$
G^k = \sum_{\tau=0}^{l-1-k}{\gamma^\tau r_{k+1+\tau}} + \gamma^{l-k}{v^l}
$$

A less confusing way of writing it:

$$
G^k = \gamma^{l-k}{v^l} + \sum_{\tau=0}^{l-1-k}{\gamma r_{k+1+\tau}}  
$$

To the RL PhDs, this is trivial because they probably know the discounted reward formulas from Sutton and Barto off the top of their heads, but for a dumb person like me, I thought $\gamma^{l-k}{v^l}$ was *inside* the summation term...as a constant...accruing over and over each node...

Yes, in hindsight this is really dumb, but hey lesson learned! My nicer equation version clearly shows that you start by initializing $G$ with the value $v$ computed by the prediction function $f_\theta$ before going into the backup for-loop.

More importantly, I wrote an inner for loop to represent this summation performed at node $k$ but actually, there's no need for an inner for loop. As you traverse the nodes in reverse order, you can use the previously computed discounted reward $G^{k+1}$ in your current computation of $G^{k}$. i.e.:

```
G = reward + GAMMA * G
```

In the `search()` I made a major mistake in passing the wrong state to the dynamics function:

```
last_node, search_path, action_history = selection(root_node)
state, reward = DynamicsFunction(last_node.state, action_history[-1])
```

The end of the search returns `last_node` or the leaf node, which is unexpanded and does not have a hidden state defined yet! So `last_node.state` is `None`. From an RL-intuition about MuZero, it also doesn't make sense because the whole point of having dynamics function IS to predict THIS leaf node's hidden state.

So the correct thing to do is to look up the parent node's state and give it to dynamics function. This is easy with the `search_path` list keeping track of that already. 

```
last_node, search_path, action_history = selection(root_node)
parent_node = search_path[-2]
state, reward = DynamicsFunction(parent_node.state, action_history[-1])
```

Finally, with the way I have things written, the first time I make the root node and expand also requires me to do a backup phase because
- prediction function has made a value estimate based on the initial state
- `expansion()` does not "initialize" the node's `value_sum` or visit counts `N`
- so a quick `backup()` is needed before the main for loop in `search()`

Moreover I forgot to alternate the sign of the discounted reward/value `G` based on whether the node corresponds to current player or not! (same issue with regular UCT)

Now I can move on to writing the training functions, `step()` and `update()`, `ReplayBuffer` class...

Still a ways to go!


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
