# noqa: D212, D415
"""
# Tic Tac Toe

```{figure} classic_tictactoe.gif
:width: 140px
:name: tictactoe
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import tictactoe_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= ['player_1', 'player_2']`            |
| Agents             | 2                                             |
| Action Shape       | (1)                                           |
| Action Values      | [0, 8]                                        |
| Observation Shape  | (3, 3, 2)                                     |
| Observation Values | [0,1]                                         |


Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation is 2 planes of the 3x3 board. For player_1, the first plane represents the placement of Xs, and the second plane shows the placement of Os. The possible values for each cell are 0 or 1; in the first plane, 1 indicates that an X has been placed in that cell, and 0 indicates
that X is not in that cell. Similarly, in the second plane, 1 indicates that an O has been placed in that cell, while 0 indicates that an O has not been placed. For player_2, the observation is the same, but Xs and Os swap positions, so Os are encoded in plane 1 and Xs in plane 2. This allows for
self-play.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:


 ```
0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8
 ```

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.

### Version History

* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""
from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from envs.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board
from pettingzoo.utils import AgentSelector, wrappers


def get_image(path):
    """Return a pygame image loaded from the given path."""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image


def get_font(path, size):
    """Return a pygame font loaded from the given path."""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    font = pygame.font.Font((cwd + "/" + path), size)
    return font


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "tictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.board = Board()

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        observation = np.empty((3, 3, 2), dtype=np.int8)
        # this will give a copy of the board that is 1 for player 1's
        # marks and zero for every other square, whether empty or not.
        observation[:, :, 0] = np.equal(board_vals, cur_player + 1)
        observation[:, :, 1] = np.equal(board_vals, opp_player + 1)

        action_mask = self._get_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        action_mask = np.zeros(9, dtype=np.int8)

        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        if agent == self.agent_selection:
            for i in self.board.legal_moves():
                action_mask[i] = 1

        return action_mask

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), action)

        status = self.board.game_status()
        if status != TTT_GAME_NOT_OVER:
            if status == TTT_TIE:
                pass
            else:
                winner = status  # either TTT_PLAYER1_WIN or TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] += 1
                self.rewards[self.agents[loser]] -= 1

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def outcome(self, observation):
        """Return +1 if player 1 has won, -1 if player 2 has won, or 0 otherwise."""
        # The observation for player_1 has player_1 marks in plane 0.
        # The observation for player_2 has player_2 marks in plane 0.
        # observation["observation"][:, :, 0] is the current player's marks.
        # We need a consistent view of the board to check for absolute winner.

        current_player_plane = observation["observation"][:, :, 0]
        opponent_plane = observation["observation"][:, :, 1]
        total_pieces = np.sum(current_player_plane) + np.sum(opponent_plane)

        outcome_sign = 1
        # If total pieces is even, it's player 1's turn (current player is p1)
        # If total pieces is odd, it's player 2's turn (current player is p2)
        if total_pieces % 2 == 0:
            p1_plane, p2_plane = current_player_plane, opponent_plane
            if self.agent_selection == 'player_1':
                outcome_sign = -1
        else:
            p2_plane, p1_plane = current_player_plane, opponent_plane
            if self.agent_selection == 'player_2':
                outcome_sign = -1

        # p1_plane, p2_plane = current_player_plane, opponent_plane

        # winning_combinations are tuples of flat indices
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6),             # diags
        ]

        for combo in winning_combinations:
            if all(p1_plane.flat[i] == 1 for i in combo):
                if self.agent_selection == 'player_1':
                    return 1*outcome_sign
                else:
                    return -1*outcome_sign
            if all(p2_plane.flat[i] == 1 for i in combo):
                if self.agent_selection == 'player_2':
                    return 1*outcome_sign 
                else:
                    return -1*outcome_sign

        return 0  # Draw or game not over
    
    def check_terminal(self, observation):
        """Return True if the game is over, False otherwise."""
        # Game is over if a player has won
        if self.outcome(observation) != 0:
            return True

        # Game is also over if it's a draw (board is full)
        p1_plane = observation["observation"][:, :, 0]
        p2_plane = observation["observation"][:, :, 1]
        if np.sum(p1_plane) + np.sum(p2_plane) == 9:
            return True
        return False

    def available_actions(self, observation):
        """Return a list of legal moves to place a piece on the board."""
        # The action_mask is a binary vector indicating legal moves.
        return np.where(observation["action_mask"] == 1)[0].tolist()
    
    def transition(self, observation, action) -> dict:
        """
        Given an observation and an action, return the next observation.
        Assumes the action is legal. The returned observation is from the perspective
        of the next player.
        """
        # The observation is from the current player's perspective.
        # Plane 0 is the current player's pieces, Plane 1 is the opponent's.
        current_player_plane = observation["observation"][:, :, 0].copy()
        opponent_plane = observation["observation"][:, :, 1].copy()

        # Apply the current player's action to their plane.
        # The action is a flat index for the 3x3 board.
        np.put(current_player_plane, action, 1)

        # The next observation is from the opponent's perspective.
        # Their pieces are now on plane 0, and the (now previous) player's pieces are on plane 1.
        next_obs_board = np.stack([opponent_plane, current_player_plane], axis=-1)

        # Calculate the action mask for the new state.
        occupied_mask = (next_obs_board[:, :, 0] + next_obs_board[:, :, 1]).flatten()
        next_action_mask = 1 - occupied_mask
        return {"observation": next_obs_board, "action_mask": next_action_mask.astype(np.int8)}

    def reset(self, seed=None, options=None):
        self.board.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_height, self.screen_height)
            )
            pygame.display.set_caption("Tic-Tac-Toe")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_height, self.screen_height))

    def close(self):
        pass

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_height = self.screen_height
        screen_width = self.screen_height

        # Setup dimensions for 'x' and 'o' marks
        tile_size = int(screen_height / 4)

        # Load and blit the board image for the game
        board_img = get_image(os.path.join("img", "board.png"))
        board_img = pygame.transform.scale(
            board_img, (int(screen_width), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Load and blit actions for the game
        def getSymbol(input):
            if input == 0:
                return None
            elif input == 1:
                return "cross"
            else:
                return "circle"

        board_state = list(map(getSymbol, self.board.squares))

        mark_pos = 0
        for x in range(3):
            for y in range(3):
                mark = board_state[mark_pos]
                mark_pos += 1

                if mark is None:
                    continue

                mark_img = get_image(os.path.join("img", mark + ".png"))
                mark_img = pygame.transform.scale(mark_img, (tile_size, tile_size))

                self.screen.blit(
                    mark_img,
                    (
                        (screen_width / 3.1) * x + (screen_width / 17),
                        (screen_width / 3.145) * y + (screen_height / 19),
                    ),
                )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
