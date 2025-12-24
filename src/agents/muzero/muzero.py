import math
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class StateFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)
        pass

    def forward(self, obs):
        x = self.lin1(obs)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        x = self.lin3(x)
        return x

class DynamicsFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.state_head = nn.Linear(hidden_size, output_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        pass

    def forward(self, s_prev, a):
        x = torch.cat((s_prev, a))
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        s = self.state_head(x)
        r = self.reward_head(x)
        return s, r
    
class PredictionFunction(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)
        pass

    def forward(self, s):
        x = self.lin1(s)
        x = F.gelu(x)
        x = self.lin2(x)
        x = F.gelu(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

class Node(object):
    def __init__(self, prior):
        self.state = None
        self.N, self.value_sum = 0, 0
        self.P = prior
        self.R = 0
        self.current_player = -1

        self.children = {}
        pass

    def expanded(self):
        return len(self.children.items()) > 0
    
    def mean_value(self):
        if self.N > 0:
            return  self.value_sum / self.N
        else:
            return 0

"""REPLAY BUFFER"""

class ReplayBuffer(object):
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.buffer = [] # store each game's trajectory

        # TRAJECTORY = zip(observations, player_turns, actions, immediate_rewards, target_policies, final_outcomes)
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.rewards = []
        self.root_values = []
        pass
    
    def reset_trajectory(self):
        self.observations = []
        self.player_turns = []
        self.actions = []
        self.target_policies = []
        self.rewards = []
        self.root_values = []
        pass

    def store_step(self, obs, player_turn, action, target_action_probs, reward, root_value):
        # observation vector
        self.observations.append(obs)

        # player turn
        self.player_turns.append(player_turn)

        # action direct from what was chosen by select_action (different from child visits if sampled)
        self.actions.append(action)

        self.target_policies.append(target_action_probs)

        # get the reward from environment
        self.rewards.append(reward)

        self.root_values.append(root_value)
        pass

    def store_trajectory(self):
        # assume the game is over and outcome is recorded as the last reward
        final_outcome = self.rewards[-1]
        other_player_outcome = -final_outcome
        self.rewards[-2] = other_player_outcome

        print(self.player_turns)
        print(self.actions)
        print(self.rewards)

        trajectory = dict(obs=self.observations, 
                         turns=self.player_turns, actions=self.actions, 
                         rewards=self.rewards, # dynamics function target
                         target_policies=self.target_policies,
                         root_values=self.root_values) # prediction function targets
        
        trajectory = copy.deepcopy(trajectory)
        self.buffer.append(trajectory)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        self.reset_trajectory()
        pass

    def sample_batch(self, k_unroll_steps):
        # return batches of trajectories
        # each of length k_unroll_steps
        batch = []

        num_eps = len(self.buffer)
        for b in range(self.batch_size):
            random_ep = self.buffer[random.randint(num_eps // 2, num_eps-1)]  # TODO: look into this "front-half-most-recent games" buffer lookup...
            observations, player_turns, actions = random_ep['obs'], random_ep['turns'], random_ep['actions']
            rewards, target_policies, root_values = random_ep['rewards'], random_ep['target_policies'], random_ep['root_values']
            
            # k_unroll_steps, capped k steps in trajectory for training
            ix = random.randint(0, len(random_ep) - k_unroll_steps -1)
            inputs = (observations[ix:ix+k_unroll_steps], player_turns[ix:ix+k_unroll_steps], actions[ix:ix+k_unroll_steps])

            # TODO: target_value same as final outcome for board games OR discounted from final_outcome based on td_steps
            # td_steps, n steps into the future for target_value
            # targets = (torch.tensor(rewards[ix:ix+k_unroll_steps], dtype=torch.float32), 
            #            target_policies[ix:ix+k_unroll_steps])



            sequence = (inputs, targets)
            batch.append(sequence)
        return batch


class MuZeroAgent(object):
    def __init__(self, environment, config):
        self.env = environment
        self.observation_space = self.flatten(environment.observation_space('player_1'))
        self.obs_size = self.observation_space.shape
        self.action_space = environment.action_space('player_1')
        self.action_size = self.action_space.n
        self.root_value = 0
        self.action_probs = torch.zeros(self.action_size)

        self.replay_buffer = ReplayBuffer(config['batch_size'])
        self.buffer_size = config['buffer_size']

        self.state_function = StateFunction(self.obs_size[0],
                                            config['state_size'],
                                            config['hidden_size'])
        
        self.dynamics_function = DynamicsFunction(config['state_size']+1,
                                                  config['state_size'],
                                                  config['hidden_size'])
        
        self.prediction_function = PredictionFunction(config['state_size'],
                                                      self.action_size,
                                                      config['hidden_size'])
        
        all_function_params = (list(self.state_function.parameters()) +
                               list(self.dynamics_function.parameters()) +
                               list(self.prediction_function.parameters()))
        
        # three neural nets' weights given to the optimizer
        self.optimizer = torch.optim.AdamW(params=all_function_params,
                                           lr=config['lr'],
                                           weight_decay=config['weight_decay'])

        self.min_Q = float('inf')
        self.max_Q = -float('inf')
        self.max_iters = config['max_iters']
        self.gamma = config['gamma']
        self.k_unroll_steps = config['k_unroll_steps']

        self.state_function.eval()
        self.dynamics_function.eval()
        self.prediction_function.eval()
        pass


    """ENIVORNMENT HELPER FUNCTIONS"""
    def flatten(self, observation_space):
        H, W, C = observation_space['observation'].shape
        # return observation space as a 1-D vector
        return torch.zeros(H*W)

    def preprocess_obs(self, observation):
        # pre-process observation dictionary into tensor
        obs = torch.zeros((3,3), dtype=torch.float32)
        current_player_plane = torch.tensor(observation["observation"][:, :, 0])
        opponent_plane = torch.tensor(observation["observation"][:, :, 1]) * 2
        obs = obs + current_player_plane + opponent_plane
        obs = obs.reshape((9,1)).squeeze()
        return obs

    def get_legal_actions(self, temp_board, history):
        temp_board = temp_board.clone()
        if len(history) > 0:
            for a in history:
                temp_board[a] = -1
        return torch.where(temp_board == 0)[0].tolist()


    def whose_turn(self, action_history):
        # based on the environment and action history in simulation
        # NOT the current player index in the game being played
        if self.env.agent_selection == 'player_1':
            if len(action_history) % 2 == 0:
                return 0 # player 1's turn
            else:
                return 1 # player 2's turn
        if self.env.agent_selection == 'player_2':
            if len(action_history) % 2 == 0:
                return 1 # player 2's turn
            else:
                return 0 # player 1's turn

    def update_min_max_Q(self, node_mean_value):
        # keep track of min Q and max Q over entire tree
        if node_mean_value > self.max_Q:
            self.max_Q = node_mean_value
        elif node_mean_value < self.min_Q:
            self.min_Q = node_mean_value
        pass

    def expansion(self, last_node, state, reward, policy_logits, legal_actions, action_history):
        last_node.state = state
        last_node.reward = reward
        last_node.current_player = self.whose_turn(action_history)

        # mask illegal actions and normalize the policy over legal moves
        policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for a in policy.keys():
            last_node.children[a] = Node(policy[a]/policy_sum)
        pass

    def pUCT(self, node, sum_visits):
        # refer to Equation 2, Appendix B
        c1 = 1.25
        c2 = 19652
        N = node.N
        if N > 0:
            Q = node.mean_value()
            if self.max_Q > self.min_Q:
                Q = (Q - self.min_Q) / (self.max_Q - self.min_Q)
            Q = node.R + self.gamma * Q 
        else:
            Q = 0
        P = node.P
        N_sum = sum_visits
        return (Q + (P*math.sqrt(N_sum)/(1+N))*(c1+math.log((N_sum+c2+1)/c2)))

    def select_child(self, node):
        # parent node visit count is the sum of its children's visit counts.
        sum_visits = node.N

        best_uct = -float('inf')
        best_child = None
        for a, child_node in node.children.items():
            uct = self.pUCT(child_node, sum_visits)
            if uct > best_uct:
                best_uct = uct
                best_action = a
                best_child = child_node
        return best_child, best_action

    def selection(self, node):
        search_path = [node]
        action_history = []

        while node.expanded():
            node, action = self.select_child(node)
            search_path.append(node)
            action_history.append(action)
        return node, search_path, action_history

    def backup(self, value, search_path, leaf_player):
        G = value
        for current_node in search_path.reverse():
            current_node.value_sum += G if current_node.current_player == leaf_player else -G
            current_node.N += 1
            self.update_min_max_Q(current_node.mean_value())
            G = current_node.R + self.gamma * G
        pass

    def select_action(self, node):
        # TODO: revise this to sample with softmax and temperature
        sum_visits = node.N
        self.action_probs = torch.zeros(self.action_size)
        max_visits = -1
        best_action = None
        for a, child_node in node.children.items():
            self.action_probs[a] = child_node.N / sum_visits
            if child_node.N > max_visits:
                max_visits = child_node.N
                best_action = a
        # print(f"Action Probabilities: {self.action_probs.tolist()}")
        return best_action

    def search(self, obs):
        with torch.no_grad():
            root_node = Node(0)

            initial_state = self.state_function(obs)
            policy_logits, value = self.prediction_function(initial_state)

            action_history = []
            legal_actions = self.get_legal_actions(obs, action_history)

            self.expansion(root_node, initial_state, 0, policy_logits, legal_actions, action_history)
            
            for _ in range(self.max_iters):
                last_node, search_path, action_history = self.selection(root_node)

                parent_node = search_path[-2]
                latest_action = torch.tensor(action_history[-1], dtype=torch.float32).unsqueeze(0)
                state, reward = self.dynamics_function(parent_node.state, latest_action)
                policy_logits, value = self.prediction_function(state)
                
                legal_actions = self.get_legal_actions(obs, action_history)
                self.expansion(last_node, state, reward, policy_logits, legal_actions, action_history)
                
                self.backup(value, search_path, self.whose_turn(action_history))
            self.root_value = search_path[0].value_sum
        return self.select_action(root_node)

    def experience(self, observation, player_turn, action, reward, terminal):
        obs = self.preprocess_obs(observation)
        self.replay_buffer.store_step(obs, player_turn, action, self.action_probs, reward, self.root_value)
        if terminal:
            # once final_outcome is nonzero, label the entire trajectory with the final outcome 
            # self.replay_buffer.final_outcomes = [final_outcome if i == player_turn else -final_outcome for i in self.replay_buffer.player_turns]
            self.replay_buffer.store_trajectory()
        pass

    def step(self, observation):
        obs = self.preprocess_obs(observation)
        action = self.search(obs)
        return action
    
    def scale_gradient(self, tensor, scale):
        return tensor * scale + tensor.detach() * (1.0 - scale)

    def update(self):
        if len(self.replay_buffer.buffer) > self.buffer_size:
            self.state_function.train()
            self.dynamics_function.train()
            self.prediction_function.train()

            loss = 0
            
            # load from batch
            batch = self.replay_buffer.sample_batch(self.k_unroll_steps)
            for sequence in batch:
                inputs, targets = sequence

                obs, player_turns, actions = inputs
                
                # neural nets predict: predicted_reward, policy_logits, predicted_value
                state = self.state_function(obs[0])
                predicted_reward = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)
                policy_logits, value = self.prediction_function(state)
                
                predictions = [(policy_logits, predicted_reward, value)]

                action_history = []
                for a in actions:
                    latest_action = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                    state, predicted_reward  = self.dynamics_function(state, latest_action)
                    policy_logits, value = self.prediction_function(state)

                    legal_actions = self.get_legal_actions(obs[0], action_history)
                    mask = torch.zeros_like(policy_logits, dtype=torch.bool)
                    mask[legal_actions] = True
                    policy_logits = policy_logits.masked_fill(~mask, -float('inf'))

                    state = self.scale_gradient(state, 0.5)

                    predictions.append((policy_logits, predicted_reward, value))
                    action_history.append(a)
                
                # loss
                # TODO: revisit length of predictions vs length of targets
                # for i, pred in enumerate(predictions):
                for i in range(len(targets[0])):
                    policy_logits, predicted_reward, value = predictions[i]
                    u, target_policy, target_value = targets[0][i], targets[1][i], targets[2][i]
                    u = u.unsqueeze(0)
                    target_value = target_value.unsqueeze(0)

                    # compare with corresponding
                    # immediate_reward, MSE
                    # target_policy, cross_entropy
                    # target_value, MSE
                    l = F.mse_loss(predicted_reward, u) + F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)
                    loss += self.scale_gradient(l, (1.0 / len(actions)))

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.state_function.eval()
            self.dynamics_function.eval()
            self.prediction_function.eval()