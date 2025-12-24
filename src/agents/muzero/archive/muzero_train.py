from envs.tictactoe import tictactoe
from agents.muzero.muzero import MuZeroAgent

import datetime
# env = tictactoe.env(render_mode="human")
env = tictactoe.env()

config = {
    'batch_size': 8,
    'buffer_size': 100,
    'state_size': 16,
    'hidden_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'max_iters': 1000,
    'gamma': 0.997,
    'k_unroll_steps': 5
}

agent1 = MuZeroAgent(environment=env, config=config)

agent_list = [agent1 , 'random']

p1_record = [0,0,0] #W-L-D

start_time = datetime.datetime.now()
for ep in range(100):
    
    idx = 0
    p1_reward, p2_reward = 0, 0
    recorded = False
    env.reset(seed=42)
    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if not recorded:
                p1_reward, p2_reward = env.rewards['player_1'], env.rewards['player_2']
                recorded = True
                if a == 'player_1':
                    agent1.experience(observation, a, action, 
                                    p1_reward, p1_reward, termination)
                else:
                    agent1.experience(observation, a, action, 
                                    p2_reward, p2_reward, termination)
                if p1_reward > 0:
                    p1_record[0] += 1
                elif p1_reward < 0:
                    p1_record[1] += 1
                else:
                    p1_record[2] += 1
        else:
            # if a == 'player_1':
            #     print('\nPLAYER X TURN')
            # else:
            #     print('\nPLAYER O TURN')
            mask = observation["action_mask"]
            if agent == 'random':
                action = int(env.action_space(a).sample(mask))
            else:
                action = agent.step(observation)

        env.step(action)
        if not recorded:
            if a == 'player_1':
                agent1.experience(observation, a, action, 
                                p1_reward, p1_reward, termination)
            else:
                agent1.experience(observation, a, action, 
                                p2_reward, p2_reward, termination)
        idx = (idx+1) % 2
    agent1.update()
    print(f'{datetime.datetime.now()-start_time} EP={ep}, {p1_record}')
    # pause = input('\npress enter for new game')
env.close()



env = tictactoe.env(render_mode="human")

agent_list = [agent1 , 'random']

for ep in range(3):
    idx = 0
    env.reset(seed=42)
    for a in env.agent_iter():
        agent = agent_list[idx]
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None

        else:
            # if a == 'player_1':
            #     print('\nPLAYER X TURN')
            # else:
            #     print('\nPLAYER O TURN')
            mask = observation["action_mask"]
            if agent == 'random':
                action = int(env.action_space(a).sample(mask))
            else:
                action = agent.step(observation)
                for a in range(len(agent1.action_probs)):
                    print(f"Action Probability {a} = {(agent1.action_probs[a]*100):.2f}%")
                print()
                

        env.step(action)
        pause = input('\npress enter to continue')
        idx = (idx+1) % 2
    pause = input('\npress enter for new game')
env.close()