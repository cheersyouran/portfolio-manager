from task2.QLearningModel_beta.env import Env
from task2.QLearningModel_beta.QLearningModel import QLearningModel
from task2.QLearningModel_beta.market import Market

train_window = 30
test_window = 10
start_from = 300
env = Env(train_window, test_window, start_from, Market())
RL = QLearningModel(actions=list(range(env.nb_portcodes+1)))

env.phase = 'Train'
episode = 1
while True:
    rewards = 0
    env.reset()
    obs = env.get_obs()
    while True:
        a = RL.choose_action(str(obs))
        obs_, r, done = env.step(a)
        RL.learn(str(obs), a, r, str(obs_))
        rewards = rewards + r
        obs = obs_
        if done:
            break
    ratio = rewards/(train_window*10)
    print('Episode: ', episode, 'Rewards: ', rewards, 'Accuracy: ', ratio)
    if(ratio > 0.9):
        break
    episode = episode + 1

print('#######################################################################')

env.phase = 'Test'
rewards = 0
obs = env.get_obs()
while True:
    obs, r, done = env.step(a)
    a = RL.choose_action(str(obs))
    rewards = rewards + r
    print('Action: ', a, 'Reward: ', r)
    if done:
        break
        
ratio = rewards / (test_window * 10)
print('Rewards: ', rewards, 'Accuracy: ',ratio)

