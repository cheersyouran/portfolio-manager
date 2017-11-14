from task2.QLearningModel.env import Env
from task2.QLearningModel.QLearningModel import QLearningModel
train_window=500
test_window=10
env = Env(train_window, test_window)
RL = QLearningModel(actions=list(range(len(env.portcodes)+1)))

env.phase = 'Train'
episode = 1
while True:
    obs = env.reset()
    rewards = 0
    while True:
        a = RL.choose_action(str(obs))
        obs_, r, done = env.step(a)
        rewards = rewards + r
        RL.learn(str(obs), a, r, str(obs_))
        obs = obs_
        if done:
            break
    ratio = rewards/(train_window*10)
    print('Episode: ', episode, 'Rewards: ', rewards, 'Accuracy: ', ratio)
    if(ratio > 0.9):
        break;
    episode = episode + 1

print('#######################################################################')

env.phase = 'Test'
observation = env.reset()
rewards = 0
while True:
    a = RL.choose_action(str(obs))
    obs_, r, done = env.step(a)
    rewards = rewards + r
    obs = obs_
    print('Action: ', a, 'Reward: ', r)
    if done:
        break
        
ratio = rewards / (test_window * 10)
print('Rewards: ', rewards, 'Accuracy: ',ratio)

