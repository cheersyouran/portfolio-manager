from task2.DQNmodel import DQNmodel
from task2.DQNmodel.env import env

train_window = 30
test_window = 10

env = env(train_window, test_window)

def dqn_test():

    dqn = DQNmodel.dqn(env)
    env.phase = 'Train'
    dqn.fit()
    env.phase = 'Test'
    dqn.test()

dqn_test()