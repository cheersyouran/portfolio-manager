from task2.model import DDPGmodel
from task2.model import DQNmodel
from task2.env import env

train_window = 30
test_window = 10

env = env(train_window,test_window)
def ddpg_test():

    ddpg = DDPGmodel.DDPG(env)
    ddpg.fit()
    # ddpg.load_weights()
    ddpg.test()
    ddpg.save_weights()

def dqn_test():

    dqn = DQNmodel.dqn(env)
    env.phase = 'Train'
    dqn.fit()
    env.phase = 'Test'
    dqn.test()

dqn_test()