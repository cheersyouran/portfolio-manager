from task2.model import DDPGmodel
from task2.model import DQNmodel
from task2.env import env

env = env()
def ddpg_test():

    ddpg = DDPGmodel.DDPG(env)

    train_history = ddpg.fit()
    # ddpg.load_weights()

    history = ddpg.test()
    ddpg.save_weights()

def dqn_test():

    dqn = DQNmodel.dqn(env)
    dqn.fit()
    dqn.test()

dqn_test()