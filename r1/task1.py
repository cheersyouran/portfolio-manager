import numpy as np

from r1 import config, utils
from r1.Model.model1 import Model1
from r1.env import Env

TIME_SPACE = config.TIME_SPACE
STOCK_NUM = config.STOCK_NUM
HISTORY = config.HISTORY
ALL_HISTORY = config.ALL_HISTORY
PATH = config.PATH


def train():
    print('##############################TRAIN###################################################')
    for loop in range(10):
        env.long_time_reward_ratio = 1
        for i in range(HISTORY):
            s = env.state(i)  # [30, 4, 3]
            a = model.choose_action(s)  # [1,1,5,1]
            s_, r, R = env.step(i, a)  # r[1,]   s_[30,4,3]
            model.learn(s, a, r*100, s_, loop * HISTORY+i)
            # s = s_
            if i % 50 == 0:
                print("LOOP: %d" %loop)
                print("the %d day action is :" % i, np.reshape(a, [STOCK_NUM + 1, ]))
                print("the %d day r is :" % i, r, ", R is :", R)


def run():
    print('##############################TEST####################################################')
    env.long_time_reward_ratio = 1
    for i in range(HISTORY):
        s = env.state(i)  # [30, 3, 3]
        a = model.choose_action(s)  # [1,1,3,1]
        W_close = np.reshape(a, [STOCK_NUM + 1, 1])
        if i == 0:
            actions = W_close
        else:
            actions = np.hstack([actions, W_close])
        s_, r, R = env.step(i, a)  # r[1,]   s_[30,4,3]
        s = s_
        if i % 20 == 0:
            print("the %d day action is :" % i, np.reshape(a, [STOCK_NUM + 1, ]))
            print("the %d day r is :" % i, r, ", R is :", R)

    reward_ratio = env.total_reward_ratio(actions)
    utils.draw_reward_ratio_plot(PATH, reward_ratio)
    # utils.draw_stocks_price_plot(PATH)


model = Model1(False)
env = Env()

train()
run()