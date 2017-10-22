import numpy as np

from r1 import config, utils

TIME_SPACE = config.TIME_SPACE
STOCK_NUM = config.STOCK_NUM
HISTORY = config.HISTORY
ALL_HISTORY = config.ALL_HISTORY
PATH = config.PATH

class Env():
    def __init__(self):
        self.X = utils.load_data(PATH)
        # [719, STOCK_NUM]
        v_close = self.X[:, :, 0]
        v_high = self.X[:, :, 1]
        v_low = self.X[:, :, 2]

        # [718,3]
        self.Y_close = np.hstack([np.ones([ALL_HISTORY-1, 1]), np.divide(v_close[1:, :], v_close[:-1, :])])
        self.long_time_reward_ratio = 1

    def state(self, start):
        return self.X[start:start+TIME_SPACE, :, :]

    def step(self, day, action):
        w_close = np.reshape(action, [STOCK_NUM+1, 1])
        s_ = self.state(day + 1)
        # 当日收益率 [1,1] # np.divide(np.log(np.matmul(self.Y_close[day], w_close)), np.log(math.e)).reshape(-1, 1)*100
        daily_reward_ratio = np.sum(np.multiply(self.Y_close[TIME_SPACE + day], w_close.T), 1)[np.newaxis,:]
        self.long_time_reward_ratio *= daily_reward_ratio
        return s_, daily_reward_ratio - 1 , self.long_time_reward_ratio - 1

    def total_reward_ratio(self, actions):
        # 每日收益率矩阵[HISTORY,] （未减一）
        yw = np.multiply(self.Y_close[TIME_SPACE: HISTORY + TIME_SPACE], actions.T)
        r = np.sum(yw, 1)
        return r
