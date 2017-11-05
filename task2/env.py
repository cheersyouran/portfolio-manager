import numpy as np
import base
from rl.core import Env
from sklearn import preprocessing

class env(Env):
    def __init__(self):
        self.portcodes = ['ZH000199', 'ZH000283']
        self.port_num = len(self.portcodes)
        s1 = base.load_states_csv()
        # s1.iloc[:, 3:] = preprocessing.scale(s1.iloc[:, 3:])
        self.s1 = s1.drop(s1.index[:18])
        # self.s1 = s1
        self.nav = base.load_nav_csv()
        self.count = 1
        self.action_space = np.zeros(self.port_num + 1, )
        self.observation_space = np.zeros(3,)
        self.current_reward = 0

        print("################## INIT ENV ####################")
        print("PortCodes: ", self.portcodes )
        print("Start Date: ",  self.s1.iloc[0, 0])
        print("################################################")

    def seed(self, seed=None):
        pass

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def configure(self, *args, **kwargs):
        pass

    def step(self, action):
        return self.get_observation(), self.get_reward(action), self.whether_done(),  {}

    def reset(self):
        self.count = 1
        return self.get_observation()

    def __del__(self):
        pass

    def get_observation(self):
        obs = self.s1.iloc[self.count, 2:]
        return obs

    def get_reward(self, action):
        date_ = self.s1.iloc[self.count, 0]
        date = self.s1.iloc[self.count-1, 0]
        r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values
        r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values

        ratio = (r_-r)/r
        ratio = np.insert(ratio, 0, 0)
        reward = sum(ratio*action[:])

        if(reward > np.max(ratio)*0.95):
            # reward = reward/np.max(ratio)
            reward = 1
        else:
            reward = -1
        self.current_reward = reward
        self.count = self.count + 1
        return reward

    def whether_done(self):
        return False







