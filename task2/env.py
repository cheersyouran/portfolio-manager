import pandas as pd
import numpy as np
import base
from rl.core import Env

class env(Env):
    def __init__(self):
        self.portcodes = ['ZH000199']
        self.port_num = len(self.portcodes)
        self.s1 = base.load_strategy1_data()
        self.nav = base.load_nav_csv()
        self.count = 0
        self.action_space = np.zeros(self.port_num + 1, )
        self.observation_space = np.zeros(6,)

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
        self.count = 0
        return self.get_observation()

    def __del__(self):
        pass

    def get_observation(self):
        obs = self.s1.iloc[self.count, 4:]
        return obs

    def get_reward(self, action):
        date = self.s1.iloc[self.count, 2]
        rewards = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values
        rewards = np.insert(rewards, 0, 1)
        self.count = self.count + 1
        return sum(rewards*action) - 1

    def whether_done(self):
        return False






