import numpy as np
import base
from rl.core import Env
from sklearn import preprocessing

class env(Env):
    def __init__(self, train_window, test_window):
        self.portcodes = ['ZH010630', 'ZH016987']
        self.port_num = len(self.portcodes)
        s1 = base.load_states_csv()
        s1.iloc[:, 1:] = preprocessing.scale(s1.iloc[:, 1:])
        self.s1 = s1.drop(s1.index[:40])
        self.nav = base.load_nav_csv()
        self.count = 1
        self.action_space = np.zeros(self.port_num + 1, )
        self.observation_space = np.zeros(2,)
        self.current_reward = 0
        self.phase = 'Train'
        self.train_window = train_window
        self.test_window = test_window
        self.done = False

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
        return self.get_observation(), self.get_reward_for_dqn(action), self.whether_done(),  {}

    def reset(self):
        self.count = 1
        self.done = False
        return self.get_observation()

    def __del__(self):
        pass

    def get_observation(self):
        if (self.phase == 'Train'):
            return self.get_train_obervation()
        else:
            return self.get_test_obervation()

    def get_train_obervation(self):
        obs = self.s1.iloc[self.count, 1:]
        return obs

    def get_test_obervation(self):
        obs = self.s1.iloc[self.test_window + self.count, 1:]
        return obs

    def get_reward_for_dqn(self,action):
        date_ = self.s1.iloc[self.count, 0]
        date = self.s1.iloc[self.count - 1, 0]
        r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values
        r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values

        ratio = (r_ - r) / r
        ratio = np.insert(ratio, 0, 0)
        reward = ratio[action]

        if (reward >= np.max(ratio) * 0.9):
            # reward = 10 * reward / np.max(ratio)
            reward = 10
        else:
            reward = -10
        self.current_reward = reward
        self.count = self.count + 1
        return reward

    def get_reward_for_ddpg(self, action):
        date_ = self.s1.iloc[self.count, 0]
        date = self.s1.iloc[self.count-1, 0]
        r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values
        r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'].isin(self.portcodes))]['Nav'].values

        ratio = (r_-r)/r
        ratio = np.insert(ratio, 0, 0)
        reward = sum(ratio*action[:])

        if(reward > np.max(ratio)*0.9):
            # reward = 10 * reward/np.max(ratio)
            reward = 10
        else:
            reward = -10
        self.current_reward = reward
        self.count = self.count + 1
        return reward

    def whether_done(self):
        if (self.phase == 'Train'):
            if(self.count == self.train_window + 1):
                return True
        elif(self.phase == 'Test'):
            if(self.count == self.test_window + 1):
                return True
        else:
            return False






