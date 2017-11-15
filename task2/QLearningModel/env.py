import numpy as np
import base

class Env():
    def __init__(self, train_window, test_window):
        self.portcodes = ['ZH010630']
        data = base.load_certain_states_csv()
        self.states = data.drop(data.index[:40]).round(0)
        self.nav = base.load_nav_csv()
        self.count = 1
        self.phase = 'Train'
        self.train_window = train_window
        self.test_window = test_window
        self.done = False

        print("################## INIT ENV ####################")
        print("PortCodes: ", self.portcodes )
        print("Start Date: ", self.states.iloc[0, 0])
        print("################################################")

    def step(self, action):
        reward = self.get_reward(action)
        done = self.whether_done()
        self.count = self.count + 1
        obs_ = self.get_observation()
        return obs_, reward, done

    def get_observation(self):
        if (self.phase == 'Train'):
            return self.get_train_obervation()
        else:
            return self.get_test_obervation()

    def get_train_obervation(self):
        if(self.count > self.train_window):
            return 'terminal'
        else:
            return self.states.iloc[self.count, 1:]

    def get_test_obervation(self):
        if (self.count > self.test_window):
            return 'terminal'
        else:
            return self.states.iloc[self.test_window - 1 + self.count, 1:]

    def get_reward(self, action):
        date_ = self.states.iloc[self.count, 0]
        date = self.states.iloc[self.count - 1, 0]
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
        return reward

    def reset(self):
        self.count = 1
        self.done = False
        return self.get_observation()

    def whether_done(self):
        if (self.phase == 'Train'):
            return self.count == self.train_window
        else:
            return self.count == self.test_window