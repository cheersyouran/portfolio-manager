import pandas as pd
import numpy as np
import base
from rl.core import Env
from datetime import datetime

class env(Env):
    def __init__(self):
        self.__init__()
        self.s1 = base.load_strategy1_data().drop(['SecuAbbr', 'label'], axis=1)

    def seed(self, seed=None):
        pass

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def configure(self, *args, **kwargs):
        pass

    def step(self, action):
        return self.get_observation(), self.get_reward(), self.whether_done(), None

    def reset(self):
        pass
################need to implement#########################
    def get_observation(self):
        return

    def get_reward(self):
        return

    def whether_done(self):
        return







