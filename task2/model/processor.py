import numpy as np

from rl.core import Processor

class ShowActionProcessor(Processor):

    def __init__(self, agent):
        self.agent = agent

    def process_action(self, action):
        if( self.agent.training == False):
            print('Step: ', self.agent.step, action)
        return action