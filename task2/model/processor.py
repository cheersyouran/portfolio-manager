from rl.core import Processor

class ShowActionProcessor(Processor):

    def __init__(self, agent):
        self.agent = agent
        self.step = 1

    def process_action(self, action):
        if( self.agent.training == False):
            if(self.step == self.agent.step):
                print('Step: ', self.agent.step, action)
                self.step = self.step + 1

        return action