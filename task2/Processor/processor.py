from rl.core import Processor

class ShowActionProcessor(Processor):

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.step = 1

    def process_action(self, action):
        if( self.agent.training == False):
            if(self.step == self.agent.step):
                print('Step:', self.agent.step,"Action:",  action, "Reward:" , self.env.current_reward)
                self.step = self.step + 1

        return action