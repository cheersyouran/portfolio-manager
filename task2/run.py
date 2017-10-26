from task2.model import model1
from task2.env import env

env = env()
ddpg = model1.ddpg(env)
ddpg.fit()

ddpg.test()