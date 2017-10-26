from task2.model import model1
from task2.env import env

env = env()
ddpg = model1.DDPG(env)
ddpg.fit()

history = ddpg.test()