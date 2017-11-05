from task2.model import model1
from task2.env import env

env = env()
ddpg = model1.DDPG(env)

# train_history = ddpg.fit()
ddpg.load_weights()

history = ddpg.test()
ddpg.save_weights()