import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from task2.Processor.processor import ShowActionProcessor

class dqn():
    def __init__(self, Env):
        self.env = Env
        nb_actions = self.env.action_space.shape[0]

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        self.model = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy, gamma=0)
        self.model.processor = ShowActionProcessor(self.model, self.env)
        self.model.compile(Adam(lr=1e-2), metrics=['mae'])

    def fit(self):
        self.model.fit(self.env, nb_steps=30000, visualize=False, verbose=2, nb_max_episode_steps=100)

    def save_weights(self):
        self.model.save_weights('./store/dqn_{}_weights.h5f'.format('porfolio'), overwrite=True)

    def test(self):
        self.model.test(self.env, nb_episodes=1, visualize=False, nb_max_episode_steps=100)