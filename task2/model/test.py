from __future__ import division
from __future__ import absolute_import

from rl.core import Env
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, merge, Dense, Flatten
import random
from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory

def test_single_ddpg_input():
    nb_actions = 2

    actor = Sequential()
    actor.add(Flatten(input_shape=(2, 3)))
    actor.add(Dense(nb_actions))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(2, 3), name='observation_input')
    x = merge([action_input, Flatten()(observation_input)], mode='concat')
    x = Dense(1)(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=10, window_length=2)
    agent = DDPGAgent(actor=actor, critic=critic, critic_action_input=action_input, memory=memory,
                      nb_actions=2, nb_steps_warmup_critic=5, nb_steps_warmup_actor=5, batch_size=4)
    agent.compile('sgd')
    agent.fit(MultiInputTestEnv((3,)), nb_steps=10)


class MultiInputTestEnv(Env):
    def __init__(self, observation_shape):
        self.observation_shape = observation_shape

    def step(self, action):
        return self._get_obs(), random.choice([0, 1]), random.choice([True, False]), {}

    def reset(self):
        return self._get_obs()

    def _get_obs(self):
        if type(self.observation_shape) is list:
            return [np.random.random(s) for s in self.observation_shape]
        else:
            return np.random.random(self.observation_shape)

    def __del__(self):
        pass

test_single_ddpg_input()