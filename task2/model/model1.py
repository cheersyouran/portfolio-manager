from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

class ddpg():
    def __init__(self, Env):
        self.env = Env
        nb_actions = self.env.action_space.shape[0]
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('linear'))
        print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + Env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = merge([action_input, flattened_observation], mode='concat')
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(input=[action_input, observation_input], output=x)
        print(critic.summary())

        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    def fit(self):
        self.agent.fit(self.env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

    def save_weights(self):
        self.agent.save_weights('./store/ddpg_{}_weights.h5f'.format("porfolio"), overwrite=True)

    def test(self):
        self.agent.test(self.env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)