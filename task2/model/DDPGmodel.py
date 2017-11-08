from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from task2.model.processor import ShowActionProcessor

class DDPG():
    def __init__(self, Env):
        self.env = Env
        nb_actions = self.env.action_space.shape[0]
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        actor.add(Dense(5))
        actor.add(Activation('relu'))
        actor.add(Dense(8))
        actor.add(Activation('relu'))
        actor.add(Dense(5))
        actor.add(Activation('relu'))
        # actor.add(Dense(16))
        # actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('softmax'))
        # print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + Env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = concatenate([action_input, flattened_observation], name = 'concatenate')
        x = Dense(5)(x)
        x = Activation('relu')(x)
        x = Dense(8)(x)
        x = Activation('relu')(x)
        x = Dense(5)(x)
        x = Activation('relu')(x)
        # x = Dense(32)(x)
        # x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        # print(critic.summary())

        memory = SequentialMemory(limit=100000, window_length=1)
        # random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        random_process = None
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=32, nb_steps_warmup_actor=32,
                          random_process=random_process, gamma=0, target_model_update=0.001)
        self.agent.processor = ShowActionProcessor(self.agent, self.env)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    def fit(self):
        history = self.agent.fit(self.env, action_repetition=1, nb_steps=20000, visualize=False, verbose=1, nb_max_episode_steps=10)
        return history

    def save_weights(self):
        self.agent.save_weights('./store/ddpg_{}_weights2.h5f'.format("porfolio"), overwrite=True)

    def test(self):
        history = self.agent.test(self.env, nb_episodes=1, visualize=False, nb_max_episode_steps=10)
        return history

    def load_weights(self):
        self.agent.load_weights('./store/ddpg_{}_weights2.h5f'.format("porfolio"))