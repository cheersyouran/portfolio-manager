import numpy as np
import pandas as pd

#learning_rate=0.1, gamma=0, e_greedy=1时候，最多迭代action次

class QLearningModel:
    def __init__(self, actions, learning_rate=0.1, gamma=0, e_greedy=0.9, random_step=200):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_frequent = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.random_step = random_step

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # choose action randomly(random_step), to accumulate ddds distribution(赔率分布) for each action.
        if (np.random.uniform() < self.epsilon) & (self.random_step < 0):
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        self.random_step = self.random_step - 1
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
        self.q_table_frequent.ix[s, a] += 1

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state,))
        if state not in self.q_table_frequent.index:
            self.q_table_frequent = self.q_table_frequent.append(pd.Series([0]*len(self.actions),index=self.q_table_frequent.columns,name=state,))