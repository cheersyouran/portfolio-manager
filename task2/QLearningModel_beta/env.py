import numpy as np
import pandas as pd
import threading
from State_Space import feat_selection

class Env():
    def __init__(self, train_window, test_window, market, kind=None):
        self.nb_portcodes = 2
        self.count = 0
        self.portcodes = []
        self.industry_quote, self.records, self.nav = market.get_past_market()
        self.IR_rank = pd.read_csv('~/Projects/PortfolioManagement/State_Space/IR_rank_week.csv', index_col=0)
        self.current_state = None
        self.next_state = None
        self.phase = 'Train'
        self.train_window = train_window
        self.test_window = test_window
        self.done = False
        self.kind = kind
        self.window_states = None

        # print("################## INIT ENV ####################")
        # print("PortCodes: ", self.portcodes )
        # print("Start Date: ", self.states.iloc[0, 0])
        # print("################################################")

    def step(self, action):
        reward = self.get_reward(action)
        self.whether_done()
        self.count = self.count + 1
        obs_ = self.get_obs()
        return obs_, reward, self.done

    def get_obs(self):
        if(self.window_states is None):
            self.window_states = self.generate_state(self.industry_quote).iloc[-(self.test_window + self.train_window + 1):].round(0)
        self.current_state = self.window_states.iloc[self.count]
        return self.current_state.values

    def get_reward(self, action):

        def get_today_portcodes(df, date, kind):
            p = df[df['NavDate'] == date]['PortCode']
            p = feat_selection.search_port(kind, self.records, self.IR_rank, port_list=p, output=2)
            return p

        self.next_state = self.window_states.iloc[self.count + 1]
        date = self.current_state.name
        date_ = self.next_state.name
        rs = np.array([])
        rs_ = np.array([])
        self.portcodes = get_today_portcodes(self.nav, date, self.kind)
        for p in self.portcodes:
            r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'] == p)]['Nav'].values
            r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'] == p)]['Nav'].values
            rs = np.append(rs, r if r.size==1 else 0)
            rs_ = np.append(rs_, r_ if r_.size==1 else 0)

        ratio = (rs_ - rs) / rs
        ratio = np.insert(ratio, 0, 0)
        reward = ratio[action]
        if (reward >= np.max(ratio) * 0.9):
            reward = 10
        else:
            reward = -10
        return reward

    def reset(self):
        self.count = 0
        self.done = False

    def whether_done(self):
        if (self.count == self.train_window - 1) | (self.count == self.train_window + self.test_window - 1):
            self.done = True
        else:
            self.done = False

    def generate_state(self, df_industry_quote):
        status = feat_selection.generate_states(self.kind, df_industry_quote, ['MACD', [5, 10], [2, 5], 'BOLL'])
        return status

