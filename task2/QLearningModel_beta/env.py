import numpy as np
import pandas as pd
import base
from State_Space import feat_selection

class Env():
    def __init__(self, train_window, test_window, start_from, market):
        self.nb_portcodes = 2
        self.count = 0
        self.portcodes = []
        industry_quote, records, nav = market.get_past_market()
        self.nav = nav
        self.records = records
        self.IR_rank = pd.read_csv('~/Projects/PortfolioManagement/State_Space/IR_rank_week.csv', index_col=0)
        self.window_states = self.generate_state(industry_quote).iloc[-(test_window + train_window):].round(0)
        self.current_state = None
        self.next_state = None
        self.phase = 'Train'
        self.train_window = train_window
        self.test_window = test_window
        self.done = False

        # print("################## INIT ENV ####################")
        # print("PortCodes: ", self.portcodes )
        # print("Start Date: ", self.states.iloc[0, 0])
        # print("################################################")

    def step(self, action):
        reward = self.get_reward(action)
        done = self.whether_done()
        self.count = self.count + 1
        obs = self.get_obs()
        return obs, reward, done

    def get_obs(self):
        self.current_state = self.window_states.iloc[self.count]
        self.next_state = self.window_states.iloc[self.count + 1]
        return self.current_state

    def get_reward(self, action):

        def get_today_portcodes(df, day):
            p = df[df['NavDate'] == day]['PortCode']
            p = feat_selection.search_port('银行', self.records, self.IR_rank, port_list=p, output=2)
            return p

        date_ = self.current_state.name
        date = self.next_state.name
        rs = np.array([])
        rs_ = np.array([])
        self.portcodes = get_today_portcodes(self.nav, date)
        for p in self.portcodes:
            r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'] == p)]['Nav'].values
            r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'] == p)]['Nav'].values
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
        if (self.phase == 'Train'):
            return self.count == self.train_window
        else:
            return self.count == self.test_window

    def generate_state(self, df_industry_quote):
        status = feat_selection.generate_states('银行', df_industry_quote, ['MACD', 'KDJ', [5, 10], [1, 3], 'BOLL'])
        return status

