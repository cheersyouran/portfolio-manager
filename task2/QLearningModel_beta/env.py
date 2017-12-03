import numpy as np

from State_Space import feat_selection

class Env():
    def __init__(self, train_window, test_window, market, kind, features, irrank):
        self.nb_portcodes = 3
        self.count = 0
        self.industry_quote, self.records, self.nav, self.quote = market.get_past_market()
        self.IR_rank = irrank
        self.portcodes = feat_selection.search_port(kind, self.records, self.nav, self.IR_rank, day=train_window+test_window, output=3)
        self.current_state = None
        self.next_state = None
        self.phase = 'Train'
        self.train_window = train_window
        self.test_window = test_window
        self.done = False
        self.kind = kind
        self.window_states = None
        self.features = features

        print(self.kind, ", " , self.portcodes, ", ", market.current_date)

    def step(self, action):
        reward = self.get_reward(action)
        self.whether_done()
        self.count += 1
        obs_ = self.get_obs()
        return obs_, reward, self.done

    def get_obs(self):
        if self.window_states is None:
            self.window_states = self.generate_state(self.industry_quote, self.features).iloc[-(self.test_window + self.train_window + 1):].round(0)
        self.current_state = self.window_states.iloc[self.count]
        return self.current_state.values

    def get_reward(self, action):
        self.next_state = self.window_states.iloc[self.count + 1]
        date = self.current_state.name
        date_ = self.next_state.name

        rs = np.array([])
        rs_ = np.array([])

        if self.portcodes.size == 0:
            self.portcodes = np.zeros(self.nb_portcodes)
        for p in self.portcodes:
            r = self.nav[(self.nav['NavDate'] == date) & (self.nav['PortCode'] == p)]['Nav'].values
            r_ = self.nav[(self.nav['NavDate'] == date_) & (self.nav['PortCode'] == p)]['Nav'].values
            rs = np.append(rs, r if r.size == 1 else 0)
            rs_ = np.append(rs_, r_ if r_.size == 1 else 0)
        ratio = np.nan_to_num((rs_ - rs) / rs)

        ratio = np.insert(ratio, 0, 0)
        reward = ratio[action]

        # if reward >= np.max(ratio) * 0.9:
        #     reward = 10
        # else:
        #     reward = -10

        return reward * 1000

    def reset(self):
        self.count = 0
        self.done = False

    def whether_done(self):
        if (self.count == self.train_window - 1) | (self.count == self.train_window + self.test_window - 1):
            self.done = True
        else:
            self.done = False

    def generate_state(self, df_industry_quote, feature):
        status = feat_selection.generate_states(self.kind, df_industry_quote, feature)
        return status

