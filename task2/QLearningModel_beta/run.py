#encoding:utf-8
import sys
import threading
sys.path.append('/Users/Youran/Projects/PortfolioManagement')
from task2.QLearningModel_beta.env import Env
from task2.QLearningModel_beta.QLearningModel import QLearningModel
from task2.QLearningModel_beta.market import Market
from State_Space.Update_IR_rank import Update_IR_rank
import base

train_window = 10
test_window = 1

market = Market()

def run_a_kind_of_model(kind, features, irrank):
    ir = irrank
    print('Kind:', kind)
    while True:
        env = Env(train_window, test_window, market, kind, features, ir)
        RL = QLearningModel(actions=list(range(env.nb_portcodes + 1)), random_step=300)
        count = 0
        episode = 1
        while True:
            rewards = 0
            env.reset()
            obs = env.get_obs()
            while True:
                a = RL.choose_action(str(obs))
                obs_, r, done = env.step(a)
                RL.learn(str(obs), a, r, str(obs_))
                rewards = rewards + r
                obs = obs_
                count = count + 1
                if done:
                    break
            ratio = rewards / (train_window * 10)
            ir = Update_IR_rank(str(market.current_date), env.records, env.industry_quote, env.nav, env.quote, ir)

            print('Episode: ', episode, 'Rewards: ', rewards, 'Accuracy: ', ratio)
            if (ratio > 0.9) | (episode >= 10):
                break
            episode = episode + 1

        rewards = 0
        obs = env.get_obs()
        fd = open('result.csv', 'a')
        while True:
            a = RL.choose_action(str(obs))
            obs_, r, done = env.step(a)
            rewards = rewards + r
            print('Action: ', a, 'Reward: ', r)
            obs = obs_
            for i in range(env.nb_portcodes):
                if (a==i):
                    fd.write(''.join([str(market.current_date), ',', env.portcodes[i], ',', '1', '\n']))
                else:
                    fd.write(''.join([str(market.current_date), ',', env.portcodes[i], ',', '0', '\n']))
            if done:
                break
        fd.close()
        market.pass_a_day()

irrank = base.load_irweek_csv()
run_a_kind_of_model('计算机', [[1, 2]], irrank)
# t1 = threading.Thread(target=run_a_kind_of_model, args=('计算机', [[1, 2]], irrank), name='T1')
# t1.start()
# t2 = threading.Thread(target=run_a_kind_of_model, args=('银行', [[1, 2]], irrank), name='T2')
# t2.start()
