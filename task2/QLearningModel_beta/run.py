import sys
import threading
sys.path.append('/Users/Youran/Projects/PortfolioManagement')
from task2.QLearningModel_beta.env import Env
from task2.QLearningModel_beta.QLearningModel import QLearningModel
from task2.QLearningModel_beta.market import Market

train_window = 150
test_window = 30

market = Market()

def run_a_kind_of_model(kind, features):
    print('#######################################################################')
    print('Kind:', kind)
    while True:
        env = Env(train_window, test_window, market, kind, features)
        RL = QLearningModel(actions=list(range(env.nb_portcodes + 1)))
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
            print('Episode: ', episode, 'Rewards: ', rewards, 'Accuracy: ', ratio)
            if (ratio > 0.9) | (episode >= 10):
                break
            episode = episode + 1

        rewards = 0
        obs = env.get_obs()
        while True:
            a = RL.choose_action(str(obs))
            obs_, r, done = env.step(a)
            rewards = rewards + r
            print('Action: ', a, 'Reward: ', r)
            obs = obs_
            # fd = open('result.csv', 'a')
            # for i in range(env.nb_portcodes) :
            #     if (a == i):
            #         fd.write(''.join([str(market.current_date),',', env.portcodes[i],',','1', '\n']))
            #     else:
            #         fd.write(''.join([str(market.current_date),',',env.portcodes[i],',','0', '\n']))
            # fd.close()
            if done:
                break
        ratio = rewards / (test_window * 10)
        print('Rewards: ', rewards, 'Accuracy: ', ratio)
        market.pass_a_day()

# run_a_kind_of_model('银行')
# t1 = run_a_kind_of_model('计算机', [[1, 2], [1, 3], [1, 4], [1, 5]])
# t2 = run_a_kind_of_model('计算机', [[1, 2]])
# t3 = run_a_kind_of_model('计算机', ['MACD', 'BOLL', [1,3]])
# t4 = run_a_kind_of_model('银行', [[1, 2], [1, 3], [1, 4], [1, 5]])


added_thread = threading.Thread(target=run_a_kind_of_model('计算机', [[1, 3]]), name='T1')
