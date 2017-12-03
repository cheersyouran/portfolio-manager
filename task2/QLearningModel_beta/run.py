#encoding:utf-8
from task2.QLearningModel_beta.env import Env
from task2.QLearningModel_beta.QLearningModel import QLearningModel
from State_Space.Update_IR_rank import Update_IR_rank
import run_task2

train_window = run_task2.train_window
test_window = run_task2.test_window

def run_model(kind, features, irrank, market):
    ir = irrank
    env = Env(train_window, test_window, market, kind, features, ir)
    model = QLearningModel(actions=list(range(env.nb_portcodes + 1)), random_step=train_window)
    count = 0
    episode = 1
    while episode < 4:
        rewards = 0
        env.reset()
        obs = env.get_obs()
        while True:
            a = model.choose_action(str(obs))
            obs_, r, done = env.step(a)
            model.learn(str(obs), a, r, str(obs_))
            rewards = rewards + r
            obs = obs_
            count += 1
            if done:
                break
        print('Kind:', kind, ', Episode:', episode, ', Rewards:', rewards)
        ir = Update_IR_rank(str(market.current_date), env.records, env.industry_quote, env.nav, env.quote, ir)
        episode += 1

    rewards = 0
    obs = env.get_obs()
    fd = open(str(kind)+'.csv', 'a')
    while True:
        a = model.choose_action(str(obs))
        obs_, r, done = env.step(a)
        rewards = rewards + r
        print('Action: ', a, 'Reward: ', r)
        obs = obs_
        for i in range(env.nb_portcodes):
            if (a - 1) == i:
                fd.write(''.join([str(market.current_date), ',', env.portcodes[i], ',', '1', '\n']))
            else:
                fd.write(''.join([str(market.current_date), ',', env.portcodes[i], ',', '0', '\n']))
        if done:
            break
    fd.close()

