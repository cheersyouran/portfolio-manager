#encoding:utf-8
import sys
import warnings
import threading
import pandas as pd
import os
import numpy as np
warnings.filterwarnings('ignore')

project_path = '/Users/Youran/Projects/PortfolioManagement'
start_date = '2017-01-06'
train_window = 100
test_window = 1

sys.path.append(project_path)

def save_file(ind):
    def method(x):
        df = x.groupby(['tradingday','portcode']).sum().reset_index()
        size = np.unique(x['portcode']).size
        df['portweight'] = df['portweight'] / size
        return df

    tmp_path = 'result/tmp.csv'
    result_path = 'result/task2_result.csv'

    for i in ind:
        df = pd.read_csv('result/'+i + '.csv', header=None)
        df.to_csv(tmp_path, encoding="utf_8_sig", index=False, header=False, mode='a+')

    df = pd.read_csv(tmp_path, header=None)
    df.columns = ['tradingday', 'portcode', 'portweight']
    unique = df.groupby(['tradingday']).apply(func=method)
    unique.to_csv(result_path, encoding="utf_8_sig", mode='a+',index=False)

    os.remove(tmp_path)

def run_qLearning_model_beta():
    from task2.QLearningModel_beta.market import Market
    from task2.QLearningModel_beta.run import run_model
    import base

    irrank = base.load_irweek_csv()
    market = Market()
    while True:
        threads = []
        t1 = threading.Thread(target=run_model, args=('计算机', [[1, 2]], irrank, market), name='T1')
        t2 = threading.Thread(target=run_model, args=('银行', [[1, 2]], irrank, market), name='T2')
        t3 = threading.Thread(target=run_model, args=('家电', [[1, 2]], irrank, market), name='T3')

        threads.append(t1)
        threads.append(t2)
        threads.append(t3)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ind = ['计算机', '银行', '家电']
        save_file(ind)

        market.pass_a_day()


if __name__ == '__main__':
    run_qLearning_model_beta()