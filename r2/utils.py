import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r2 import config

TIME_SPACE = config.TIME_SPACE
STOCK_NUM = config.STOCK_NUM
HISTORY = config.HISTORY
ALL_HISTORY = config.ALL_HISTORY
PATH = config.PATH

# 'SecuCode','SecuAbbr','TradingDay','Close','High','Low','Avg','ChangePCT','TurnoverVolume','TurnoverValue'
def load_data(path):
    df = pd.read_csv(path, parse_dates=[2]).head(10)
    # gb = df.groupby("SecuAbbr")
    # for x in gb.groups:
    #     stock=gb.get_group(x)
    return df

def withDayAvg(df, n_day):
    df1 = df.groupby('SecuAbbr').apply(lambda x: x.set_index('TradingDay'))
    avg = df1.groupby(level=0)['Avg'].apply(lambda x: x.rolling(min_periods=1, window=n_day).mean()).reset_index(name='2_days_avg')
    merged = pd.merge(df, avg, on=['SecuAbbr', 'TradingDay'], how='left')
    return merged

df = load_data(PATH)

df = withDayAvg(df, 2)

print(df)